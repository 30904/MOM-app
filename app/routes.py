from flask import Blueprint, render_template, request, jsonify, current_app
from flask_socketio import emit
from werkzeug.utils import secure_filename
import os
import uuid
import logging
from app import socketio
from services.audio_service import AudioService
from services.transcription_service import TranscriptionService
from services.nlp_service import NLPService
import numpy as np
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Changed to DEBUG for more detailed logs
logger = logging.getLogger(__name__)

main = Blueprint('main', __name__)

# Initialize services
audio_service = AudioService()
transcription_service = TranscriptionService()
nlp_service = NLPService()

# In-memory storage for development
meetings = {}

@main.route('/')
def index():
    """Render the main application page."""
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
def upload_audio():
    """Handle audio file uploads."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and audio_service.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            
            # Ensure upload directory exists
            os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Save the file
            file.save(filepath)
            logger.info(f"File saved successfully: {filepath}")
            
            # Create a new meeting record
            meeting_id = str(uuid.uuid4())
            meetings[meeting_id] = {
                'status': 'processing',
                'transcription': '',
                'action_items': [],
                'summary': ''
            }
            
            # Start processing in background
            socketio.start_background_task(
                process_audio_file,
                filepath,
                meeting_id
            )
            
            return jsonify({
                'message': 'File uploaded successfully',
                'meeting_id': meeting_id
            })
            
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        logger.error(f"Error in upload_audio: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@socketio.on('start_recording')
def handle_recording_start():
    """Start a new recording session."""
    try:
        meeting_id = str(uuid.uuid4())
        meetings[meeting_id] = {
            'status': 'recording',
            'transcription': '',
            'action_items': [],
            'summary': ''
        }
        audio_service.start_recording(meeting_id)
        emit('recording_started', {'meeting_id': meeting_id})
        logger.info(f"Recording started for meeting: {meeting_id}")
    except Exception as e:
        logger.error(f"Error starting recording: {str(e)}", exc_info=True)
        emit('recording_error', {'error': str(e)})

@socketio.on('stop_recording')
def handle_recording_stop(data):
    """Stop the current recording session."""
    try:
        meeting_id = data.get('meeting_id')
        if not meeting_id:
            raise ValueError("No meeting_id provided")
            
        frames = audio_service.stop_recording()
        
        if frames and meeting_id in meetings:
            filename = f"recording_{meeting_id}.wav"
            filepath = audio_service.save_audio(frames, filename)
            
            if filepath:
                meetings[meeting_id]['status'] = 'processing'
                socketio.start_background_task(
                    process_audio_file,
                    filepath,
                    meeting_id
                )
                logger.info(f"Recording stopped and saved: {filepath}")
        
        emit('recording_stopped', {'meeting_id': meeting_id})
    except Exception as e:
        logger.error(f"Error stopping recording: {str(e)}", exc_info=True)
        emit('recording_error', {'error': str(e)})

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Process incoming audio chunks in real-time."""
    try:
        audio_chunk = data.get('audio')
        meeting_id = data.get('meeting_id')
        sample_rate = data.get('sample_rate', 44100)
        
        if not audio_chunk or not meeting_id or meeting_id not in meetings:
            logger.warning(f"Invalid audio chunk data received: meeting_id={meeting_id}")
            return
        
        logger.debug(f"Received audio chunk for meeting {meeting_id}, length: {len(audio_chunk)}")
        
        # Convert audio chunk to numpy array if it isn't already
        if isinstance(audio_chunk, list):
            audio_chunk = np.array(audio_chunk, dtype=np.float32)
        
        # Process the audio chunk
        text = transcription_service.transcribe_chunk(audio_chunk, sample_rate=sample_rate)
        
        if text:
            logger.debug(f"Transcribed text: {text}")
            
            # Extract action items
            action_items = nlp_service.extract_action_items(text)
            logger.debug(f"Extracted action items: {action_items}")
            
            # Update in-memory storage
            if meeting_id in meetings:
                meetings[meeting_id]['transcription'] += text + ' '
                if action_items:
                    meetings[meeting_id]['action_items'].extend(action_items)
                
                # Generate interim summary if we have enough text
                if len(meetings[meeting_id]['transcription'].split()) > 50:  # After 50 words
                    interim_summary = nlp_service.generate_summary(meetings[meeting_id]['transcription'])
                    meetings[meeting_id]['summary'] = interim_summary
                    
                    # Emit the results back to the client with structured summary
                    emit('transcription_update', {
                        'text': text,
                        'action_items': action_items,
                        'summary': {
                            'main_summary': interim_summary.get('summary', interim_summary) if isinstance(interim_summary, dict) else interim_summary,
                            'topic_summaries': interim_summary.get('topic_summaries', []) if isinstance(interim_summary, dict) else [],
                            'key_points': interim_summary.get('key_points', []) if isinstance(interim_summary, dict) else []
                        }
                    })
                else:
                    # Emit just the transcription and action items
                    emit('transcription_update', {
                        'text': text,
                        'action_items': action_items
                    })
                
                logger.debug(f"Updated meeting {meeting_id} with new content")
            
    except Exception as e:
        logger.error(f"Error processing audio chunk: {str(e)}", exc_info=True)
        emit('transcription_error', {'error': str(e)})

def process_audio_file(filepath, meeting_id):
    """Process an uploaded audio file."""
    try:
        if meeting_id not in meetings:
            logger.warning(f"Meeting {meeting_id} not found")
            return
        
        logger.info(f"Starting to process audio file: {filepath}")
        
        # Transcribe the full audio file
        transcription = transcription_service.transcribe_file(filepath)
        logger.info(f"Transcription completed: {transcription[:100]}...")  # Log first 100 chars
        
        if not transcription:
            raise ValueError("Transcription failed - no text generated")
        
        # Extract action items and generate summary
        action_items = nlp_service.extract_action_items(transcription)
        logger.info(f"Action items extracted: {len(action_items)} items")
        
        summary = nlp_service.generate_summary(transcription)
        if isinstance(summary, dict):
            logger.info(f"Generated structured summary with {len(summary.get('topic_summaries', []))} topics")
        else:
            logger.info(f"Generated plain text summary: {str(summary)[:100]}...")
        
        # Update in-memory storage
        meetings[meeting_id].update({
            'status': 'completed',
            'transcription': transcription,
            'action_items': action_items,
            'summary': summary
        })
        
        # Clean up the audio file
        os.remove(filepath)
        logger.info(f"Audio file removed: {filepath}")
        
        # First, send the transcription in chunks
        chunk_size = 1000  # characters
        transcription_chunks = [transcription[i:i+chunk_size] 
                              for i in range(0, len(transcription), chunk_size)]
        
        for i, chunk in enumerate(transcription_chunks):
            socketio.emit('transcription_update', {
                'text': chunk,
                'is_final': i == len(transcription_chunks) - 1
            })
            time.sleep(0.1)  # Small delay between chunks
        
        # Then send the complete data with structured summary
        socketio.emit('processing_complete', {
            'meeting_id': meeting_id,
            'transcription': transcription,
            'summary': {
                'main_summary': summary.get('summary', summary) if isinstance(summary, dict) else summary,
                'topic_summaries': summary.get('topic_summaries', []) if isinstance(summary, dict) else [],
                'key_points': summary.get('key_points', []) if isinstance(summary, dict) else []
            },
            'action_items': action_items
        })
        
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}", exc_info=True)
        if meeting_id in meetings:
            meetings[meeting_id].update({
                'status': 'error',
                'error': str(e)
            })
        socketio.emit('processing_error', {
            'meeting_id': meeting_id,
            'error': str(e)
        }) 