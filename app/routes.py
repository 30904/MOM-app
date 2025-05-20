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
import traceback

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
    try:
        logger.debug('Received audio chunk')
        audio_chunk = data.get('audio')
        meeting_id = data.get('meeting_id')
        
        if not audio_chunk or not meeting_id:
            logger.warning('Missing audio chunk or meeting_id')
            socketio.emit('transcription_error', {'error': 'Missing audio chunk or meeting_id'})
            return
        
        logger.debug(f'Processing audio chunk for meeting {meeting_id}')
        
        # Convert audio chunk to numpy array if it's a list
        if isinstance(audio_chunk, list):
            audio_chunk = np.array(audio_chunk)
        
        logger.debug(f'Audio chunk shape: {audio_chunk.shape}, type: {audio_chunk.dtype}')
        
        # Process the audio chunk
        transcription_result = transcription_service.transcribe_chunk(audio_chunk)
        
        if not transcription_result:
            logger.debug('No transcription result')
            return
        
        logger.info(f'Transcription result: {transcription_result}')
        
        # Extract action items
        action_items = nlp_service.extract_action_items(transcription_result)
        logger.info(f'Extracted action items: {action_items}')
        
        # Update in-memory storage
        if meeting_id not in meetings:
            meetings[meeting_id] = {
                'transcription': '',
                'action_items': [],
                'summary': ''
            }
        
        # Update transcription
        meetings[meeting_id]['transcription'] += ' ' + transcription_result
        meetings[meeting_id]['action_items'].extend(action_items)
        
        # Generate interim summary if we have enough text
        if len(meetings[meeting_id]['transcription'].split()) > 50:
            summary = nlp_service.generate_summary(meetings[meeting_id]['transcription'])
            meetings[meeting_id]['summary'] = summary
            logger.info(f'Generated summary: {summary}')
        else:
            summary = None
            logger.debug('Not enough text for summary generation')
        
        # Emit the update back to the client with the correct data structure
        update_data = {
            'transcription': transcription_result,  # Changed from 'text' to 'transcription'
            'action_items': action_items,
            'summary': summary
        }
        
        logger.info(f'Emitting transcription update: {update_data}')
        socketio.emit('transcription_update', update_data)
        
    except Exception as e:
        logger.error(f'Error processing audio chunk: {str(e)}')
        logger.error(f'Stack trace: {traceback.format_exc()}')
        socketio.emit('transcription_error', {'error': str(e)})

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
        
        # Send each chunk with proper data structure
        for i, chunk in enumerate(transcription_chunks):
            update_data = {
                'transcription': chunk,  # Changed from 'text' to 'transcription'
                'is_final': i == len(transcription_chunks) - 1,
                'action_items': action_items if i == len(transcription_chunks) - 1 else [],
                'summary': summary if i == len(transcription_chunks) - 1 else None
            }
            socketio.emit('transcription_update', update_data)
            logger.info(f"Emitted chunk {i+1}/{len(transcription_chunks)}")
            time.sleep(0.2)  # Increased delay between chunks to prevent disconnects
        
        # Send the final complete data
        complete_data = {
            'meeting_id': meeting_id,
            'transcription': transcription,
            'summary': {
                'main_summary': summary.get('summary', summary) if isinstance(summary, dict) else summary,
                'topic_summaries': summary.get('topic_summaries', []) if isinstance(summary, dict) else [],
                'key_points': summary.get('key_points', []) if isinstance(summary, dict) else []
            },
            'action_items': action_items
        }
        logger.info(f"Emitting complete data")
        socketio.emit('processing_complete', complete_data)
        
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        if meeting_id in meetings:
            meetings[meeting_id].update({
                'status': 'error',
                'error': str(e)
            })
        socketio.emit('processing_error', {
            'meeting_id': meeting_id,
            'error': str(e)
        }) 