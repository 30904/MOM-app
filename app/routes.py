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

# Configure logging
logging.basicConfig(level=logging.INFO)
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
            logger.warning("Invalid audio chunk data received")
            return
        
        # Convert audio chunk to numpy array if it isn't already
        if isinstance(audio_chunk, list):
            audio_chunk = np.array(audio_chunk, dtype=np.float32)
        
        # Process the audio chunk
        text = transcription_service.transcribe_chunk(audio_chunk, sample_rate=sample_rate)
        if text:
            # Extract action items
            action_items = nlp_service.extract_action_items(text)
            
            # Update in-memory storage
            meetings[meeting_id]['transcription'] += text + ' '
            meetings[meeting_id]['action_items'].extend(action_items)
            
            # Emit the results back to the client
            emit('transcription_update', {
                'text': text,
                'action_items': action_items
            })
            logger.debug(f"Processed audio chunk for meeting {meeting_id}: {text[:50]}...")
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
        logger.info("Transcription completed")
        
        # Extract action items and generate summary
        action_items = nlp_service.extract_action_items(transcription)
        logger.info("Action items extracted")
        
        summary = nlp_service.generate_summary(transcription)
        logger.info("Summary generated")
        
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
        
        socketio.emit('processing_complete', {
            'meeting_id': meeting_id,
            'summary': summary,
            'transcription': transcription,
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