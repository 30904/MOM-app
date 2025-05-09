from flask import Blueprint, render_template, request, jsonify
from flask_socketio import emit
from werkzeug.utils import secure_filename
import os
from app import socketio, mongo
from services.audio_service import AudioService
from services.transcription_service import TranscriptionService
from services.nlp_service import NLPService
from services.storage_service import StorageService

main = Blueprint('main', __name__)

# Initialize services
audio_service = AudioService()
transcription_service = TranscriptionService()
nlp_service = NLPService()
storage_service = StorageService(mongo)

@main.route('/')
def index():
    """Render the main application page."""
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
def upload_audio():
    """Handle audio file uploads."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and audio_service.allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the audio file
        meeting_id = storage_service.create_meeting({'status': 'processing'})
        socketio.start_background_task(
            process_audio_file,
            filepath,
            meeting_id
        )
        
        return jsonify({
            'message': 'File uploaded successfully',
            'meeting_id': str(meeting_id)
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@socketio.on('start_recording')
def handle_recording_start():
    """Start a new recording session."""
    meeting_id = storage_service.create_meeting({'status': 'recording'})
    audio_service.start_recording(meeting_id)
    emit('recording_started', {'meeting_id': str(meeting_id)})

@socketio.on('stop_recording')
def handle_recording_stop(data):
    """Stop the current recording session."""
    meeting_id = data.get('meeting_id')
    audio_service.stop_recording()
    storage_service.update_meeting(meeting_id, {'status': 'completed'})
    emit('recording_stopped', {'meeting_id': meeting_id})

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Process incoming audio chunks in real-time."""
    audio_chunk = data.get('audio')
    meeting_id = data.get('meeting_id')
    
    # Process the audio chunk
    text = transcription_service.transcribe_chunk(audio_chunk)
    if text:
        # Extract action items and update transcription
        action_items = nlp_service.extract_action_items(text)
        storage_service.update_transcription(meeting_id, text, action_items)
        
        # Emit the results back to the client
        emit('transcription_update', {
            'text': text,
            'action_items': action_items
        })

def process_audio_file(filepath, meeting_id):
    """Process an uploaded audio file."""
    try:
        # Transcribe the full audio file
        transcription = transcription_service.transcribe_file(filepath)
        
        # Extract action items and generate summary
        action_items = nlp_service.extract_action_items(transcription)
        summary = nlp_service.generate_summary(transcription)
        
        # Store the results
        storage_service.update_meeting(meeting_id, {
            'status': 'completed',
            'transcription': transcription,
            'action_items': action_items,
            'summary': summary
        })
        
        # Clean up the audio file
        os.remove(filepath)
        
        socketio.emit('processing_complete', {
            'meeting_id': str(meeting_id),
            'summary': summary
        })
        
    except Exception as e:
        storage_service.update_meeting(meeting_id, {
            'status': 'error',
            'error': str(e)
        })
        socketio.emit('processing_error', {
            'meeting_id': str(meeting_id),
            'error': str(e)
        }) 