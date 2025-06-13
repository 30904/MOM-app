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
logging.basicConfig(level=logging.DEBUG)
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
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
def upload_audio():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        target_language = request.form.get('target_language', 'hi')  # Default to Hindi if not specified

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if file and audio_service.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)
            logger.info(f"File saved successfully: {filepath}")

            meeting_id = str(uuid.uuid4())
            meetings[meeting_id] = {
                'status': 'processing',
                'transcription': '',
                'translated': '',
                'target_language': target_language,
                'action_items': [],
                'summary': {
                    'english': '',
                    'translated': '',
                    'topic_summaries': [],
                    'key_points': []
                }
            }

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
    try:
        meeting_id = str(uuid.uuid4())
        meetings[meeting_id] = {
            'status': 'recording',
            'transcription': '',
            'translated': '',
            'target_language': 'hi',
            'action_items': [],
            'summary': {
                'english': '',
                'translated': '',
                'topic_summaries': [],
                'key_points': []
            }
        }
        audio_service.start_recording(meeting_id)
        emit('recording_started', {'meeting_id': meeting_id})
        logger.info(f"Recording started for meeting: {meeting_id}")
    except Exception as e:
        logger.error(f"Error starting recording: {str(e)}", exc_info=True)
        emit('recording_error', {'error': str(e)})

@socketio.on('stop_recording')
def handle_recording_stop(data):
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
        target_language = data.get('target_language', 'hi')  # Default to Hindi if not specified

        if not audio_chunk or not meeting_id:
            logger.warning('Missing audio chunk or meeting_id')
            socketio.emit('transcription_error', {'error': 'Missing audio chunk or meeting_id'})
            return

        logger.debug(f'Processing audio chunk for meeting {meeting_id}')

        if isinstance(audio_chunk, list):
            audio_chunk = np.array(audio_chunk)

        logger.debug(f'Audio chunk shape: {audio_chunk.shape}, type: {audio_chunk.dtype}')

        transcription_result = transcription_service.transcribe_chunk(audio_chunk, target_language=target_language)

        if not transcription_result:
            logger.debug('No transcription result')
            return

        logger.info(f'Transcription result: {transcription_result}')

        # Extract the text from the result
        english_text = transcription_result.get('english', '')
        translated_text = transcription_result.get('translated', '')
        action_items = transcription_result.get('action_items', [])
        current_summary = transcription_result.get('summary', {})

        if meeting_id not in meetings:
            meetings[meeting_id] = {
                'transcription': '',
                'translated': '',
                'target_language': target_language,
                'action_items': [],
                'summary': {
                    'english': '',
                    'translated': '',
                    'topic_summaries': [],
                    'key_points': []
                }
            }

        meetings[meeting_id]['transcription'] += ' ' + english_text
        meetings[meeting_id]['translated'] += ' ' + translated_text
        if action_items:
            meetings[meeting_id]['action_items'].extend(action_items)

        # Update summary if we have enough text
        if len(meetings[meeting_id]['transcription'].split()) > 50:
            if current_summary:
                meetings[meeting_id]['summary'] = {
                    'english': current_summary.get('english', ''),
                    'translated': current_summary.get('translated', ''),
                    'topic_summaries': current_summary.get('topic_summaries', []),
                    'key_points': current_summary.get('key_points', [])
                }

        # Prepare update data
        update_data = {
            'transcription': english_text,
            'translated': translated_text,
            'target_language': target_language
        }
        if action_items:
            update_data['action_items'] = action_items
        if current_summary:
            update_data['summary'] = {
                'english': current_summary.get('english', ''),
                'translated': current_summary.get('translated', ''),
                'target_language': target_language,
                'topic_summaries': current_summary.get('topic_summaries', []),
                'key_points': current_summary.get('key_points', [])
            }

        logger.info(f'Emitting transcription update: {update_data}')
        socketio.emit('transcription_update', update_data)

    except Exception as e:
        logger.error(f'Error processing audio chunk: {str(e)}')
        logger.error(f'Stack trace: {traceback.format_exc()}')
        socketio.emit('transcription_error', {'error': str(e)})

def get_transcription_text(transcription):
    if isinstance(transcription, str):
        return transcription
    elif isinstance(transcription, dict):
        for key in ('text', 'transcript', 'transcription'):
            if key in transcription and isinstance(transcription[key], str):
                return transcription[key]
        return str(transcription)
    elif transcription is None:
        return ''
    else:
        try:
            return str(transcription)
        except Exception:
            return ''

def process_audio_file(filepath, meeting_id):
    try:
        if meeting_id not in meetings:
            logger.warning(f"Meeting {meeting_id} not found")
            return

        logger.info(f"Starting to process audio file: {filepath}")

        # Get target language from meeting data or default to Hindi
        target_language = meetings[meeting_id].get('target_language', 'hi')
        transcription_result = transcription_service.transcribe_file(filepath, target_language=target_language)
        
        if not transcription_result:
            raise ValueError("Transcription failed - no result returned")

        english_text = transcription_result.get('english', '')
        translated_text = transcription_result.get('translated', '')
        action_items = transcription_result.get('action_items', [])
        summary = transcription_result.get('summary', {})

        if not english_text:
            raise ValueError("Transcription failed - no text generated")

        logger.info(f"Transcription completed: {english_text[:100]}...")
        logger.info(f"Translation completed: {translated_text[:100]}...")
        logger.info(f"Action items extracted: {len(action_items)} items")
        logger.info(f"Summary generated: {str(summary)[:100]}...")

        meetings[meeting_id].update({
            'status': 'completed',
            'transcription': english_text,
            'translated': translated_text,
            'target_language': target_language,
            'action_items': action_items,
            'summary': summary
        })

        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Audio file removed: {filepath}")

        chunk_size = 1000
        transcription_chunks = list(zip(
            [english_text[i:i+chunk_size] for i in range(0, len(english_text), chunk_size)],
            [translated_text[i:i+chunk_size] for i in range(0, len(translated_text), chunk_size)]
        ))

        for i, (eng_chunk, trans_chunk) in enumerate(transcription_chunks):
            update_data = {
                'transcription': eng_chunk,
                'translated': trans_chunk,
                'target_language': target_language
            }
            if i == len(transcription_chunks) - 1:
                update_data['is_final'] = True
                update_data['action_items'] = action_items
                update_data['summary'] = {
                    'english': summary.get('english', ''),
                    'translated': summary.get('translated', ''),
                    'target_language': target_language,
                    'topic_summaries': summary.get('topic_summaries', []),
                    'key_points': summary.get('key_points', [])
                }

            socketio.emit('transcription_update', update_data)
            logger.info(f"Emitted chunk {i+1}/{len(transcription_chunks)}")
            time.sleep(0.2)

        complete_data = {
            'meeting_id': meeting_id,
            'transcription': english_text,
            'translated': translated_text,
            'target_language': target_language,
            'summary': {
                'english': summary.get('english', ''),
                'translated': summary.get('translated', ''),
                'target_language': target_language,
                'topic_summaries': summary.get('topic_summaries', []),
                'key_points': summary.get('key_points', [])
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

@main.route('/supported_languages', methods=['GET'])
def get_supported_languages():
    """Get list of supported languages for translation."""
    try:
        languages = transcription_service.get_supported_languages()
        return jsonify({
            'success': True,
            'languages': languages
        })
    except Exception as e:
        logger.error(f"Error getting supported languages: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
