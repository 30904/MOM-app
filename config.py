import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-please-change-in-production'
    
    # MongoDB configuration
    MONGODB_URI = os.environ.get('MONGODB_URI') or 'mongodb://localhost:27017/'
    DB_NAME = os.environ.get('DB_NAME') or 'mom_db'
    
    # Audio configuration
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'm4a', 'ogg'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Whisper configuration
    WHISPER_MODEL = "base"  # Can be tiny, base, small, medium, or large
    
    # Socket.IO configuration
    SOCKETIO_ASYNC_MODE = 'eventlet'
    
    SOCKETIO_MESSAGE_QUEUE = None  # Use in-memory queue for development
    
    @staticmethod
    def init_app(app):
        # Create upload folder if it doesn't exist
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        
        # Ensure the upload folder is writable
        if not os.access(Config.UPLOAD_FOLDER, os.W_OK):
            raise RuntimeError(f"Upload folder {Config.UPLOAD_FOLDER} is not writable") 