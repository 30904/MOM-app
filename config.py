import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Base directory
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here')
    
    # MongoDB configuration
    MONGODB_URI = os.environ.get('MONGODB_URI') or 'mongodb://localhost:27017/'
    DB_NAME = os.environ.get('DB_NAME') or 'mom_db'
    
    # Audio configuration
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}
    MAX_AUDIO_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_CONTENT_LENGTH = MAX_AUDIO_SIZE
    
    # Whisper configuration - using smaller model for CPU
    WHISPER_MODEL = "tiny"  # Using tiny model for better CPU performance
    MODEL_CACHE_DIR = os.path.join(BASE_DIR, "model_cache")
    
    # Model configuration
    USE_CPU = True  # Force CPU usage
    MODEL_PRECISION = "float32"  # Use FP32 for CPU
    
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
        
        # Ensure model cache directory exists
        os.makedirs(Config.MODEL_CACHE_DIR, exist_ok=True) 