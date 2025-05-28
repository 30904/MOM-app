import whisper
import numpy as np
from config import Config
import soundfile as sf
import io
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TranscriptionService:
    def __init__(self):
        try:
            logger.info(f"Initializing Whisper model with configuration: {Config.WHISPER_MODEL}")
            self.model = whisper.load_model(Config.WHISPER_MODEL)
            logger.info("Whisper model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {str(e)}")
            raise
    
    def transcribe_file(self, audio_path):
        """Transcribe an entire audio file."""
        try:
            logger.info(f"Starting transcription of file: {audio_path}")
            
            # Verify file exists and is readable
            try:
                with open(audio_path, 'rb') as f:
                    pass
            except Exception as e:
                logger.error(f"Cannot read audio file {audio_path}: {str(e)}")
                raise ValueError(f"Cannot read audio file: {str(e)}")
            
            # Perform transcription
            result = self.model.transcribe(audio_path)
            
            if not result or "text" not in result:
                logger.error("Transcription result is invalid")
                raise ValueError("Invalid transcription result")
                
            logger.info("File transcription completed successfully")
            return result["text"]
            
        except Exception as e:
            logger.error(f"Error transcribing file: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def transcribe_chunk(self, audio_data, sample_rate=None):
        """Transcribe a chunk of audio data."""
        try:
            logger.info(f"Starting transcription of audio chunk, sample rate: {sample_rate}")
            logger.debug(f"Audio data type: {type(audio_data)}, length: {len(audio_data) if hasattr(audio_data, '__len__') else 'unknown'}")
            
            # Handle different input types
            if isinstance(audio_data, (bytes, bytearray, memoryview)):
                logger.debug("Converting bytes to numpy array")
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
            elif isinstance(audio_data, np.ndarray):
                audio_array = audio_data
            else:
                error_msg = f"Unsupported audio data type: {type(audio_data)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Log array properties
            logger.debug(f"Audio array shape: {audio_array.shape}, dtype: {audio_array.dtype}")
            
            # Ensure we have a contiguous array in the correct format
            audio_array = np.ascontiguousarray(audio_array, dtype=np.float32)
            
            # Check for invalid values
            if np.isnan(audio_array).any():
                logger.error("Audio array contains NaN values")
                raise ValueError("Audio array contains NaN values")
            
            if np.isinf(audio_array).any():
                logger.error("Audio array contains infinite values")
                raise ValueError("Audio array contains infinite values")
            
            # Normalize audio (only if it's not already normalized)
            max_val = np.max(np.abs(audio_array))
            if max_val > 1.0:
                logger.debug(f"Normalizing audio array with max value: {max_val}")
                audio_array = audio_array / max_val
            
            # Resample if needed (Whisper expects 16kHz)
            if sample_rate and sample_rate != 16000:
                logger.debug(f"Resampling audio from {sample_rate}Hz to 16000Hz")
                from scipy import signal
                audio_array = signal.resample(audio_array, 
                                           int(len(audio_array) * 16000 / sample_rate))
            
            # Ensure minimum duration (Whisper typically expects at least 30ms of audio)
            min_samples = int(16000 * 0.03)  # 30ms at 16kHz
            if len(audio_array) < min_samples:
                logger.debug(f"Padding audio array to minimum length: {min_samples} samples")
                audio_array = np.pad(audio_array, (0, min_samples - len(audio_array)))
            
            # Log final array properties before transcription
            logger.debug(f"Final audio array shape: {audio_array.shape}, min: {np.min(audio_array)}, max: {np.max(audio_array)}")
            
            # Transcribe the audio chunk
            result = self.model.transcribe(audio_array)
            
            if not result or "text" not in result:
                logger.error("Transcription result is invalid")
                raise ValueError("Invalid transcription result")
            
            logger.info("Chunk transcription completed successfully")
            return result["text"]
            
        except Exception as e:
            logger.error(f"Error transcribing chunk: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None 