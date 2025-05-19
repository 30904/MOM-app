import whisper
import numpy as np
from config import Config
import soundfile as sf
import io
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TranscriptionService:
    def __init__(self):
        try:
            logger.info(f"Initializing Whisper model with configuration: {Config.WHISPER_MODEL}")
            self.model = whisper.load_model(Config.WHISPER_MODEL)
            self.buffer = np.array([], dtype=np.float32)  # Add buffer for accumulating audio
            self.last_transcription_time = time.time()
            self.min_chunk_duration = 2.0  # Minimum duration in seconds before processing
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
            current_time = time.time()
            logger.debug(f"Starting transcription of audio chunk, sample rate: {sample_rate}")
            
            # Handle different input types
            if isinstance(audio_data, (bytes, bytearray, memoryview)):
                logger.debug("Converting bytes to numpy array")
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
            elif isinstance(audio_data, np.ndarray):
                audio_array = audio_data
            elif isinstance(audio_data, list):
                logger.debug("Converting list to numpy array")
                audio_array = np.array(audio_data, dtype=np.float32)
            else:
                error_msg = f"Unsupported audio data type: {type(audio_data)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Ensure we have a contiguous array in the correct format
            audio_array = np.ascontiguousarray(audio_array, dtype=np.float32)
            
            # Add to buffer
            self.buffer = np.concatenate([self.buffer, audio_array])
            
            # Calculate buffer duration in seconds (assuming 16kHz sample rate)
            buffer_duration = len(self.buffer) / 16000
            time_since_last = current_time - self.last_transcription_time
            
            logger.debug(f"Buffer duration: {buffer_duration}s, Time since last: {time_since_last}s")
            
            # Only process if we have enough audio data
            if buffer_duration >= self.min_chunk_duration and time_since_last >= 1.0:
                logger.debug("Processing accumulated audio buffer")
                
                # Normalize buffer
                max_val = np.max(np.abs(self.buffer))
                if max_val > 1.0:
                    self.buffer = self.buffer / max_val
                
                # Resample if needed (Whisper expects 16kHz)
                if sample_rate and sample_rate != 16000:
                    logger.debug(f"Resampling audio from {sample_rate}Hz to 16000Hz")
                    from scipy import signal
                    self.buffer = signal.resample(self.buffer, 
                                                int(len(self.buffer) * 16000 / sample_rate))
                
                # Transcribe the buffer
                result = self.model.transcribe(self.buffer)
                
                # Clear the buffer and update last transcription time
                self.buffer = np.array([], dtype=np.float32)
                self.last_transcription_time = current_time
                
                if result and "text" in result and result["text"].strip():
                    logger.info(f"Transcribed text: {result['text']}")
                    return result["text"]
            
            return None
            
        except Exception as e:
            logger.error(f"Error transcribing chunk: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None 