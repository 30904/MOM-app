import whisper
import numpy as np
from config import Config
import soundfile as sf
import io

class TranscriptionService:
    def __init__(self):
        self.model = whisper.load_model(Config.WHISPER_MODEL)
    
    def transcribe_file(self, audio_path):
        """Transcribe an entire audio file."""
        try:
            result = self.model.transcribe(audio_path)
            return result["text"]
        except Exception as e:
            print(f"Error transcribing file: {str(e)}")
            raise
    
    def transcribe_chunk(self, audio_data, sample_rate=None):
        """Transcribe a chunk of audio data."""
        try:
            # Handle different input types
            if isinstance(audio_data, (bytes, bytearray, memoryview)):
                # Convert bytes to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
            elif isinstance(audio_data, np.ndarray):
                audio_array = audio_data
            else:
                raise ValueError(f"Unsupported audio data type: {type(audio_data)}")
            
            # Ensure we have a contiguous array in the correct format
            audio_array = np.ascontiguousarray(audio_array, dtype=np.float32)
            
            # Normalize audio (only if it's not already normalized)
            max_val = np.max(np.abs(audio_array))
            if max_val > 1.0:
                audio_array = audio_array / max_val
            
            # Resample if needed (Whisper expects 16kHz)
            if sample_rate and sample_rate != 16000:
                from scipy import signal
                audio_array = signal.resample(audio_array, 
                                           int(len(audio_array) * 16000 / sample_rate))
            
            # Ensure minimum duration (Whisper typically expects at least 30ms of audio)
            min_samples = int(16000 * 0.03)  # 30ms at 16kHz
            if len(audio_array) < min_samples:
                audio_array = np.pad(audio_array, (0, min_samples - len(audio_array)))
            
            # Transcribe the audio chunk
            result = self.model.transcribe(audio_array)
            return result["text"]
            
        except Exception as e:
            print(f"Error transcribing chunk: {str(e)}")
            import traceback
            traceback.print_exc()
            return None 