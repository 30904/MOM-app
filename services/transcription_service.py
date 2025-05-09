import whisper
import numpy as np
from config import Config

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
    
    def transcribe_chunk(self, audio_chunk):
        """Transcribe a chunk of audio data."""
        try:
            # Convert audio chunk to numpy array
            audio_data = np.frombuffer(audio_chunk, dtype=np.float32)
            
            # Whisper expects audio to be normalized between -1 and 1
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Transcribe the audio chunk
            result = self.model.transcribe(audio_data)
            return result["text"]
        except Exception as e:
            print(f"Error transcribing chunk: {str(e)}")
            return None 