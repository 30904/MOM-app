import pyaudio
import wave
import os
from config import Config

class AudioService:
    def __init__(self):
        self.format = pyaudio.paFloat32
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.recording = False
        self.frames = []
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
    
    def allowed_file(self, filename):
        """Check if the file extension is allowed."""
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_AUDIO_EXTENSIONS
    
    def start_recording(self, meeting_id):
        """Start recording audio from the microphone."""
        if self.recording:
            return
        
        self.recording = True
        self.frames = []
        self.stream = self.pyaudio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=self._audio_callback
        )
        self.stream.start_stream()
    
    def stop_recording(self):
        """Stop recording and save the audio file."""
        if not self.recording:
            return
        
        self.recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        self.stream = None
        return self.frames
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream."""
        if self.recording:
            self.frames.append(in_data)
        return (in_data, pyaudio.paContinue)
    
    def save_audio(self, frames, filename):
        """Save recorded audio to a WAV file."""
        if not frames:
            return None
        
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.pyaudio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
        
        return filepath
    
    def __del__(self):
        """Cleanup PyAudio resources."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pyaudio.terminate() 