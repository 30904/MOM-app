import pyaudio
import wave
import os
import numpy as np
from scipy import signal
import atexit
import threading
import logging
import queue
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum
import contextlib
from pydub import AudioSegment  # For audio file conversion
import sys

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

if not logger.hasHandlers():
    logger.addHandler(console_handler)

class AudioQuality(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

@dataclass
class AudioConfig:
    sample_rate: int
    channels: int
    chunk_size: int
    format: int
    quality: AudioQuality

    @classmethod
    def from_quality(cls, quality: AudioQuality) -> 'AudioConfig':
        configs = {
            AudioQuality.LOW: {"sample_rate": 16000, "chunk_size": 1024},
            AudioQuality.MEDIUM: {"sample_rate": 32000, "chunk_size": 2048},
            AudioQuality.HIGH: {"sample_rate": 44100, "chunk_size": 4096},
            AudioQuality.ULTRA: {"sample_rate": 48000, "chunk_size": 8192}
        }
        base_config = configs[quality]
        return cls(
            sample_rate=base_config["sample_rate"],
            channels=1,  # MONO
            chunk_size=base_config["chunk_size"],
            format=pyaudio.paInt16,
            quality=quality
        )

class AudioProcessor:
    def __init__(self, config: AudioConfig):
        self.config = config
        self.setup_filters()

    def setup_filters(self):
        self.speech_low = 85
        self.speech_high = 3500
        self.filters = {
            "prefilter": self._create_filter(20, 4000, 3),
            "speech_main": self._create_filter(self.speech_low, self.speech_high, 5),
            "noise_reduction": self._create_filter(self.speech_low, self.speech_high, 2)
        }

    def _create_filter(self, lowcut, highcut, order):
        nyq = 0.5 * self.config.sample_rate
        low = lowcut / nyq
        high = highcut / nyq
        return signal.butter(order, [low, high], btype='band')

    def apply_filters(self, audio: np.ndarray) -> np.ndarray:
        audio = signal.filtfilt(*self.filters["prefilter"], audio)
        audio = signal.filtfilt(*self.filters["speech_main"], audio)
        audio = signal.filtfilt(*self.filters["noise_reduction"], audio)
        return audio

    def compress_dynamic_range(self, audio: np.ndarray, threshold=-20, ratio=4) -> np.ndarray:
        db = 20 * np.log10(np.abs(audio) + 1e-10)
        mask = db > threshold
        db[mask] = threshold + (db[mask] - threshold) / ratio
        return np.sign(audio) * (10 ** (db / 20))

class AudioService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                atexit.register(cls._instance.cleanup_resources)
            return cls._instance

    def __init__(self):
        logger.info("Initializing AudioService")
        if hasattr(self, '_initialized'):
            return
        self.config = AudioConfig.from_quality(AudioQuality.HIGH)
        self.processor = AudioProcessor(self.config)
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        self.audio_queue = queue.Queue(maxsize=100)
        self.should_process = threading.Event()
        self.processing_thread = None
        self.recording = False
        self.frames = []
        self.noise_profile = None
        self.noise_samples_count = 0
        self.required_noise_samples = 50
        self._initialized = True

    # --- Live Recording Methods ---

    def start_recording(self, meeting_id: str):
        if self.recording:
            logger.warning("Already recording")
            return

        self.recording = True
        self.frames.clear()
        self.noise_profile = None
        self.noise_samples_count = 0
        self.should_process.set()

        self.processing_thread = threading.Thread(target=self._process_audio_queue, daemon=True)
        self.processing_thread.start()

        self._setup_audio_stream()

        logger.info(f"Recording started for meeting: {meeting_id}")

    def _setup_audio_stream(self):
        device_info = self.pyaudio.get_default_input_device_info()
        self.stream = self.pyaudio.open(
            format=self.config.format,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=self.config.chunk_size,
            stream_callback=self._audio_callback
        )
        if not self.stream.is_active():
            raise RuntimeError("Stream did not start properly")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self.recording:
            try:
                self.audio_queue.put_nowait(in_data)
            except queue.Full:
                logger.warning("Dropped audio frame")
        return (in_data, pyaudio.paContinue)

    def _process_audio_queue(self):
        while self.should_process.is_set():
            try:
                in_data = self.audio_queue.get(timeout=0.1)
                audio_array = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0

                if self.noise_samples_count < self.required_noise_samples:
                    self.noise_profile = audio_array if self.noise_profile is None else np.concatenate([self.noise_profile, audio_array])
                    self.noise_samples_count += 1
                    continue

                processed = self._process_audio_chunk(audio_array)
                self.frames.append((processed * 32767).astype(np.int16).tobytes())

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audio processing error: {e}")

    def _process_audio_chunk(self, audio: np.ndarray) -> np.ndarray:
        filtered = self.processor.apply_filters(audio)

        # Optional spectral gating if needed
        # if self.noise_profile is not None:
        #     filtered = self.processor.spectral_gate(filtered, self.noise_profile[:len(filtered)])

        compressed = self.processor.compress_dynamic_range(filtered)

        peak = np.max(np.abs(compressed))
        if peak > 1e-4:
            compressed = compressed / peak

        return compressed

    def stop_recording(self):
        self.should_process.clear()
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        if self.stream:
            with contextlib.suppress(Exception):
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
            self.stream = None
        self.recording = False
        logger.info("Recording stopped")
        return self.frames

    def save_audio(self, frames: List[bytes], filename: str) -> Optional[str]:
        if not frames:
            logger.warning("No audio frames to save")
            return None

        filepath = os.path.join("recordings", filename)
        os.makedirs("recordings", exist_ok=True)

        try:
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(self.config.channels)
                wf.setsampwidth(self.pyaudio.get_sample_size(self.config.format))
                wf.setframerate(self.config.sample_rate)
                wf.writeframes(b''.join(frames))
            logger.info(f"Audio saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return None

    def cleanup_resources(self):
        if self.recording:
            self.stop_recording()
        if self.pyaudio:
            self.pyaudio.terminate()
        self.frames.clear()
        while not self.audio_queue.empty():
            with contextlib.suppress(queue.Empty):
                self.audio_queue.get_nowait()

    def allowed_file(self, filename: str) -> bool:
        """Check if file type is allowed."""
        from config import Config  # Assuming Config.ALLOWED_AUDIO_EXTENSIONS exists
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_AUDIO_EXTENSIONS

    # --- File Loading / Conversion Methods ---

    def load_audio(self, file_path: str) -> str:
        """
        Takes input audio file path.
        If needed, converts audio to supported format (e.g., WAV).
        Returns processed file path (same or converted).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # For Whisper or similar, mp3 or wav are supported, so convert only if needed.
        # Here just return path for simplicity.
        return file_path

    def convert_to_wav(self, file_path: str) -> str:
        """
        Convert input audio file to WAV format if not already WAV.
        Returns the WAV file path.
        """
        wav_path = os.path.splitext(file_path)[0] + ".wav"
        if not os.path.exists(wav_path):
            audio = AudioSegment.from_file(file_path)
            audio.export(wav_path, format="wav")
        return wav_path

    def preprocess_audio(self, audio_data, sample_rate=None):
        """
        Preprocess audio data for transcription
        """
        try:
            if isinstance(audio_data, (bytes, bytearray, memoryview)):
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
            elif isinstance(audio_data, np.ndarray):
                audio_array = audio_data
            else:
                raise ValueError(f"Unsupported audio data type: {type(audio_data)}")

            audio_array = np.ascontiguousarray(audio_array, dtype=np.float32)

            if np.isnan(audio_array).any():
                raise ValueError("Audio array contains NaN values")
            if np.isinf(audio_array).any():
                raise ValueError("Audio array contains infinite values")

            # Normalize audio
            max_val = np.max(np.abs(audio_array))
            if max_val > 1.0:
                audio_array = audio_array / max_val

            # Resample to 16kHz if needed
            if sample_rate and sample_rate != 16000:
                audio_array = signal.resample(audio_array, int(len(audio_array) * 16000 / sample_rate))

            # Ensure minimum length (30ms)
            min_samples = int(16000 * 0.03)
            if len(audio_array) < min_samples:
                audio_array = np.pad(audio_array, (0, min_samples - len(audio_array)))

            return audio_array
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {str(e)}")
            raise
