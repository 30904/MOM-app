import pyaudio
import wave
import os
import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
from config import Config
import atexit
import threading
import logging
import queue
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
import contextlib
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioQuality(Enum):
    """Audio quality levels for processing."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    sample_rate: int
    channels: int
    chunk_size: int
    format: int
    quality: AudioQuality
    
    @classmethod
    def from_quality(cls, quality: AudioQuality) -> 'AudioConfig':
        """Create audio configuration based on quality level."""
        configs = {
            AudioQuality.LOW: {
                "sample_rate": 16000,
                "chunk_size": 1024
            },
            AudioQuality.MEDIUM: {
                "sample_rate": 32000,
                "chunk_size": 2048
            },
            AudioQuality.HIGH: {
                "sample_rate": 44100,
                "chunk_size": 4096
            },
            AudioQuality.ULTRA: {
                "sample_rate": 48000,
                "chunk_size": 8192
            }
        }
        base_config = configs[quality]
        return cls(
            sample_rate=base_config["sample_rate"],
            channels=1,  # Mono for better processing
            chunk_size=base_config["chunk_size"],
            format=pyaudio.paFloat32,
            quality=quality
        )

class AudioProcessor:
    """Handles advanced audio processing operations."""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.setup_filters()
        
    def setup_filters(self):
        """Initialize all audio filters."""
        # Speech frequency range filters
        self.nyquist = self.config.sample_rate / 2
        self.speech_low = 85  # Hz
        self.speech_high = 3500  # Hz
        
        # Multiple band-pass filters for different frequency ranges
        self.filters = {
            "prefilter": self._create_butterworth_filter(20, 4000, 'band', 3),
            "speech_main": self._create_butterworth_filter(self.speech_low, self.speech_high, 'band', 5),
            "noise_reduction": self._create_butterworth_filter(self.speech_low, self.speech_high, 'band', 2),
        }
        
        # Spectral gates
        self.spectral_floor = -65  # dB
        self.spectral_gate_threshold = -45  # dB
        
    def _create_butterworth_filter(self, lowcut: float, highcut: float, 
                                 filter_type: str, order: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create a Butterworth filter with specified parameters."""
        nyquist = self.config.sample_rate / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        return signal.butter(order, [low, high], btype=filter_type, analog=False)
    
    def apply_filters(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply all filters in sequence."""
        # Ensure input is a numpy array
        audio_data = np.asarray(audio_data)
        
        # Apply pre-filtering
        audio = signal.filtfilt(*self.filters["prefilter"], audio_data)
        
        # Apply main speech filter
        audio = signal.filtfilt(*self.filters["speech_main"], audio)
        
        # Apply noise reduction filter
        audio = signal.filtfilt(*self.filters["noise_reduction"], audio)
        
        return audio
    
    def spectral_gate(self, audio_data: np.ndarray, noise_profile: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply spectral gating for noise reduction."""
        # Ensure inputs are numpy arrays
        audio_data = np.asarray(audio_data)
        if noise_profile is not None:
            noise_profile = np.asarray(noise_profile)
            # Ensure noise profile length matches audio data
            if len(noise_profile) > len(audio_data):
                noise_profile = noise_profile[:len(audio_data)]
            elif len(noise_profile) < len(audio_data):
                # Pad noise profile if it's shorter
                noise_profile = np.pad(noise_profile, 
                                     (0, len(audio_data) - len(noise_profile)),
                                     mode='wrap')
        
        # Compute FFT
        spectrum = fft(audio_data)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        
        # Apply spectral gating
        if noise_profile is not None:
            noise_spectrum = fft(noise_profile)
            noise_magnitude = np.abs(noise_spectrum)
            # Dynamic threshold based on noise profile
            gate = magnitude > (noise_magnitude * 2)
        else:
            # Static threshold when no noise profile is available
            gate = magnitude > np.power(10, self.spectral_gate_threshold / 20)
        
        # Apply gate and reconstruct signal
        gated_spectrum = magnitude * gate * np.exp(1j * phase)
        return np.real(ifft(gated_spectrum))
    
    def compress_dynamic_range(self, audio_data: np.ndarray, 
                             threshold: float = -20, 
                             ratio: float = 4) -> np.ndarray:
        """Apply dynamic range compression."""
        # Ensure input is a numpy array
        audio_data = np.asarray(audio_data)
        
        # Convert to dB
        db = 20 * np.log10(np.abs(audio_data) + 1e-10)
        
        # Apply compression
        mask = db > threshold
        db[mask] = threshold + (db[mask] - threshold) / ratio
        
        # Convert back to linear
        return np.sign(audio_data) * np.power(10, db / 20)

class AudioService:
    """Main audio service with comprehensive audio processing capabilities."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AudioService, cls).__new__(cls)
                atexit.register(cls._instance.cleanup_resources)
            return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        # Initialize configuration
        self.config = AudioConfig.from_quality(AudioQuality.HIGH)
        self.processor = AudioProcessor(self.config)
        
        # Initialize PyAudio
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        
        # Threading and queues
        self.audio_queue = queue.Queue(maxsize=100)
        self.processing_thread = None
        self.should_process = threading.Event()
        
        # State management
        self.recording = False
        self.frames = []
        self.noise_profile = None
        self.noise_samples_count = 0
        self.required_noise_samples = 50
        
        # Performance monitoring
        self.processing_stats = {
            "total_processed": 0,
            "avg_processing_time": 0,
            "dropped_frames": 0
        }
        
        self._initialized = True
        logger.info("AudioService initialized with configuration: %s", self.config)
    
    def start_recording(self, meeting_id: str) -> None:
        """Start recording with comprehensive error handling and monitoring."""
        try:
            if self.recording:
                logger.warning("Recording already in progress")
                return
            
            self.recording = True
            self.frames = []
            self.noise_profile = None
            self.noise_samples_count = 0
            self.should_process.set()
            
            # Start processing thread
            self.processing_thread = threading.Thread(
                target=self._process_audio_queue,
                daemon=True
            )
            self.processing_thread.start()
            
            # Configure and start audio stream
            self._setup_audio_stream()
            logger.info("Recording started for meeting: %s", meeting_id)
            
        except Exception as e:
            self.recording = False
            logger.error("Failed to start recording: %s", str(e))
            raise
    
    def _setup_audio_stream(self) -> None:
        """Set up the audio input stream with error handling."""
        try:
            # Get default input device
            device_info = self.pyaudio.get_default_input_device_info()
            
            # Validate device capabilities
            self._validate_device(device_info)
            
            # Open stream
            self.stream = self.pyaudio.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=device_info['index'],
                frames_per_buffer=self.config.chunk_size,
                stream_callback=self._audio_callback
            )
            
            if not self.stream.is_active():
                raise RuntimeError("Failed to activate audio stream")
            
        except Exception as e:
            logger.error("Failed to setup audio stream: %s", str(e))
            self.cleanup_resources()
            raise
    
    def _validate_device(self, device_info: Dict) -> None:
        """Validate audio device capabilities."""
        required_sample_rates = [16000, 32000, 44100, 48000]
        supported_rates = [
            rate for rate in required_sample_rates
            if self.pyaudio.is_format_supported(
                rate,
                input_device=device_info['index'],
                input_channels=self.config.channels,
                input_format=self.config.format
            )
        ]
        
        if not supported_rates:
            raise ValueError("No supported sample rates found for device")
        
        if self.config.sample_rate not in supported_rates:
            # Fall back to highest supported rate
            self.config.sample_rate = max(supported_rates)
            logger.warning(
                "Adjusted sample rate to %d Hz due to device limitations",
                self.config.sample_rate
            )
    
    def _audio_callback(self, in_data, frame_count, time_info, status) -> Tuple[bytes, int]:
        """Handle incoming audio data with comprehensive processing."""
        try:
            if status:
                logger.warning("Audio callback status: %s", status)
            
            if self.recording:
                # Add to processing queue
                try:
                    self.audio_queue.put_nowait(in_data)
                except queue.Full:
                    logger.warning("Audio queue full, dropping frame")
                    self.processing_stats["dropped_frames"] += 1
            
            return (in_data, pyaudio.paContinue)
            
        except Exception as e:
            logger.error("Error in audio callback: %s", str(e))
            return (in_data, pyaudio.paAbort)
    
    def _process_audio_queue(self) -> None:
        """Process audio data from the queue."""
        while self.should_process.is_set():
            try:
                # Get audio data from queue
                audio_data = self.audio_queue.get(timeout=0.1)
                start_time = time.time()
                
                # Convert to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
                audio_array = np.asarray(audio_array)  # Ensure it's a proper numpy array
                
                # Collect noise profile
                if self.noise_samples_count < self.required_noise_samples:
                    if self.noise_profile is None:
                        self.noise_profile = audio_array
                    else:
                        # Ensure noise_profile is a numpy array
                        self.noise_profile = np.asarray(self.noise_profile)
                        self.noise_profile = np.concatenate([self.noise_profile, audio_array])
                    self.noise_samples_count += 1
                    continue
                
                # Process audio
                processed_audio = self._process_audio_chunk(audio_array)
                
                # Ensure processed_audio is a numpy array
                processed_audio = np.asarray(processed_audio)
                
                # Add to frames
                self.frames.append(processed_audio.tobytes())
                
                # Update processing stats
                processing_time = time.time() - start_time
                self.processing_stats["total_processed"] += 1
                self.processing_stats["avg_processing_time"] = (
                    self.processing_stats["avg_processing_time"] * 
                    (self.processing_stats["total_processed"] - 1) +
                    processing_time
                ) / self.processing_stats["total_processed"]
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Error processing audio: %s", str(e))
                logger.exception("Full traceback:")
    
    def _process_audio_chunk(self, audio_array: np.ndarray) -> np.ndarray:
        """Apply comprehensive audio processing to a chunk of audio."""
        # Convert generator to array if needed
        if hasattr(audio_array, '__iter__') and not isinstance(audio_array, (np.ndarray, list)):
            audio_array = np.array(list(audio_array))
            
        # Apply filters
        filtered_audio = self.processor.apply_filters(audio_array)
        
        # Apply spectral gating if we have a noise profile
        if self.noise_profile is not None:
            # Convert noise profile to array if needed
            if hasattr(self.noise_profile, '__iter__') and not isinstance(self.noise_profile, (np.ndarray, list)):
                self.noise_profile = np.array(list(self.noise_profile))
            filtered_audio = self.processor.spectral_gate(
                filtered_audio,
                self.noise_profile[:len(filtered_audio)]
            )
        
        # Apply compression
        compressed_audio = self.processor.compress_dynamic_range(filtered_audio)
        
        # Normalize
        if np.max(np.abs(compressed_audio)) > 0:
            compressed_audio = compressed_audio / np.max(np.abs(compressed_audio))
        
        return compressed_audio
    
    def stop_recording(self) -> Optional[List[bytes]]:
        """Stop recording and cleanup resources."""
        if not self.recording:
            return None
        
        try:
            # Stop processing
            self.should_process.clear()
            if self.processing_thread:
                self.processing_thread.join(timeout=2.0)
            
            # Stop stream
            if self.stream:
                with contextlib.suppress(Exception):
                    if self.stream.is_active():
                        self.stream.stop_stream()
                    self.stream.close()
                self.stream = None
            
            self.recording = False
            logger.info("Recording stopped. Processing stats: %s", self.processing_stats)
            
            return self.frames
            
        except Exception as e:
            logger.error("Error stopping recording: %s", str(e))
            raise
        finally:
            self.frames = []
            self.noise_profile = None
    
    def cleanup_resources(self) -> None:
        """Comprehensive cleanup of all resources."""
        try:
            # Stop recording if active
            if self.recording:
                self.stop_recording()
            
            # Clean up PyAudio
            if hasattr(self, 'pyaudio'):
                with contextlib.suppress(Exception):
                    self.pyaudio.terminate()
                
            # Clear all buffers
            self.frames = []
            self.noise_profile = None
            
            # Clear queues
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            logger.info("AudioService resources cleaned up successfully")
            
        except Exception as e:
            logger.error("Error during resource cleanup: %s", str(e))
    
    def save_audio(self, frames: List[bytes], filename: str) -> Optional[str]:
        """Save processed audio to file with error handling."""
        if not frames:
            logger.warning("No frames to save")
            return None
        
        try:
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(self.config.channels)
                wf.setsampwidth(self.pyaudio.get_sample_size(self.config.format))
                wf.setframerate(self.config.sample_rate)
                wf.writeframes(b''.join(frames))
            
            logger.info("Audio saved successfully to: %s", filepath)
            return filepath
            
        except Exception as e:
            logger.error("Failed to save audio file: %s", str(e))
            if os.path.exists(filepath):
                os.remove(filepath)
            raise
    
    def allowed_file(self, filename: str) -> bool:
        """Check if file type is allowed."""
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_AUDIO_EXTENSIONS 