import whisper 
import numpy as np
import logging
from transformers import pipeline
from scipy import signal
import torch
import textwrap
from config import Config
from services.nlp_service import NLPService
import sys
from .audio_service import AudioService

# Configure logging with UTF-8 support
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
console_handler.setFormatter(console_formatter)

if not logger.hasHandlers():
    logger.addHandler(console_handler)
else:
    logger.handlers.clear()
    logger.addHandler(console_handler)

def chunk_text(text, max_len=450):
    """
    Split text into chunks of at most max_len characters, respecting word boundaries.
    Returns a list of text chunks.
    """
    return textwrap.wrap(text, max_len)

class TranscriptionService:
    def __init__(self):
        try:
            # Force CPU usage
            self.device = "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Initialize Whisper with CPU device
            logger.info(f"Initializing Whisper model with configuration: {Config.WHISPER_MODEL}")
            self.model = whisper.load_model(
                Config.WHISPER_MODEL,
                device=self.device,
                download_root=Config.MODEL_CACHE_DIR
            )
            logger.info("Whisper model initialized successfully")
            
            self.audio_service = AudioService()
            self.nlp_service = NLPService()
            
        except Exception as e:
            logger.error(f"Failed to initialize TranscriptionService: {str(e)}")
            raise

    def transcribe_file(self, audio_path, target_language='hi'):
        """
        Transcribe an audio file and process the results
        """
        try:
            logger.info(f"Starting transcription of file: {audio_path}")
            
            # Transcribe using CPU
            result = self.model.transcribe(
                audio_path,
                task="transcribe",
                fp16=False  # Disable FP16 since we're using CPU
            )
            detected_lang = result.get("language", "en")
            transcribed_text = result["text"].strip()
            logger.info(f"Detected language: {detected_lang}")
            
            return self._process_transcription(
                transcribed_text, 
                detected_lang, 
                target_language
            )

        except Exception as e:
            logger.error(f"Error transcribing file: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def transcribe_chunk(self, audio_data, target_language='hi', sample_rate=None):
        """
        Transcribe an audio chunk and process the results
        """
        try:
            logger.info("Starting transcription of audio chunk")
            
            # Preprocess audio data
            processed_audio = self.audio_service.preprocess_audio(audio_data, sample_rate)
            
            # Transcribe using CPU
            result = self.model.transcribe(
                processed_audio,
                task="transcribe",
                fp16=False  # Disable FP16 since we're using CPU
            )
            detected_lang = result.get("language", "en")
            transcribed_text = result["text"].strip()
            logger.info(f"Detected language: {detected_lang}")
            
            return self._process_transcription(
                transcribed_text, 
                detected_lang, 
                target_language
            )

        except Exception as e:
            logger.error(f"Error transcribing chunk: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _process_transcription(self, transcribed_text, detected_lang, target_language):
        """
        Process transcribed text: translate if needed and generate summaries
        """
        try:
            # If the detected language is not English, first translate to English
            english_text = transcribed_text
            if detected_lang != "en":
                logger.info(f"Translating from {detected_lang} to English")
                english_text = self.nlp_service.translate_to_english(transcribed_text, detected_lang)

            # Generate summary and extract action items from English text
            summary = self.nlp_service.generate_summary(english_text)
            action_items = self.nlp_service.extract_action_items(english_text)

            # Translate to target language if needed
            translated_text = english_text
            if target_language != "en":
                translated_text = self.nlp_service.translate_to_language(english_text, target_language)
                
                # Translate summary and action items
                if summary['english']:
                    summary['translated'] = self.nlp_service.translate_to_language(
                        summary['english'], 
                        target_language
                    )
                    summary['translated_topic_summaries'] = [
                        self.nlp_service.translate_to_language(topic, target_language)
                        for topic in summary.get('topic_summaries', [])
                    ]
                    summary['translated_key_points'] = [
                        self.nlp_service.translate_to_language(point, target_language)
                        for point in summary.get('key_points', [])
                    ]

                # Translate action items
                for item in action_items:
                    if item.get('text'):
                        item['translated_text'] = self.nlp_service.translate_to_language(
                            item['text'], 
                            target_language
                        )

            return {
                "original": transcribed_text,
                "english": english_text,
                "translated": translated_text,
                "target_language": target_language,
                "summary": summary,
                "action_items": action_items,
                "detected_language": detected_lang
            }

        except Exception as e:
            logger.error(f"Error processing transcription: {str(e)}")
            return {
                "original": transcribed_text,
                "english": english_text,
                "translated": transcribed_text,
                "target_language": target_language,
                "summary": {"english": "", "translated": "", "topic_summaries": [], "key_points": []},
                "action_items": [],
                "detected_language": detected_lang
            }

    def get_supported_languages(self):
        """
        Get list of supported languages for translation
        """
        return self.nlp_service.get_supported_languages()
