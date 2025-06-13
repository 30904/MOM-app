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
    return textwrap.wrap(text, max_len)

class TranscriptionService:
    def __init__(self):
        try:
            logger.info(f"Initializing Whisper model with configuration: {Config.WHISPER_MODEL}")
            self.model = whisper.load_model(Config.WHISPER_MODEL)
            logger.info("Whisper model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {str(e)}")
            raise

        self.device = 0 if torch.cuda.is_available() else -1
        self.batch_size = 8 if self.device >= 0 else 1

        # Translation models for English to other languages
        self.translation_models = {
            "hi": {
                "en-to-lang": "Helsinki-NLP/opus-mt-en-hi",
                "lang-to-en": "Helsinki-NLP/opus-mt-hi-en"
            },
            "fr": {
                "en-to-lang": "Helsinki-NLP/opus-mt-en-fr",
                "lang-to-en": "Helsinki-NLP/opus-mt-fr-en"
            },
            "es": {
                "en-to-lang": "Helsinki-NLP/opus-mt-en-es",
                "lang-to-en": "Helsinki-NLP/opus-mt-es-en"
            },
            "de": {
                "en-to-lang": "Helsinki-NLP/opus-mt-en-de",
                "lang-to-en": "Helsinki-NLP/opus-mt-de-en"
            },
            "zh": {
                "en-to-lang": "Helsinki-NLP/opus-mt-en-zh",
                "lang-to-en": "Helsinki-NLP/opus-mt-zh-en"
            },
            "ja": {
                "en-to-lang": "Helsinki-NLP/opus-mt-en-jap",
                "lang-to-en": "Helsinki-NLP/opus-mt-jap-en"
            },
            "ko": {
                "en-to-lang": "Helsinki-NLP/opus-mt-en-kor",
                "lang-to-en": "Helsinki-NLP/opus-mt-kor-en"
            },
            "ar": {
                "en-to-lang": "Helsinki-NLP/opus-mt-en-ar",
                "lang-to-en": "Helsinki-NLP/opus-mt-ar-en"
            },
        }
        
        self.translators = {}
        self.reverse_translators = {}

        # Initialize both forward and reverse translation models
        for lang, models in self.translation_models.items():
            try:
                # Forward translation (English to target language)
                self.translators[lang] = pipeline(
                    "translation",
                    model=models["en-to-lang"],
                    device=self.device,
                    model_kwargs={"low_cpu_mem_usage": True}
                )
                logger.info(f"Loaded forward translation model for {lang}: {models['en-to-lang']}")

                # Reverse translation (target language to English)
                self.reverse_translators[lang] = pipeline(
                    "translation",
                    model=models["lang-to-en"],
                    device=self.device,
                    model_kwargs={"low_cpu_mem_usage": True}
                )
                logger.info(f"Loaded reverse translation model for {lang}: {models['lang-to-en']}")
            except Exception as e:
                logger.warning(f"Failed to load translation models for {lang}: {e}")

        try:
            self.nlp_service = NLPService()
            logger.info("NLPService initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize NLPService: {e}")
            raise

    def transcribe_file(self, audio_path, target_language='hi'):
        try:
            logger.info(f"Starting transcription of file: {audio_path}")
            # Let Whisper auto-detect the language
            result = self.model.transcribe(audio_path, task="transcribe")
            detected_lang = result.get("language", "en")
            transcribed_text = result["text"].strip()
            logger.info(f"Detected language: {detected_lang}, Transcription: {transcribed_text[:100]}...")

            # If the detected language is not English, first translate to English
            english_text = transcribed_text
            if detected_lang != "en" and detected_lang in self.reverse_translators:
                logger.info(f"Translating from {detected_lang} to English")
                english_text = self._translate_from_lang(transcribed_text, detected_lang)

            # Then translate to target language if needed
            translated_text = english_text
            if target_language != "en":
                translated_text = self._translate_to_lang(english_text, target_language)

            try:
                logger.info("Generating summary from English text")
                summary = self.nlp_service.generate_summary(english_text)
                logger.info(f"Summary generated successfully: {summary['english'][:100]}...")

                if summary['english'] and summary['english'] != english_text:
                    translated_summary = self._translate_to_lang(summary['english'], target_language)
                    translated_topics = [self._translate_to_lang(topic, target_language) for topic in summary.get('topic_summaries', [])]
                    translated_points = [self._translate_to_lang(point, target_language) for point in summary.get('key_points', [])]
                    summary['translated'] = translated_summary
                    summary['translated_topic_summaries'] = translated_topics
                    summary['translated_key_points'] = translated_points
                    logger.info("Summary translated successfully")
                else:
                    logger.warning("Summary generation returned original text or empty result")
                    summary['translated'] = translated_text
                    summary['translated_topic_summaries'] = []
                    summary['translated_key_points'] = []
            except Exception as e:
                logger.error(f"Error in summary generation: {str(e)}")
                summary = {
                    'english': english_text,
                    'translated': translated_text,
                    'topic_summaries': [],
                    'key_points': [],
                    'translated_topic_summaries': [],
                    'translated_key_points': []
                }

            action_items = self.nlp_service.extract_action_items(english_text)
            for item in action_items:
                if item.get('text'):
                    item['translated_text'] = self._translate_to_lang(item['text'], target_language)

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
            logger.error(f"Error transcribing file: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def transcribe_chunk(self, audio_data, target_language='hi', sample_rate=None):
        try:
            logger.info(f"Starting transcription of audio chunk, sample rate: {sample_rate}")

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

            max_val = np.max(np.abs(audio_array))
            if max_val > 1.0:
                audio_array = audio_array / max_val

            if sample_rate and sample_rate != 16000:
                audio_array = signal.resample(audio_array, int(len(audio_array) * 16000 / sample_rate))

            min_samples = int(16000 * 0.03)
            if len(audio_array) < min_samples:
                audio_array = np.pad(audio_array, (0, min_samples - len(audio_array)))

            # Let Whisper auto-detect the language
            result = self.model.transcribe(audio_array, task="transcribe")
            detected_lang = result.get("language", "en")
            transcribed_text = result["text"].strip()
            logger.info(f"Detected language: {detected_lang}, Transcription: {transcribed_text[:100]}...")

            # If the detected language is not English, first translate to English
            english_text = transcribed_text
            if detected_lang != "en" and detected_lang in self.reverse_translators:
                logger.info(f"Translating from {detected_lang} to English")
                english_text = self._translate_from_lang(transcribed_text, detected_lang)

            # Then translate to target language if needed
            translated_text = english_text
            if target_language != "en":
                translated_text = self._translate_to_lang(english_text, target_language)

            try:
                logger.info("Generating summary from English text")
                summary = self.nlp_service.generate_summary(english_text)
                logger.info(f"Summary generated successfully: {summary['english'][:100]}...")

                if summary['english'] and summary['english'] != english_text:
                    translated_summary = self._translate_to_lang(summary['english'], target_language)
                    translated_topics = [self._translate_to_lang(topic, target_language) for topic in summary.get('topic_summaries', [])]
                    translated_points = [self._translate_to_lang(point, target_language) for point in summary.get('key_points', [])]
                    summary['translated'] = translated_summary
                    summary['translated_topic_summaries'] = translated_topics
                    summary['translated_key_points'] = translated_points
                    logger.info("Summary translated successfully")
                else:
                    logger.warning("Summary generation returned original text or empty result")
                    summary['translated'] = translated_text
                    summary['translated_topic_summaries'] = []
                    summary['translated_key_points'] = []
            except Exception as e:
                logger.error(f"Error in summary generation: {str(e)}")
                summary = {
                    'english': english_text,
                    'translated': translated_text,
                    'topic_summaries': [],
                    'key_points': [],
                    'translated_topic_summaries': [],
                    'translated_key_points': []
                }

            action_items = self.nlp_service.extract_action_items(english_text)
            for item in action_items:
                if item.get('text'):
                    item['translated_text'] = self._translate_to_lang(item['text'], target_language)

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
            logger.error(f"Error transcribing chunk: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _translate_to_lang(self, text, lang_code):
        if not text.strip() or lang_code == "en":
            return text

        if lang_code not in self.translators:
            logger.warning(f"No translator loaded for language '{lang_code}'. Returning original text.")
            return text

        try:
            logger.info(f"Translating text to {lang_code}")
            chunks = chunk_text(text, max_len=450)
            batches = [chunks[i:i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]
            translations = []
            for batch in batches:
                results = self.translators[lang_code](batch, max_length=512)
                translations.extend(result["translation_text"] for result in results)
            return " ".join(translations)
        except Exception as e:
            logger.error(f"Translation error for language {lang_code}: {e}")
            return text

    def _translate_from_lang(self, text, source_lang):
        if not text.strip() or source_lang == "en":
            return text

        if source_lang not in self.reverse_translators:
            logger.warning(f"No reverse translator loaded for language '{source_lang}'. Returning original text.")
            return text

        try:
            logger.info(f"Translating text from {source_lang} to English")
            chunks = chunk_text(text, max_len=450)
            batches = [chunks[i:i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]
            translations = []
            for batch in batches:
                results = self.reverse_translators[source_lang](batch, max_length=512)
                translations.extend(result["translation_text"] for result in results)
            return " ".join(translations)
        except Exception as e:
            logger.error(f"Translation error from language {source_lang}: {e}")
            return text

    def get_supported_languages(self):
        return list(self.translation_models.keys()) + ["en"]
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
    return textwrap.wrap(text, max_len)

class TranscriptionService:
    def __init__(self):
        try:
            logger.info(f"Initializing Whisper model with configuration: {Config.WHISPER_MODEL}")
            self.model = whisper.load_model(Config.WHISPER_MODEL)
            logger.info("Whisper model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {str(e)}")
            raise

        self.device = 0 if torch.cuda.is_available() else -1
        self.batch_size = 8 if self.device >= 0 else 1

        # Translation models for English to other languages
        self.translation_models = {
            "hi": {
                "en-to-lang": "Helsinki-NLP/opus-mt-en-hi",
                "lang-to-en": "Helsinki-NLP/opus-mt-hi-en"
            },
            "fr": {
                "en-to-lang": "Helsinki-NLP/opus-mt-en-fr",
                "lang-to-en": "Helsinki-NLP/opus-mt-fr-en"
            },
            "es": {
                "en-to-lang": "Helsinki-NLP/opus-mt-en-es",
                "lang-to-en": "Helsinki-NLP/opus-mt-es-en"
            },
            "de": {
                "en-to-lang": "Helsinki-NLP/opus-mt-en-de",
                "lang-to-en": "Helsinki-NLP/opus-mt-de-en"
            },
            "zh": {
                "en-to-lang": "Helsinki-NLP/opus-mt-en-zh",
                "lang-to-en": "Helsinki-NLP/opus-mt-zh-en"
            },
            "ja": {
                "en-to-lang": "Helsinki-NLP/opus-mt-en-jap",
                "lang-to-en": "Helsinki-NLP/opus-mt-jap-en"
            },
            "ko": {
                "en-to-lang": "Helsinki-NLP/opus-mt-en-kor",
                "lang-to-en": "Helsinki-NLP/opus-mt-kor-en"
            },
            "ar": {
                "en-to-lang": "Helsinki-NLP/opus-mt-en-ar",
                "lang-to-en": "Helsinki-NLP/opus-mt-ar-en"
            },
        }
        
        self.translators = {}
        self.reverse_translators = {}

        # Initialize both forward and reverse translation models
        for lang, models in self.translation_models.items():
            try:
                # Forward translation (English to target language)
                self.translators[lang] = pipeline(
                    "translation",
                    model=models["en-to-lang"],
                    device=self.device,
                    model_kwargs={"low_cpu_mem_usage": True}
                )
                logger.info(f"Loaded forward translation model for {lang}: {models['en-to-lang']}")

                # Reverse translation (target language to English)
                self.reverse_translators[lang] = pipeline(
                    "translation",
                    model=models["lang-to-en"],
                    device=self.device,
                    model_kwargs={"low_cpu_mem_usage": True}
                )
                logger.info(f"Loaded reverse translation model for {lang}: {models['lang-to-en']}")
            except Exception as e:
                logger.warning(f"Failed to load translation models for {lang}: {e}")

        try:
            self.nlp_service = NLPService()
            logger.info("NLPService initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize NLPService: {e}")
            raise

    def transcribe_file(self, audio_path, target_language='hi'):
        try:
            logger.info(f"Starting transcription of file: {audio_path}")
            # Let Whisper auto-detect the language
            result = self.model.transcribe(audio_path, task="transcribe")
            detected_lang = result.get("language", "en")
            transcribed_text = result["text"].strip()
            logger.info(f"Detected language: {detected_lang}, Transcription: {transcribed_text[:100]}...")

            # If the detected language is not English, first translate to English
            english_text = transcribed_text
            if detected_lang != "en" and detected_lang in self.reverse_translators:
                logger.info(f"Translating from {detected_lang} to English")
                english_text = self._translate_from_lang(transcribed_text, detected_lang)

            # Then translate to target language if needed
            translated_text = english_text
            if target_language != "en":
                translated_text = self._translate_to_lang(english_text, target_language)

            try:
                logger.info("Generating summary from English text")
                summary = self.nlp_service.generate_summary(english_text)
                logger.info(f"Summary generated successfully: {summary['english'][:100]}...")

                if summary['english'] and summary['english'] != english_text:
                    translated_summary = self._translate_to_lang(summary['english'], target_language)
                    translated_topics = [self._translate_to_lang(topic, target_language) for topic in summary.get('topic_summaries', [])]
                    translated_points = [self._translate_to_lang(point, target_language) for point in summary.get('key_points', [])]
                    summary['translated'] = translated_summary
                    summary['translated_topic_summaries'] = translated_topics
                    summary['translated_key_points'] = translated_points
                    logger.info("Summary translated successfully")
                else:
                    logger.warning("Summary generation returned original text or empty result")
                    summary['translated'] = translated_text
                    summary['translated_topic_summaries'] = []
                    summary['translated_key_points'] = []
            except Exception as e:
                logger.error(f"Error in summary generation: {str(e)}")
                summary = {
                    'english': english_text,
                    'translated': translated_text,
                    'topic_summaries': [],
                    'key_points': [],
                    'translated_topic_summaries': [],
                    'translated_key_points': []
                }

            action_items = self.nlp_service.extract_action_items(english_text)
            for item in action_items:
                if item.get('text'):
                    item['translated_text'] = self._translate_to_lang(item['text'], target_language)

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
            logger.error(f"Error transcribing file: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def transcribe_chunk(self, audio_data, target_language='hi', sample_rate=None):
        try:
            logger.info(f"Starting transcription of audio chunk, sample rate: {sample_rate}")

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

            max_val = np.max(np.abs(audio_array))
            if max_val > 1.0:
                audio_array = audio_array / max_val

            if sample_rate and sample_rate != 16000:
                audio_array = signal.resample(audio_array, int(len(audio_array) * 16000 / sample_rate))

            min_samples = int(16000 * 0.03)
            if len(audio_array) < min_samples:
                audio_array = np.pad(audio_array, (0, min_samples - len(audio_array)))

            # Let Whisper auto-detect the language
            result = self.model.transcribe(audio_array, task="transcribe")
            detected_lang = result.get("language", "en")
            transcribed_text = result["text"].strip()
            logger.info(f"Detected language: {detected_lang}, Transcription: {transcribed_text[:100]}...")

            # If the detected language is not English, first translate to English
            english_text = transcribed_text
            if detected_lang != "en" and detected_lang in self.reverse_translators:
                logger.info(f"Translating from {detected_lang} to English")
                english_text = self._translate_from_lang(transcribed_text, detected_lang)

            # Then translate to target language if needed
            translated_text = english_text
            if target_language != "en":
                translated_text = self._translate_to_lang(english_text, target_language)

            try:
                logger.info("Generating summary from English text")
                summary = self.nlp_service.generate_summary(english_text)
                logger.info(f"Summary generated successfully: {summary['english'][:100]}...")

                if summary['english'] and summary['english'] != english_text:
                    translated_summary = self._translate_to_lang(summary['english'], target_language)
                    translated_topics = [self._translate_to_lang(topic, target_language) for topic in summary.get('topic_summaries', [])]
                    translated_points = [self._translate_to_lang(point, target_language) for point in summary.get('key_points', [])]
                    summary['translated'] = translated_summary
                    summary['translated_topic_summaries'] = translated_topics
                    summary['translated_key_points'] = translated_points
                    logger.info("Summary translated successfully")
                else:
                    logger.warning("Summary generation returned original text or empty result")
                    summary['translated'] = translated_text
                    summary['translated_topic_summaries'] = []
                    summary['translated_key_points'] = []
            except Exception as e:
                logger.error(f"Error in summary generation: {str(e)}")
                summary = {
                    'english': english_text,
                    'translated': translated_text,
                    'topic_summaries': [],
                    'key_points': [],
                    'translated_topic_summaries': [],
                    'translated_key_points': []
                }

            action_items = self.nlp_service.extract_action_items(english_text)
            for item in action_items:
                if item.get('text'):
                    item['translated_text'] = self._translate_to_lang(item['text'], target_language)

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
            logger.error(f"Error transcribing chunk: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _translate_to_lang(self, text, lang_code):
        if not text.strip() or lang_code == "en":
            return text

        if lang_code not in self.translators:
            logger.warning(f"No translator loaded for language '{lang_code}'. Returning original text.")
            return text

        try:
            logger.info(f"Translating text to {lang_code}")
            chunks = chunk_text(text, max_len=450)
            batches = [chunks[i:i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]
            translations = []
            for batch in batches:
                results = self.translators[lang_code](batch, max_length=512)
                translations.extend(result["translation_text"] for result in results)
            return " ".join(translations)
        except Exception as e:
            logger.error(f"Translation error for language {lang_code}: {e}")
            return text

    def _translate_from_lang(self, text, source_lang):
        if not text.strip() or source_lang == "en":
            return text

        if source_lang not in self.reverse_translators:
            logger.warning(f"No reverse translator loaded for language '{source_lang}'. Returning original text.")
            return text

        try:
            logger.info(f"Translating text from {source_lang} to English")
            chunks = chunk_text(text, max_len=450)
            batches = [chunks[i:i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]
            translations = []
            for batch in batches:
                results = self.reverse_translators[source_lang](batch, max_length=512)
                translations.extend(result["translation_text"] for result in results)
            return " ".join(translations)
        except Exception as e:
            logger.error(f"Translation error from language {source_lang}: {e}")
            return text

    def get_supported_languages(self):
        return list(self.translation_models.keys()) + ["en"]
