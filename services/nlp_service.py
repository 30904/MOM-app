import spacy
import re
from datetime import datetime
import logging
import json
import os
from logging.handlers import RotatingFileHandler
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import torch
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import traceback
from .translation_service import TranslationService
import sys
import textwrap
import psutil

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def setup_logging():
    """Set up logging with proper directory creation and rotation."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure summary logger
    summary_logger = logging.getLogger('summary_logger')
    summary_logger.setLevel(logging.INFO)
    
    # Set up rotating file handler (10MB per file, keep 5 backup files)
    log_file = os.path.join(log_dir, 'meeting_summaries.log')
    handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    
    # Set up formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add handler to logger
    summary_logger.addHandler(handler)
    
    return summary_logger

# Set up logging
summary_logger = setup_logging()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

if not logger.hasHandlers():
    logger.addHandler(console_handler)

def chunk_text(text, max_len=450):
    return textwrap.wrap(text, max_len)

class TopicSegmenter:
    """Segments text into coherent topics."""
    
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self.nlp = spacy.load("en_core_web_sm")
        self.stopwords = set(stopwords.words('english'))
    
    def get_sentence_vector(self, sentence: str) -> np.ndarray:
        """Convert a sentence to its vector representation."""
        doc = self.nlp(sentence)
        return doc.vector
    
    def segment(self, sentences: List[str]) -> List[List[str]]:
        """Segment sentences into coherent topics."""
        if len(sentences) <= self.window_size:
            return [sentences]
        
        # Calculate similarity scores between adjacent windows
        scores = []
        for i in range(len(sentences) - self.window_size):
            window1 = ' '.join(sentences[i:i + self.window_size])
            window2 = ' '.join(sentences[i + 1:i + 1 + self.window_size])
            vec1 = self.get_sentence_vector(window1)
            vec2 = self.get_sentence_vector(window2)
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            scores.append(similarity)
        
        # Find topic boundaries (local minima in similarity scores)
        boundaries = [0]
        for i in range(1, len(scores) - 1):
            if scores[i] < scores[i-1] and scores[i] < scores[i+1]:
                boundaries.append(i + self.window_size)
        boundaries.append(len(sentences))
        
        # Create segments
        segments = []
        for i in range(len(boundaries) - 1):
            segment = sentences[boundaries[i]:boundaries[i+1]]
            segments.append(segment)
        
        return segments

class PriorityClassifier:
    """Classifies the priority of action items."""
    
    def __init__(self):
        self.priority_indicators = {
            'high': ['urgent', 'critical', 'asap', 'immediately', 'crucial', 'vital',
                    'important', 'priority', 'emergency', 'deadline'],
            'medium': ['soon', 'next', 'following', 'upcoming', 'later', 'should',
                      'would', 'could', 'may', 'might'],
            'low': ['eventually', 'sometime', 'when possible', 'if possible',
                   'optional', 'nice to have', 'consider', 'think about']
        }
    
    def classify(self, text: str) -> str:
        """Classify the priority of an action item."""
        text = text.lower()
        
        # Check for explicit priority mentions
        if any(word in text for word in self.priority_indicators['high']):
            return 'high'
        if any(word in text for word in self.priority_indicators['medium']):
            return 'medium'
        if any(word in text for word in self.priority_indicators['low']):
            return 'low'
        
        # Default to medium priority
        return 'medium'

class NLPService:
    def __init__(self):
        logger.info("Initializing NLPService")
        
        # Force CPU usage
        self.device = "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.batch_size = 1  # Reduced batch size for CPU
        
        # Initialize translation models
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
        
        # Initialize summarization pipeline
        try:
            logger.info("Loading summarization model...")
            model_name = "facebook/bart-large-cnn"
            
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model with appropriate settings
            model = self._load_model(model_name)
            
            # Create pipeline with appropriate device mapping
            self.summarizer = pipeline(
                "summarization",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            logger.info("Summarization model initialized successfully")
            
            # Initialize translation models after summarizer is loaded
            self._initialize_translation_models()
            
        except Exception as e:
            logger.error(f"Failed to initialize summarization model: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        # Initialize topic segmenter
        self.topic_segmenter = TopicSegmenter()
        
        # Initialize priority classifier
        self.priority_classifier = PriorityClassifier()
        
        # Enhanced patterns for action item detection
        self.action_patterns = [
            # Direct actions with assignee patterns
            r"(?i)(?P<assignee>[A-Z][a-z]+ (?:[A-Z][a-z]+)?)\s+(need|needs|needed) to\s+(?P<action>.+?)(?=\.|$)",
            r"(?i)(?P<assignee>[A-Z][a-z]+ (?:[A-Z][a-z]+)?)\s+(should|must|will|shall)\s+(?P<action>.+?)(?=\.|$)",
            r"(?i)(?P<assignee>[A-Z][a-z]+ (?:[A-Z][a-z]+)?)\s+(has|have) to\s+(?P<action>.+?)(?=\.|$)",
            
            # Team/Role based assignments
            r"(?i)(?P<assignee>team|developer|manager|lead|designer)\s+(should|must|will|need to)\s+(?P<action>.+?)(?=\.|$)",
            
            # Deadline-specific patterns
            r"(?i)(need|needs|needed) to\s+(?P<action>.+?)\s+by\s+(?P<deadline>[A-Za-z]+day|tomorrow|next week|[A-Za-z]+ \d{1,2}(?:st|nd|rd|th)?|end of [A-Za-z]+)(?=\.|$)",
            
            # General action patterns
            r"(?i)(action item|todo|task):\s*(?P<action>.+?)(?=\.|$)",
            r"(?i)please\s+(?P<action>.+?)(?=\.|$)",
            r"(?i)(important|crucial|urgent) to\s+(?P<action>.+?)(?=\.|$)",
            r"(?i)(must|should|needs to) be\s+(?P<action>.+?)(?=\.|$)",
            r"(?i)(follow[- ]up|action):\s*(?P<action>.+?)(?=\.|$)"
        ]
        
        # Key phrases for importance detection
        self.key_phrases = [
            # Decision indicators
            "decided", "agreed", "concluded", "resolved", "determined",
            # Summary indicators
            "in summary", "to summarize", "key points", "main takeaways",
            # Action indicators
            "next steps", "action items", "follow up", "to do",
            # Important points
            "important", "critical", "crucial", "essential", "significant",
            # Meeting-specific
            "agenda", "discussion points", "meeting objectives"
        ]
    
    def _load_model(self, model_name: str):
        """Load a model with appropriate device placement"""
        try:
            # Load model in CPU mode
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map=None
            )

            # Create empty model on CPU and load state dict
            empty_model = model.to_empty(device="cpu")
            state_dict = model.state_dict()
            empty_model.load_state_dict(state_dict)
            return empty_model

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _initialize_translation_models(self):
        """Initialize translation models with proper device mapping"""
        for lang, models in self.translation_models.items():
            try:
                # Forward translation (English to target language)
                logger.info(f"Loading forward translation model for {lang}...")
                
                # Initialize forward translation
                tokenizer = AutoTokenizer.from_pretrained(models["en-to-lang"])
                model = self._load_model(models["en-to-lang"])
                
                self.translators[lang] = pipeline(
                    "translation",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1  # Force CPU usage
                )
                logger.info(f"Loaded forward translation model for {lang}")

                # Initialize reverse translation
                logger.info(f"Loading reverse translation model for {lang}...")
                tokenizer = AutoTokenizer.from_pretrained(models["lang-to-en"])
                model = self._load_model(models["lang-to-en"])
                
                self.reverse_translators[lang] = pipeline(
                    "translation",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1  # Force CPU usage
                )
                logger.info(f"Loaded reverse translation model for {lang}")
                
            except Exception as e:
                logger.warning(f"Failed to load translation models for {lang}: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")

    def _determine_device(self):
        """
        Determine the best device to use based on available resources
        """
        if not torch.cuda.is_available():
            logger.info("CUDA not available, using CPU")
            return "cpu"
        
        try:
            # Get GPU memory info
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)  # Convert to GB
            
            # Get system memory info
            system_memory_gb = psutil.virtual_memory().total / (1024**3)
            
            logger.info(f"GPU Memory: {gpu_memory_gb:.2f} GB")
            logger.info(f"System Memory: {system_memory_gb:.2f} GB")
            
            # If GPU has less than 6GB or system has less than 16GB RAM, use CPU
            if gpu_memory_gb < 6 or system_memory_gb < 16:
                logger.info("Limited resources detected, using CPU for better stability")
                return "cpu"
            
            # Test GPU memory allocation
            try:
                torch.cuda.empty_cache()
                test_tensor = torch.zeros((1, 1000, 1000), device="cuda")
                del test_tensor
                torch.cuda.empty_cache()
                return "cuda"
            except RuntimeError:
                logger.warning("GPU memory test failed, falling back to CPU")
                return "cpu"
                
        except Exception as e:
            logger.warning(f"Error determining device capabilities: {e}")
            return "cpu"

    def translate_to_language(self, text, target_lang):
        if not text.strip() or target_lang == "en":
            return text

        if target_lang not in self.translators:
            logger.warning(f"No translator loaded for language '{target_lang}'. Returning original text.")
            return text

        try:
            chunks = chunk_text(text)
            batches = [chunks[i:i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]
            translations = []
            for batch in batches:
                results = self.translators[target_lang](batch, max_length=512)
                translations.extend(result["translation_text"] for result in results)
            return " ".join(translations)
        except Exception as e:
            logger.error(f"Translation error for language {target_lang}: {e}")
            return text

    def translate_to_english(self, text, source_lang):
        if not text.strip() or source_lang == "en":
            return text

        if source_lang not in self.reverse_translators:
            logger.warning(f"No reverse translator loaded for language '{source_lang}'. Returning original text.")
            return text

        try:
            chunks = chunk_text(text)
            batches = [chunks[i:i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]
            translations = []
            for batch in batches:
                results = self.reverse_translators[source_lang](batch, max_length=512)
                translations.extend(result["translation_text"] for result in results)
            return " ".join(translations)
        except Exception as e:
            logger.error(f"Translation error from language {source_lang}: {e}")
            return text

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for summarization."""
        try:
            # Remove excessive whitespace and normalize
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Split into sentences
            sentences = sent_tokenize(text)
            
            # Remove very short sentences and noise
            sentences = [s for s in sentences if len(s.split()) > 3]
            sentences = [s for s in sentences if not self._is_noise(s)]
            
            # Rejoin text
            cleaned_text = ' '.join(sentences)
            
            return cleaned_text
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return text
    
    def _is_noise(self, sentence: str) -> bool:
        """Check if a sentence is likely noise."""
        # Check for common noise patterns
        noise_patterns = [
            r'^\s*[0-9:]+\s*$',  # Just timestamps
            r'^\s*\[.*?\]\s*$',  # Just bracketed content
            r'^\s*[A-Za-z]+:\s*$',  # Just speaker labels
        ]
        return any(re.match(pattern, sentence) for pattern in noise_patterns)
    
    def _calculate_summary_length(self, text_length: int) -> Tuple[int, int]:
        """Calculate appropriate min and max summary lengths based on input length."""
        # For very short text (< 50 words)
        if text_length < 50:
            return max(5, text_length // 3), min(text_length - 1, 30)
        
        # For short text (50-100 words)
        if text_length < 100:
            return max(10, text_length // 3), min(text_length // 2, 50)
        
        # For medium text (100-300 words)
        if text_length < 300:
            return 40, min(text_length // 2, 100)
        
        # For medium-long text (300-1000 words)
        if text_length < 1000:
            return 75, min(text_length // 3, 150)
        
        # For long text (> 1000 words)
        return 100, min(text_length // 4, 200)

    def generate_summary(self, text: str, meeting_id: Optional[str] = None, target_language: Optional[str] = None, paraphrase: bool = True) -> Dict:
        """Generate a comprehensive summary of the meeting transcript with optional translation and paraphrasing."""
        try:
            logger.info(f"Generating summary for text: {text[:100]}...")
            
            # Preprocess the text
            cleaned_text = self.preprocess_text(text)
            word_count = len(cleaned_text.split())
            logger.debug(f"Preprocessed text word count: {word_count}")
            
            # If text is extremely short, still try to summarize but with adjusted parameters
            if word_count < 20:
                logger.info("Text very short, returning original with key points")
                result = {
                    "summary": cleaned_text,
                    "key_points": self._extract_key_points(cleaned_text),
                    "topics": []
                }
                if target_language:
                    result["translations"] = {
                        "summary": self.translate_to_language(cleaned_text, target_language),
                        "key_points": []
                    }
                self._log_summary(result, meeting_id, word_count)
                return result
            
            # For all other cases, proceed with summarization
            min_length, max_length = self._calculate_summary_length(word_count)
            
            # Generate summary with adjusted parameters for short text
            summary = self.summarizer(
                cleaned_text,
                max_length=max_length,
                min_length=min_length,
                length_penalty=1.5,  # Reduced from 2.0 for shorter texts
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,  # Reduced from 3 for shorter texts
                do_sample=False
            )[0]['summary_text']
            
            processed_summary = self.postprocess_summary(summary)

            # Apply paraphrasing if requested
            if paraphrase:
                try:
                    if processed_summary.strip():
                        input_text = f"paraphrase: {processed_summary} </s>"
                        outputs = self.summarizer(
                            input_text,
                            max_length=60,
                            num_beams=5,
                            num_return_sequences=1,
                            temperature=1.5,
                            do_sample=True
                        )
                        processed_summary = outputs[0]["generated_text"].strip()
                except Exception as e:
                    logger.error(f"Paraphrasing error: {e}")
                    # Keep original summary if paraphrasing fails
            
            
            key_points = self._extract_key_points(cleaned_text)
            
            result = {
                "summary": processed_summary,
                "key_points": key_points,
                "topics": []  # For very short texts, we skip topic segmentation
            }
            
            # Add translations if target language is specified
            if target_language:
                result["translations"] = {
                    "summary": self.translate_to_language(processed_summary, target_language),
                    "key_points": [
                        self.translate_to_language(point, target_language)
                        for point in key_points
                    ]
                }
            
            logger.info(f"Generated summary with {len(key_points)} key points")
            self._log_summary(result, meeting_id, word_count)
            return result
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "summary": "Error generating summary",
                "topic_summaries": [],
                "key_points": []
            }

    def _log_summary(self, summary_data: Dict, meeting_id: Optional[str], word_count: int) -> None:
        """Log summary and metadata to the summary log file."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "meeting_id": meeting_id or "unknown",
                "word_count": word_count,
                "summary_length": len(summary_data["summary"].split()),
                "num_topics": len(summary_data.get("topic_summaries", [])),
                "num_key_points": len(summary_data.get("key_points", [])),
                "data": summary_data
            }
            summary_logger.info(json.dumps(log_entry))
        except Exception as e:
            logger.error(f"Error logging summary: {str(e)}")

    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from the text."""
        key_points = []
        doc = self.summarizer.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        doc.to(self.device)
        outputs = self.summarizer(doc.input_ids)
        summary = outputs[0]["summary_text"]
        
        # Split into sentences
        sentences = sent_tokenize(summary)
        
        for sentence in sentences:
            # Check for key phrases
            if any(phrase in sentence.lower() for phrase in self.key_phrases):
                key_points.append(sentence)
            
            # Check for important entities
            sent_doc = self.summarizer.tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
            sent_doc.to(self.device)
            if any(ent.label_ in ['ORG', 'PERSON', 'DATE', 'MONEY'] for ent in self.summarizer.model.generate_output(sent_doc.input_ids).ents):
                key_points.append(sentence)
        
        # Remove duplicates while preserving order
        seen = set()
        key_points = [x for x in key_points if not (x.lower() in seen or seen.add(x.lower()))]
        
        return key_points
    
    def postprocess_summary(self, summary: str) -> str:
        """Post-process the generated summary."""
        try:
            # Remove any redundant periods at the end
            summary = re.sub(r'\.+$', '.', summary)
            
            # Ensure the summary ends with a period
            if not summary.endswith('.'):
                summary += '.'
            
            # Capitalize the first letter
            summary = summary[0].upper() + summary[1:]
            
            # Fix common issues
            summary = re.sub(r'\s+', ' ', summary)  # Remove extra spaces
            summary = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', summary)  # Fix sentence spacing
            
            return summary
        except Exception as e:
            logger.error(f"Error post-processing summary: {str(e)}")
            return summary
    
    def extract_action_items(self, text: str, target_language: Optional[str] = None) -> List[Dict]:
        """Extract action items from text with optional translation."""
        try:
            logger.info(f"Extracting action items from text: {text[:100]}...")
            
            # Preprocess text
            cleaned_text = self.preprocess_text(text)
            logger.debug(f"Preprocessed text: {cleaned_text[:100]}...")
            
            # Split into sentences
            sentences = sent_tokenize(cleaned_text)
            logger.debug(f"Split into {len(sentences)} sentences")
            
            action_items = []
            for sentence in sentences:
                # Check for action item patterns
                if any(pattern in sentence.lower() for pattern in ['todo', 'task', 'action', 'need to', 'should', 'must']):
                    logger.debug(f"Found potential action item: {sentence}")
                    
                    # Extract priority
                    priority = 'medium'
                    if any(word in sentence.lower() for word in ['urgent', 'asap', 'critical']):
                        priority = 'high'
                    elif any(word in sentence.lower() for word in ['low', 'when possible']):
                        priority = 'low'
                    
                    # Extract assignee if present
                    assignee = None
                    assignee_match = re.search(r'(?:assigned to|for|by)\s+([A-Za-z\s]+)', sentence)
                    if assignee_match:
                        assignee = assignee_match.group(1).strip()
                    
                    # Extract deadline if present
                    deadline = None
                    deadline_match = re.search(r'(?:by|due|deadline|on)\s+([A-Za-z0-9\s,]+)', sentence)
                    if deadline_match:
                        deadline = deadline_match.group(1).strip()
                    
                    action_item = {
                        'text': sentence,
                        'priority': priority,
                        'assignee': assignee,
                        'deadline': deadline,
                        'context': text[:200]  # Include some context
                    }
                    
                    # Add translation if target language is specified
                    if target_language:
                        action_item['translation'] = self.translate_to_language(sentence, target_language)
                    
                    logger.info(f"Created action item: {action_item}")
                    action_items.append(action_item)
            
            logger.info(f"Extracted {len(action_items)} action items")
            return action_items
            
        except Exception as e:
            logger.error(f"Error extracting action items: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _extract_dependencies(self, action_text: str) -> List[str]:
        """Extract dependencies from action text."""
        dependencies = []
        dependency_patterns = [
            r"(?i)after\s+(.+?)(?=\.|$)",
            r"(?i)depends on\s+(.+?)(?=\.|$)",
            r"(?i)following\s+(.+?)(?=\.|$)",
            r"(?i)once\s+(.+?)(?=\.|$)"
        ]
        
        for pattern in dependency_patterns:
            matches = re.finditer(pattern, action_text)
            for match in matches:
                dependency = match.group(1).strip()
                if dependency:
                    dependencies.append(dependency)
        
        return dependencies
    
    def _extract_context(self, full_text: str, action_text: str) -> str:
        """Extract relevant context around an action item."""
        # Find the sentence containing the action
        sentences = sent_tokenize(full_text)
        for i, sentence in enumerate(sentences):
            if action_text in sentence:
                # Get surrounding context (previous and next sentence)
                start = max(0, i - 1)
                end = min(len(sentences), i + 2)
                context = ' '.join(sentences[start:end])
                return context
        
        return ""

    def _log_action_items(self, action_items: List[Dict], meeting_id: Optional[str]) -> None:
        """Log action items to the summary log file."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "meeting_id": meeting_id or "unknown",
                "num_action_items": len(action_items),
                "priority_distribution": {
                    "high": len([x for x in action_items if x["priority"] == "high"]),
                    "medium": len([x for x in action_items if x["priority"] == "medium"]),
                    "low": len([x for x in action_items if x["priority"] == "low"])
                },
                "action_items": action_items
            }
            summary_logger.info(json.dumps(log_entry))
        except Exception as e:
            logger.error(f"Error logging action items: {str(e)}")

    def get_supported_languages(self):
        return list(self.translation_models.keys()) + ["en"] 