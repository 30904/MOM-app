import spacy
import re
from datetime import datetime
import logging
import json
import os
from logging.handlers import RotatingFileHandler
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import torch
from typing import List, Dict, Optional, Tuple
import numpy as np

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
        # Initialize spaCy
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize the summarization pipeline
        try:
            # Use GPU if available
            self.device = 0 if torch.cuda.is_available() else -1
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=self.device
            )
            logger.info(f"Summarization model loaded successfully. Using device: {'cuda' if self.device >= 0 else 'cpu'}")
        except Exception as e:
            logger.error(f"Error loading summarization model: {str(e)}")
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
        # For very short text (< 100 words)
        if text_length < 100:
            return max(10, text_length // 2), min(text_length, 50)
        
        # For short text (100-300 words)
        if text_length < 300:
            return 40, min(text_length // 2, 100)
        
        # For medium text (300-1000 words)
        if text_length < 1000:
            return 75, min(text_length // 3, 150)
        
        # For long text (> 1000 words)
        return 100, min(text_length // 4, 200)

    def generate_summary(self, text: str, meeting_id: Optional[str] = None) -> Dict:
        """Generate a comprehensive summary of the meeting transcript."""
        try:
            # Preprocess the text
            cleaned_text = self.preprocess_text(text)
            word_count = len(cleaned_text.split())
            
            # If text is too short, return as is
            if word_count < 50:
                logger.info("Text too short for summarization, returning original")
                result = {
                    "summary": cleaned_text,
                    "key_points": [],
                    "topics": []
                }
                self._log_summary(result, meeting_id, word_count)
                return result
            
            # Split text into topics
            sentences = sent_tokenize(cleaned_text)
            topic_segments = self.topic_segmenter.segment(sentences)
            
            # Generate summary for each topic
            topic_summaries = []
            for segment in topic_segments:
                segment_text = ' '.join(segment)
                segment_length = len(segment_text.split())
                min_length, max_length = self._calculate_summary_length(segment_length)
                
                summary = self.summarizer(
                    segment_text,
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    do_sample=False
                )[0]['summary_text']
                topic_summaries.append(self.postprocess_summary(summary))
            
            # Generate overall summary
            min_length, max_length = self._calculate_summary_length(word_count)
            overall_summary = self.summarizer(
                cleaned_text,
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                do_sample=False
            )[0]['summary_text']
            
            # Extract key points
            key_points = self._extract_key_points(cleaned_text)
            
            result = {
                "summary": self.postprocess_summary(overall_summary),
                "topic_summaries": topic_summaries,
                "key_points": key_points
            }
            
            # Log the summary
            self._log_summary(result, meeting_id, word_count)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {
                "summary": "Error generating summary. Please try again.",
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
        doc = self.nlp(text)
        
        # Split into sentences
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            # Check for key phrases
            if any(phrase in sentence.lower() for phrase in self.key_phrases):
                key_points.append(sentence)
            
            # Check for important entities
            sent_doc = self.nlp(sentence)
            if any(ent.label_ in ['ORG', 'PERSON', 'DATE', 'MONEY'] for ent in sent_doc.ents):
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
    
    def extract_action_items(self, text: str, meeting_id: Optional[str] = None) -> List[Dict]:
        """Extract action items from the text with enhanced metadata."""
        try:
            doc = self.nlp(text)
            action_items = []
            seen_actions = set()
            
            # Extract using patterns
            for pattern in self.action_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    match_dict = match.groupdict()
                    
                    # Get action text
                    action_text = match_dict.get('action', match.group(0))
                    action_text = action_text.strip()
                    
                    # Skip if too short or already seen
                    if len(action_text.split()) < 3 or action_text.lower() in seen_actions:
                        continue
                    
                    # Extract metadata
                    action_doc = self.nlp(action_text)
                    
                    # Get assignee
                    assignee = match_dict.get('assignee')
                    if not assignee:
                        for ent in action_doc.ents:
                            if ent.label_ == "PERSON":
                                assignee = ent.text
                                break
                    
                    # Get deadline
                    deadline = match_dict.get('deadline')
                    if not deadline:
                        for ent in action_doc.ents:
                            if ent.label_ in ["DATE", "TIME"]:
                                deadline = ent.text
                                break
                    
                    # Determine priority
                    priority = self.priority_classifier.classify(action_text)
                    
                    # Extract dependencies
                    dependencies = self._extract_dependencies(action_text)
                    
                    # Create action item
                    action_item = {
                        "text": action_text,
                        "assignee": assignee,
                        "deadline": deadline,
                        "priority": priority,
                        "dependencies": dependencies,
                        "status": "pending",
                        "context": self._extract_context(text, action_text),
                        "created_at": datetime.now().isoformat()
                    }
                    
                    action_items.append(action_item)
                    seen_actions.add(action_text.lower())
            
            # Sort by priority
            priority_order = {'high': 0, 'medium': 1, 'low': 2}
            action_items.sort(key=lambda x: priority_order[x['priority']])
            
            # Log action items
            self._log_action_items(action_items, meeting_id)
            
            logger.info(f"Extracted {len(action_items)} action items")
            return action_items
            
        except Exception as e:
            logger.error(f"Error extracting action items: {str(e)}")
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