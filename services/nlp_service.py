import spacy
import re
from datetime import datetime
import logging
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
import torch

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)

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
        
        # Enhanced patterns for action item detection
        self.action_patterns = [
            # Direct actions
            r"(?i)(need|needs|needed) to\s+(.+?)(?=\.|$)",
            r"(?i)(should|must|will|shall)\s+(.+?)(?=\.|$)",
            r"(?i)(has|have) to\s+(.+?)(?=\.|$)",
            r"(?i)(going to|gonna)\s+(.+?)(?=\.|$)",
            # Questions and demands
            r"(?i)why (is|isn't|doesn't|can't)\s+(.+?)(?=\.|$)",
            r"(?i)how (can|should|will)\s+(.+?)(?=\.|$)",
            # Calls for action
            r"(?i)(important to|crucial to|necessary to)\s+(.+?)(?=\.|$)",
            r"(?i)(must be|should be|needs to be)\s+(.+?)(?=\.|$)",
            # Direct statements
            r"(?i)(demands?|calls for|requires)\s+(.+?)(?=\.|$)",
            r"(?i)(urged|recommended|suggested)\s+(.+?)(?=\.|$)",
        ]
        
        # Key phrases that indicate important content
        self.key_phrases = [
            "conclusion", "summary", "decision", "agreed", "important",
            "therefore", "thus", "hence", "consequently", "as a result",
            "in conclusion", "finally", "ultimately", "in summary",
            "key point", "critical", "significant", "notably", "specifically",
            "particularly", "especially", "primarily", "mainly", "chiefly"
        ]
    
    def preprocess_text(self, text):
        """Preprocess text for summarization."""
        try:
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Split into sentences
            sentences = sent_tokenize(text)
            
            # Remove very short sentences (likely noise)
            sentences = [s for s in sentences if len(s.split()) > 3]
            
            # Rejoin text
            cleaned_text = ' '.join(sentences)
            
            return cleaned_text
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return text
    
    def generate_summary(self, text):
        """Generate a summary of the meeting transcript using the BART model."""
        try:
            # Preprocess the text
            cleaned_text = self.preprocess_text(text)
            
            # If text is too short, return as is
            if len(cleaned_text.split()) < 50:
                logger.info("Text too short for summarization, returning original")
                return cleaned_text
            
            # Generate summary using the pipeline
            summary = self.summarizer(
                cleaned_text,
                max_length=150,  # Maximum length of summary
                min_length=40,   # Minimum length of summary
                length_penalty=2.0,  # Encourage longer summaries
                num_beams=4,     # Beam search for better quality
                early_stopping=True,
                no_repeat_ngram_size=3,  # Avoid repetition
                do_sample=False  # Use beam search instead of sampling
            )[0]['summary_text']
            
            # Post-process the summary
            summary = self.postprocess_summary(summary)
            
            logger.info("Summary generated successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Error generating summary. Please try again."
    
    def postprocess_summary(self, summary):
        """Post-process the generated summary."""
        try:
            # Remove any redundant periods at the end
            summary = re.sub(r'\.+$', '.', summary)
            
            # Ensure the summary ends with a period
            if not summary.endswith('.'):
                summary += '.'
            
            # Capitalize the first letter
            summary = summary[0].upper() + summary[1:]
            
            return summary
        except Exception as e:
            logger.error(f"Error post-processing summary: {str(e)}")
            return summary
    
    def extract_action_items(self, text):
        """Extract action items from the text."""
        try:
            doc = self.nlp(text)
            action_items = []
            
            # Extract using patterns
            for pattern in self.action_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    # Get the full match if there's only one group, otherwise get the second group
                    action_text = match.group(2) if match.lastindex > 1 else match.group(1)
                    action_text = action_text.strip()
                    
                    # Skip if too short
                    if len(action_text.split()) < 3:
                        continue
                    
                    # Extract entities from the action item
                    action_doc = self.nlp(action_text)
                    assignee = None
                    deadline = None
                    
                    for ent in action_doc.ents:
                        if ent.label_ == "PERSON":
                            assignee = ent.text
                        elif ent.label_ in ["DATE", "TIME"]:
                            deadline = ent.text
                    
                    # Only add if not already present
                    if not any(item["text"].lower() == action_text.lower() for item in action_items):
                        action_items.append({
                            "text": action_text,
                            "assignee": assignee,
                            "deadline": deadline,
                            "status": "pending"
                        })
            
            logger.info(f"Extracted {len(action_items)} action items")
            return action_items
            
        except Exception as e:
            logger.error(f"Error extracting action items: {str(e)}")
            return [] 