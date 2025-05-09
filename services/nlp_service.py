import spacy
import re
from datetime import datetime

class NLPService:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
        # Custom patterns for action item detection
        self.action_patterns = [
            r"(?i)need to\s+(.+?)(?=\.|$)",
            r"(?i)should\s+(.+?)(?=\.|$)",
            r"(?i)will\s+(.+?)(?=\.|$)",
            r"(?i)must\s+(.+?)(?=\.|$)",
            r"(?i)action item:\s*(.+?)(?=\.|$)",
            r"(?i)todo:\s*(.+?)(?=\.|$)"
        ]
    
    def extract_action_items(self, text):
        """Extract action items from the text."""
        doc = self.nlp(text)
        action_items = []
        
        # Extract using patterns
        for pattern in self.action_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                action_text = match.group(1).strip()
                
                # Extract entities from the action item
                action_doc = self.nlp(action_text)
                assignee = None
                deadline = None
                
                for ent in action_doc.ents:
                    if ent.label_ == "PERSON":
                        assignee = ent.text
                    elif ent.label_ in ["DATE", "TIME"]:
                        deadline = ent.text
                
                action_items.append({
                    "text": action_text,
                    "assignee": assignee,
                    "deadline": deadline,
                    "status": "pending"
                })
        
        return action_items
    
    def generate_summary(self, text):
        """Generate a summary of the meeting transcript."""
        doc = self.nlp(text)
        
        # Extract key sentences based on important entities and noun phrases
        important_sentences = []
        sentence_scores = {}
        
        for sent in doc.sents:
            score = 0
            
            # Score based on entities
            for ent in sent.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "TIME"]:
                    score += 1
            
            # Score based on noun phrases
            for chunk in sent.noun_chunks:
                score += 0.5
            
            # Score based on key phrases
            key_phrases = ["conclusion", "summary", "decision", "agreed", "important"]
            sent_text = sent.text.lower()
            for phrase in key_phrases:
                if phrase in sent_text:
                    score += 2
            
            sentence_scores[sent] = score
        
        # Select top scoring sentences
        sorted_sentences = sorted(
            sentence_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Take top 3 sentences or 20% of total sentences, whichever is greater
        num_sentences = max(3, int(len(doc.sents) * 0.2))
        summary_sentences = sorted_sentences[:num_sentences]
        
        # Sort sentences by their original order
        summary_sentences.sort(key=lambda x: text.index(x[0].text))
        
        # Combine sentences into summary
        summary = " ".join([sent[0].text for sent in summary_sentences])
        
        return summary 