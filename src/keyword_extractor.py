from keybert import KeyBERT
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KeywordExtractor:
    """Extract keywords and key phrases from text using multiple methods"""
    
    def __init__(self):
        self.keybert = None
        self.nlp = None
        self._setup_models()
        self._download_nltk_data()
    
    def _setup_models(self):
        """Initialize KeyBERT and spaCy models"""
        try:
            logger.info("Loading KeyBERT model...")
            self.keybert = KeyBERT()
            logger.info("KeyBERT loaded successfully")
        except Exception as e:
            logger.error(f"Error loading KeyBERT: {str(e)}")
        
        try:
            logger.info("Loading spaCy model...")
            # Try to load English model, download if not available
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy English model not found, using basic model")
                self.nlp = None
            logger.info("spaCy loaded successfully")
        except Exception as e:
            logger.error(f"Error loading spaCy: {str(e)}")
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        except Exception as e:
            logger.warning(f"Error downloading NLTK data: {str(e)}")
    
    def extract_keywords_keybert(self, text, num_keywords=10):
        """Extract keywords using KeyBERT"""
        try:
            if not self.keybert:
                return []
            
            keywords = self.keybert.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_k=num_keywords,
                diversity=0.5
            )
            
            # Return only the keywords, not scores
            return [kw[0] for kw in keywords]
            
        except Exception as e:
            logger.error(f"Error extracting keywords with KeyBERT: {str(e)}")
            return []
    
    def extract_entities_spacy(self, text):
        """Extract named entities using spaCy"""
        try:
            if not self.nlp:
                return []
            
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'PRODUCT', 'WORK_OF_ART']:
                    entities.append(ent.text.strip())
            
            # Remove duplicates while preserving order
            seen = set()
            unique_entities = []
            for entity in entities:
                if entity.lower() not in seen:
                    seen.add(entity.lower())
                    unique_entities.append(entity)
            
            return unique_entities[:10]  # Return top 10
            
        except Exception as e:
            logger.error(f"Error extracting entities with spaCy: {str(e)}")
            return []
    
    def extract_keywords_nltk(self, text, num_keywords=10):
        """Extract keywords using NLTK (fallback method)"""
        try:
            # Tokenize and get POS tags
            tokens = word_tokenize(text.lower())
            pos_tags = pos_tag(tokens)
            
            # Get stopwords
            try:
                stop_words = set(stopwords.words('english'))
            except:
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
            
            # Extract meaningful words (nouns, adjectives, verbs)
            meaningful_words = []
            for word, pos in pos_tags:
                if (word.isalpha() and 
                    len(word) > 3 and 
                    word not in stop_words and
                    pos.startswith(('NN', 'JJ', 'VB', 'CD'))):  # Added CD for numbers
                    meaningful_words.append(word)
            
            # Also add important words without POS filtering as backup
            if len(meaningful_words) < num_keywords:
                backup_words = [w for w in tokens if w.isalpha() and len(w) > 4 and w not in stop_words]
                meaningful_words.extend(backup_words)
            
            # Count frequency
            word_freq = Counter(meaningful_words)
            
            # Return most common words
            return [word for word, count in word_freq.most_common(num_keywords)]
            
        except Exception as e:
            logger.error(f"Error extracting keywords with NLTK: {str(e)}")
            # Final fallback - simple word extraction
            words = text.lower().split()
            simple_keywords = []
            for word in words:
                clean_word = ''.join(c for c in word if c.isalpha())
                if len(clean_word) > 4 and clean_word not in ['artificial', 'intelligence', 'healthcare', 'technology']:
                    simple_keywords.append(clean_word)
            return list(set(simple_keywords))[:num_keywords]
    
    def extract_key_phrases_rake(self, text, num_phrases=5):
        """Extract key phrases using RAKE algorithm (simple implementation)"""
        try:
            # Simple RAKE implementation
            stop_words = set(stopwords.words('english'))
            sentences = text.split('.')
            
            phrases = []
            for sentence in sentences:
                words = word_tokenize(sentence.lower())
                # Filter out stop words and short words
                filtered_words = [w for w in words if w.isalpha() and len(w) > 3 and w not in stop_words]
                
                # Create phrases from consecutive words
                if len(filtered_words) >= 2:
                    for i in range(len(filtered_words) - 1):
                        phrase = ' '.join(filtered_words[i:i+2])
                        phrases.append(phrase)
            
            # Count phrase frequency
            phrase_freq = Counter(phrases)
            return [phrase for phrase, count in phrase_freq.most_common(num_phrases)]
            
        except Exception as e:
            logger.error(f"Error extracting phrases with RAKE: {str(e)}")
            return []
    
    def extract_keywords(self, text, num_keywords=10):
        """Main method to extract keywords using best available method"""
        all_keywords = []
        
        # Method 1: KeyBERT (primary method)
        try:
            keybert_keywords = self.extract_keywords_keybert(text, num_keywords)
            all_keywords.extend(keybert_keywords[:num_keywords//2])
        except Exception as e:
            logger.warning(f"KeyBERT extraction failed: {e}")
        
        # Method 2: Named Entities (spaCy)
        try:
            entities = self.extract_entities_spacy(text)
            all_keywords.extend(entities[:num_keywords//4])
        except Exception as e:
            logger.warning(f"spaCy extraction failed: {e}")
        
        # Method 3: NLTK fallback
        if len(all_keywords) < num_keywords:
            nltk_keywords = self.extract_keywords_nltk(text, num_keywords - len(all_keywords))
            all_keywords.extend(nltk_keywords)
        
        # Method 4: Key phrases
        if len(all_keywords) < num_keywords:
            phrases = self.extract_key_phrases_rake(text, 3)
            all_keywords.extend(phrases)
        
        # Final fallback if still no keywords
        if not all_keywords:
            # Simple frequency-based extraction
            words = text.lower().split()
            word_counts = Counter(word for word in words if len(word) > 4 and word.isalpha())
            all_keywords = [word for word, count in word_counts.most_common(num_keywords)]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in all_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower not in seen and len(keyword_lower) > 2:
                seen.add(keyword_lower)
                unique_keywords.append(keyword)
        
        return unique_keywords[:num_keywords]
    
    def get_keyword_statistics(self, text, keywords):
        """Get statistics about extracted keywords"""
        stats = {}
        text_lower = text.lower()
        
        for keyword in keywords:
            count = text_lower.count(keyword.lower())
            stats[keyword] = count
        
        return stats