from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging
from .utils import calculate_optimal_summary_length

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextSummarizer:
    """AI-powered text summarization using HuggingFace transformers"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Predefined model configurations
        self.model_configs = {
            "facebook/bart-large-cnn": {
                "max_input_length": 1024,
                "default_max_length": 150,
                "default_min_length": 30
            },
            "t5-small": {
                "max_input_length": 512,
                "default_max_length": 100,
                "default_min_length": 20
            },
            "google/pegasus-xsum": {
                "max_input_length": 512,
                "default_max_length": 120,
                "default_min_length": 25
            }
        }
    
    def load_model(self, model_name):
        """Load and cache a specific model"""
        if model_name not in self.models:
            try:
                logger.info(f"Loading model: {model_name}")
                
                if model_name == "t5-small":
                    # T5 requires special handling
                    self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
                    self.models[model_name] = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                else:
                    # Use pipeline for BART and Pegasus
                    self.models[model_name] = pipeline(
                        "summarization",
                        model=model_name,
                        tokenizer=model_name,
                        device=0 if self.device == "cuda" else -1
                    )
                
                logger.info(f"Successfully loaded {model_name}")
                
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}")
                raise e
    
    def chunk_text(self, text, max_length=1000, overlap=100):
        """Split long text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_length - overlap):
            chunk = ' '.join(words[i:i + max_length])
            chunks.append(chunk)
            
            # Break if we've reached the end
            if i + max_length >= len(words):
                break
        
        return chunks
    
    def summarize_with_t5(self, text, max_length=100, min_length=20):
        """Summarize text using T5 model"""
        model = self.models["t5-small"]
        tokenizer = self.tokenizers["t5-small"]
        
        # T5 requires "summarize: " prefix
        input_text = f"summarize: {text}"
        
        # Tokenize input
        inputs = tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        # Generate summary
        with torch.no_grad():
            summary_ids = model.generate(
                inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
        
        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def summarize(self, text, model_name="facebook/bart-large-cnn", max_length=None, min_length=None):
        """Generate summary using specified model"""
        try:
            # Load model if not already loaded
            self.load_model(model_name)
            
            # Get model configuration
            config = self.model_configs.get(model_name, {})
            max_input_length = config.get("max_input_length", 1024)
            
            # Set default lengths if not provided
            if max_length is None:
                max_length = config.get("default_max_length", 150)
            if min_length is None:
                min_length = config.get("default_min_length", 30)
            
            # Calculate optimal length based on text
            text_length = len(text.split())
            if text_length < 100:
                # Short text - adjust summary length
                max_length = min(max_length, text_length // 2)
                min_length = min(min_length, max_length // 2)
            
            # Handle long texts by chunking
            if text_length > max_input_length:
                chunks = self.chunk_text(text, max_input_length - 50)
                summaries = []
                
                for i, chunk in enumerate(chunks):
                    try:
                        if model_name == "t5-small":
                            chunk_summary = self.summarize_with_t5(
                                chunk, 
                                max_length=max_length // len(chunks), 
                                min_length=min_length // len(chunks)
                            )
                        else:
                            result = self.models[model_name](
                                chunk,
                                max_length=max_length // len(chunks),
                                min_length=min_length // len(chunks),
                                do_sample=False
                            )
                            chunk_summary = result[0]['summary_text']
                        
                        summaries.append(chunk_summary)
                        
                    except Exception as e:
                        logger.warning(f"Error summarizing chunk {i+1}: {str(e)}")
                        continue
                
                # Combine chunk summaries
                combined_summary = " ".join(summaries)
                
                # If combined summary is still too long, summarize it again
                if len(combined_summary.split()) > max_length:
                    if model_name == "t5-small":
                        return self.summarize_with_t5(combined_summary, max_length, min_length)
                    else:
                        result = self.models[model_name](
                            combined_summary,
                            max_length=max_length,
                            min_length=min_length,
                            do_sample=False
                        )
                        return result[0]['summary_text']
                
                return combined_summary
            
            else:
                # Text fits in single pass
                if model_name == "t5-small":
                    return self.summarize_with_t5(text, max_length, min_length)
                else:
                    result = self.models[model_name](
                        text,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False
                    )
                    return result[0]['summary_text']
        
        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            # Fallback to simple truncation
            words = text.split()[:max_length]
            return f"Error generating summary. Text preview: {' '.join(words)}..."
    
    def get_model_info(self, model_name):
        """Get information about a specific model"""
        return self.model_configs.get(model_name, {})
    
    def clear_cache(self):
        """Clear loaded models to free memory"""
        self.models.clear()
        self.tokenizers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model cache cleared")