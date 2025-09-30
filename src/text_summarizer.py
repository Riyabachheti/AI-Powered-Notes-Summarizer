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
        """Generate summary using specified model with enhanced logic for meaningful summaries"""

        try:
            # Preprocess text
            text = self.preprocess_text(text)
            
            # Load model if not already loaded
            self.load_model(model_name)
            
            # Get model configuration
            config = self.model_configs.get(model_name, {})
            text_length = len(text.split())
            
            logger.info(f"Processing text with {text_length} words using {model_name}")
            
            # Calculate adaptive lengths if not provided
            if max_length is None or min_length is None:
                calc_min, calc_max = self.calculate_adaptive_length(text_length, model_name)
                if max_length is None:
                    max_length = calc_max
                if min_length is None:
                    min_length = calc_min
            
            logger.info(f"Summary target: {min_length}-{max_length} words")
            
            # Handle different text lengths
            max_input_words = config.get("max_input_words", 400)
            
            if text_length <= max_input_words:
                # Single pass summarization for shorter texts
                logger.info("Processing as single chunk")
                if model_name == "t5-small":
                    summary = self.summarize_with_t5(text, max_length, min_length)
                else:
                    result = self.models[model_name](
                        text,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False,
                        repetition_penalty=1.2,
                        length_penalty=1.0
                    )
                    summary = result[0]['summary_text']
                
            else:
                # Multi-chunk processing for longer texts
                logger.info(f"Text too long ({text_length} words), using semantic chunking")
                chunks = self.create_semantic_chunks(text, model_name)
                
                if not chunks:
                    logger.error("No valid chunks created")
                    return "Error: Could not process the text."
                
                summaries = []
                # Calculate per-chunk length (ensuring meaningful summaries)
                chunk_max = max(max_length // len(chunks), 60)  # Minimum 60 words per chunk
                chunk_min = max(min_length // len(chunks), 20)  # Minimum 20 words per chunk
                
                for i, chunk in enumerate(chunks):
                    try:
                        logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                        
                        if model_name == "t5-small":
                            chunk_summary = self.summarize_with_t5(chunk, chunk_max, chunk_min)
                        else:
                            result = self.models[model_name](
                                chunk,
                                max_length=chunk_max,
                                min_length=chunk_min,
                                do_sample=False,
                                repetition_penalty=1.2,
                                length_penalty=1.0
                            )
                            chunk_summary = result[0]['summary_text']
                        
                        if chunk_summary and chunk_summary.strip():
                            summaries.append(chunk_summary.strip())
                        
                    except Exception as e:
                        logger.warning(f"Error summarizing chunk {i+1}: {str(e)}")
                        # Continue with other chunks instead of failing completely
                        continue
                
                if not summaries:
                    logger.error("No chunk summaries were generated")
                    return "Error: Could not generate summaries for any text chunks."
                
                # Merge summaries coherently
                summary = self.merge_summaries_coherently(summaries, model_name, max_length)
            
            # Post-process summary
            summary = summary.strip()
            final_length = len(summary.split())
            
            logger.info(f"Generated summary with {final_length} words")
            
            # Validate summary quality
            if final_length < 10:
                logger.warning("Summary too short, attempting with relaxed parameters")
                # Retry with more relaxed parameters
                return self.summarize(text, model_name, max_length * 2, max(min_length, 30))
            
            return summary
        
        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            # Enhanced fallback - provide a meaningful excerpt
            sentences = text.split('.')[:3]  # First 3 sentences
            fallback = '. '.join(sentences).strip()
            if len(fallback) > 20:
                return f"{fallback}..."
            else:
                words = text.split()[:100]
                return f"Processing error. Text preview: {' '.join(words)}..."
    # def summarize(self, text, model_name="facebook/bart-large-cnn", max_length=None, min_length=None):
    #     """Generate summary using specified model"""
    #     try:
    #         # Load model if not already loaded
    #         self.load_model(model_name)
            
    #         # Get model configuration
    #         config = self.model_configs.get(model_name, {})
    #         max_input_length = config.get("max_input_length", 1024)
            
    #         # Set default lengths if not provided
    #         if max_length is None:
    #             max_length = config.get("default_max_length", 150)
    #         if min_length is None:
    #             min_length = config.get("default_min_length", 30)
            
    #         # Calculate optimal length based on text
    #         text_length = len(text.split())
    #         if text_length < 100:
    #             # Short text - adjust summary length
    #             max_length = min(max_length, text_length // 2)
    #             min_length = min(min_length, max_length // 2)
            
    #         # Handle long texts by chunking
    #         if text_length > max_input_length:
    #             chunks = self.chunk_text(text, max_input_length - 50)
    #             summaries = []
                
    #             for i, chunk in enumerate(chunks):
    #                 try:
    #                     if model_name == "t5-small":
    #                         chunk_summary = self.summarize_with_t5(
    #                             chunk, 
    #                             max_length=max_length // len(chunks), 
    #                             min_length=min_length // len(chunks)
    #                         )
    #                     else:
    #                         result = self.models[model_name](
    #                             chunk,
    #                             max_length=max_length // len(chunks),
    #                             min_length=min_length // len(chunks),
    #                             do_sample=False
    #                         )
    #                         chunk_summary = result[0]['summary_text']
                        
    #                     summaries.append(chunk_summary)
                        
    #                 except Exception as e:
    #                     logger.warning(f"Error summarizing chunk {i+1}: {str(e)}")
    #                     continue
                
    #             # Combine chunk summaries
    #             combined_summary = " ".join(summaries)
                
    #             # If combined summary is still too long, summarize it again
    #             if len(combined_summary.split()) > max_length:
    #                 if model_name == "t5-small":
    #                     return self.summarize_with_t5(combined_summary, max_length, min_length)
    #                 else:
    #                     result = self.models[model_name](
    #                         combined_summary,
    #                         max_length=max_length,
    #                         min_length=min_length,
    #                         do_sample=False
    #                     )
    #                     return result[0]['summary_text']
                
    #             return combined_summary
            
    #         else:
    #             # Text fits in single pass
    #             if model_name == "t5-small":
    #                 return self.summarize_with_t5(text, max_length, min_length)
    #             else:
    #                 result = self.models[model_name](
    #                     text,
    #                     max_length=max_length,
    #                     min_length=min_length,
    #                     do_sample=False
    #                 )
    #                 return result[0]['summary_text']
        
    #     except Exception as e:
    #         logger.error(f"Error during summarization: {str(e)}")
    #         # Fallback to simple truncation
    #         words = text.split()[:max_length]
    #         return f"Error generating summary. Text preview: {' '.join(words)}..."
    
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