from ..LLMInterface import LLMInterface
import logging

class SentenceTransformerProvider(LLMInterface):
    def __init__(self, 
                 api_key: str = None,  # Not used, but kept for interface consistency
                 defualt_input_max_characters: int = 1000, 
                 defualt_generation_max_out_tokens: int = 1000,
                 default_generation_temperature: float = 0.1):
        
        # Keep same parameters as other providers for consistency
        self.api_key = api_key  # Not used for local
        
        self.defualt_input_max_characters = defualt_input_max_characters
        self.defualt_generation_max_out_tokens = defualt_generation_max_out_tokens
        self.default_generation_temperature = default_generation_temperature
        
        self.generation_model_id = None
        
        self.embedding_model_id = None
        self.embedding_size = None
        self.embedding_model = None  # Will hold the sentence-transformers model
        
        self.logger = logging.getLogger(__name__)
    
    def set_generation_model(self, model_id: str):
        """Local provider doesn't support text generation"""
        self.logger.warning("LocalProvider does not support text generation. This method does nothing.")
        self.generation_model_id = model_id  # Store it but don't use it
    
    def set_embedding_model(self, model_id: str, embedding_size: int):
        """Load the sentence-transformers model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            self.embedding_model_id = model_id
            self.embedding_size = embedding_size
            
            # Load the model
            self.logger.info(f"Loading local embedding model: {model_id}")
            self.embedding_model = SentenceTransformer(model_id)
            self.logger.info(f"Local embedding model loaded successfully")
            
        except ImportError:
            self.logger.error("sentence-transformers library not installed. Run: pip install sentence-transformers")
            raise ImportError("sentence-transformers is required for LocalProvider. Install it with: pip install sentence-transformers")
        except Exception as e:
            self.logger.error(f"Error loading embedding model {model_id}: {e}")
            raise
    
    def process_text(self, text: str):
        """Process text to fit within character limit"""
        return text[:self.defualt_input_max_characters + 1].strip()
    
    def generate_text(self, prompt: str, chat_history: list = [],
                      max_output_tokens: int = None,
                      temperature: float = None):
        """Local provider cannot generate text"""
        raise NotImplementedError("LocalProvider does not support text generation. Use GroqProvider or OpenAIProvider instead.")
    
    def construct_prompt(self, prompt: str, role: str):
        """Not used for embeddings, but required by interface"""
        return {
            "role": role,
            "content": self.process_text(prompt)
        }
    
    def embed_text(self, text: str, document_type: str = None):
        """Generate embeddings using sentence-transformers"""
        if not self.embedding_model:
            self.logger.error("Embedding model not loaded. Call set_embedding_model() first.")
            return None
        
        try:
            # Process text to fit character limit
            processed_text = self.process_text(text)
            
            # Generate embedding
            embedding = self.embedding_model.encode(processed_text)
            
            # Convert to list (sentence-transformers returns numpy array)
            return embedding.tolist()
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            return None