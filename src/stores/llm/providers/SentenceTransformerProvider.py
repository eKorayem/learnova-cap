from ..LLMInterface import LLMInterface
import logging


class SentenceTransformerProvider(LLMInterface):
    def __init__(self,
                 api_key: str = None,
                 defualt_input_max_characters: int = 1000,
                 defualt_generation_max_out_tokens: int = 1000,
                 default_generation_temperature: float = 0.1):

        self.api_key = api_key

        self.defualt_input_max_characters = defualt_input_max_characters
        self.defualt_generation_max_out_tokens = defualt_generation_max_out_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None
        self.embedding_model = None
        self.device = None

        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.logger.warning("SentenceTransformerProvider does not support text generation.")
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            self.embedding_model_id = model_id
            self.embedding_size = embedding_size

            # Use GPU if available, fall back to CPU automatically
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Loading embedding model '{model_id}' on {self.device.upper()}")

            self.embedding_model = SentenceTransformer(model_id, device=self.device)
            self.logger.info(f"Embedding model loaded successfully on {self.device.upper()}")

        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install it with: pip install sentence-transformers"
            )
        except Exception as e:
            self.logger.error(f"Error loading embedding model {model_id}: {e}")
            raise

    def process_text(self, text: str):
        return text[:self.defualt_input_max_characters + 1].strip()

    def generate_text(self, prompt: str, chat_history: list = [],
                      max_output_tokens: int = None,
                      temperature: float = None):
        raise NotImplementedError(
            "SentenceTransformerProvider does not support text generation. "
            "Use GroqProvider or OpenAIProvider instead."
        )

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": self.process_text(prompt)
        }

    def embed_text(self, text: str, document_type: str = None):
        """
        Embed a single text. Used for query embedding during RAG search.
        For indexing many chunks at once, use embed_batch() instead.
        """
        if not self.embedding_model:
            self.logger.error("Embedding model not loaded.")
            return None

        try:
            processed_text = self.process_text(text)
            embedding = self.embedding_model.encode(
                processed_text,
                convert_to_numpy=True
            )
            return embedding.tolist()

        except Exception as e:
            self.logger.error(f"Error generating single embedding: {e}")
            return None

    def embed_batch(self, texts: list, batch_size: int = 64):
        """
        Embed a list of texts in one GPU-accelerated batch call.
        Used for indexing chunks — much faster than calling embed_text() in a loop.
        
        batch_size=64 works well for GTX 1650 (4GB VRAM).
        Reduce to 32 if you get CUDA out-of-memory errors.
        """
        if not self.embedding_model:
            self.logger.error("Embedding model not loaded.")
            return None

        if not texts:
            return []

        try:
            # Truncate all texts to max characters
            processed_texts = [self.process_text(text) for text in texts]

            self.logger.info(
                f"Batch embedding {len(processed_texts)} texts "
                f"on {self.device.upper()} with batch_size={batch_size}"
            )

            embeddings = self.embedding_model.encode(
                processed_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True
            )

            self.logger.info(f"Batch embedding complete — {len(embeddings)} vectors generated")
            return embeddings.tolist()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.logger.warning(
                    f"CUDA out of memory with batch_size={batch_size}. "
                    f"Retrying with batch_size={batch_size // 2}"
                )
                # Auto-retry with smaller batch size
                return self.embed_batch(texts, batch_size=batch_size // 2)
            self.logger.error(f"Error during batch embedding: {e}")
            return None

        except Exception as e:
            self.logger.error(f"Error during batch embedding: {e}")
            return None