from ..LLMInterface import LLMInterface
from openai import OpenAI
import logging
from ..LLMEnums import OpenAIEnums


class OpenRouterProvider(LLMInterface):

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self,
                 api_key: str,
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

        # Always points to OpenRouter — never conflicts with existing OpenAI setup
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.OPENROUTER_BASE_URL
        )

        self.enums = OpenAIEnums
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.logger.warning("OpenRouterProvider: embedding not supported. Use SentenceTransformerProvider instead.")
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        return text[:self.defualt_input_max_characters + 1].strip()

    def generate_text(self, prompt: str, chat_history: list = [],
                      max_output_tokens: int = None,
                      temperature: float = None):

        if not self.client:
            self.logger.error("OpenRouterProvider: client not initialized")
            return None

        if not self.generation_model_id:
            self.logger.error("OpenRouterProvider: generation model not set")
            return None

        max_output_tokens = max_output_tokens if max_output_tokens else self.defualt_generation_max_out_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        messages = chat_history.copy()
        messages.append(
            self.construct_prompt(prompt=prompt, role=OpenAIEnums.USER.value)
        )

        try:
            response = self.client.chat.completions.create(
                model=self.generation_model_id,
                messages=messages,
                max_tokens=max_output_tokens,
                temperature=temperature
            )

            if not response or not response.choices or len(response.choices) == 0:
                self.logger.error("OpenRouterProvider: empty response")
                return None

            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"OpenRouterProvider: error during generation: {e}")
            return None

    def embed_text(self, text: str, document_type: str = None):
        raise NotImplementedError(
            "OpenRouterProvider does not support embeddings. "
            "Use SentenceTransformerProvider instead."
        )

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": self.process_text(prompt)
        }