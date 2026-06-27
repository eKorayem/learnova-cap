from ..LLMInterface import LLMInterface
import logging
from openai import OpenAI
import os


class DeepSeekProvider(LLMInterface):

    def __init__(self, api_key: str,
                 defualt_input_max_characters: int = 4000000, # Approximating 1M tokens
                 defualt_generation_max_out_tokens: int = 8000,
                 default_generation_temperature: float = 0.1):

        self.api_key = api_key
        self.defualt_input_max_characters = defualt_input_max_characters
        self.defualt_generation_max_out_tokens = defualt_generation_max_out_tokens
        self.default_generation_temperature = default_generation_temperature
        self.generation_model_id = None
        
        self.logger = logging.getLogger(__name__)

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id
        self.logger.info(f"DeepSeek generation model set: {model_id}")

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.logger.warning("DeepSeekProvider does not currently support embeddings via this interface structure.")
        # Deepseek does not natively provide embeddings in the same way, usually paired with something else.
        pass

    def process_text(self, text: str):
        return text[:self.defualt_input_max_characters + 1].strip()

    def generate_text(self, prompt: str, chat_history: list = None,
                      max_output_tokens: int = None,
                      temperature: float = None) -> str:
        
        if not self.generation_model_id:
             raise ValueError("Generation model ID must be set before calling generate_text.")
             
        if chat_history is None:
            chat_history = []

        messages = chat_history.copy()
        messages.append(self.construct_prompt(prompt, "user"))

        out_tokens = max_output_tokens if max_output_tokens is not None else self.defualt_generation_max_out_tokens
        temp = temperature if temperature is not None else self.default_generation_temperature

        try:
            response = self.client.chat.completions.create(
                model=self.generation_model_id,
                messages=messages,
                temperature=temp,
                max_tokens=out_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"DeepSeek generate_text failed: {e}")
            return ""

    def construct_prompt(self, prompt: str, role: str):
        return {"role": role, "content": self.process_text(prompt)}

    def embed_text(self, text: str, document_type: str = None):
         raise NotImplementedError("DeepSeekProvider does not support embeddings.")

    async def embed_batch_async(self, texts: list, batch_size: int = 100):
        raise NotImplementedError("DeepSeekProvider does not support batch embeddings.")

    async def generate_structured_response(self, system_prompt: str, user_prompt: str, response_schema: dict, temperature: float = None):
        if not self.client or not self.generation_model_id:
            self.logger.error("DeepSeekProvider: client or model not initialized")
            return None

        import json
        schema_instruction = f"\n\nYou MUST return ONLY valid JSON matching this schema:\n{json.dumps(response_schema)}"
        
        messages = [
            {"role": "system", "content": system_prompt + schema_instruction},
            {"role": "user", "content": user_prompt}
        ]

        temp = temperature if temperature is not None else self.default_generation_temperature # <-- ADDED

        try:
            response = self.client.chat.completions.create(
                model=self.generation_model_id,
                messages=messages,
                max_tokens=self.defualt_generation_max_out_tokens,
                temperature=temp, # <-- UPDATED
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"DeepSeekProvider structured generation error: {e}")
            return None