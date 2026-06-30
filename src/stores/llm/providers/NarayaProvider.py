from ..LLMInterface import LLMInterface
from ..LLMEnums import OpenAIEnums
from openai import OpenAI
import logging
import json

class NarayaProvider(LLMInterface):
    def __init__(self, api_key: str,
                 defualt_input_max_characters: int = 200000,
                 defualt_generation_max_out_tokens: int = 4000,
                 default_generation_temperature: float = 0.1):

        self.api_key = api_key
        self.defualt_input_max_characters = defualt_input_max_characters
        self.defualt_generation_max_out_tokens = defualt_generation_max_out_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        # Naraya uses the OpenAI SDK with a custom base URL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://gateway.naraya.ai/v1"
        )

        self.enums = OpenAIEnums
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id
        self.logger.info(f"Naraya generation model set: {model_id}")

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.logger.warning("NarayaProvider does not support embeddings. Use Jina/SentenceTransformers.")
        pass

    def process_text(self, text: str):
        return text[:self.defualt_input_max_characters + 1].strip()

    def generate_text(self, prompt: str, chat_history: list = None,
                      max_output_tokens: int = None,
                      temperature: float = None):

        if not self.client or not self.generation_model_id:
            self.logger.error("NarayaProvider: client or model not initialized")
            return None

        if chat_history is None:
            chat_history = []

        max_output_tokens = max_output_tokens if max_output_tokens else self.defualt_generation_max_out_tokens
        temperature = temperature if temperature is not None else self.default_generation_temperature

        messages = chat_history.copy()
        messages.append(self.construct_prompt(prompt=prompt, role=self.enums.USER.value))

        try:
            response = self.client.chat.completions.create(
                model=self.generation_model_id,
                messages=messages,
                max_tokens=max_output_tokens,
                temperature=temperature
            )

            if not response or not response.choices or len(response.choices) == 0:
                self.logger.error("Error while generating text with Naraya")
                return None

            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Naraya generate_text failed: {e}")
            return None

    def embed_text(self, text: str, document_type: str = None):
        raise NotImplementedError("NarayaProvider does not support embeddings.")

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": self.process_text(prompt)
        }

    async def generate_structured_response(self, system_prompt: str, user_prompt: str, response_schema: dict, temperature: float = None):
        if not self.client or not self.generation_model_id:
            self.logger.error("NarayaProvider: client or model not initialized")
            return None

        schema_instruction = f"\n\nYou MUST return ONLY valid JSON matching this schema:\n{json.dumps(response_schema)}"

        messages = [
            {"role": "system", "content": system_prompt + schema_instruction},
            {"role": "user", "content": user_prompt}
        ]

        temp = temperature if temperature is not None else self.default_generation_temperature

        try:
            # Naraya supports the JSON object format natively
            response = self.client.chat.completions.create(
                model=self.generation_model_id,
                messages=messages,
                max_tokens=self.defualt_generation_max_out_tokens,
                temperature=temp,
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            print(f"\n=== RAW NARAYA RESPONSE ===\n{result}\n========================\n", flush=True)
            return result

        except Exception as e:
            self.logger.error(f"NarayaProvider structured generation error: {e}")
            return None