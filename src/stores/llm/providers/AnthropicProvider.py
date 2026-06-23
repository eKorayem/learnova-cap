from ..LLMInterface import LLMInterface
from ..LLMEnums import AnthropicEnums
import logging
from anthropic import AnthropicBedrock
import json

class AnthropicProvider(LLMInterface):
    def __init__(self, 
                 aws_access_key: str, 
                 aws_secret_key: str, 
                 aws_region: str,
                 defualt_input_max_characters: int = 200000, 
                 defualt_generation_max_out_tokens: int = 4000,
                 default_generation_temperature: float = 0.1):

        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key
        self.aws_region = aws_region

        self.defualt_input_max_characters = defualt_input_max_characters
        self.defualt_generation_max_out_tokens = defualt_generation_max_out_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        self.enums = AnthropicEnums
        self.logger = logging.getLogger(__name__)

        # Initialize the AWS Bedrock Anthropic Client
        self.client = AnthropicBedrock(
            aws_access_key=self.aws_access_key,
            aws_secret_key=self.aws_secret_key,
            aws_region=self.aws_region,
        )

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id
        self.logger.info(f"Anthropic generation model set: {model_id}")

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.logger.warning("AnthropicProvider does not support embeddings. Use Jina/SentenceTransformers.")
        pass

    def process_text(self, text: str):
        return text[:self.defualt_input_max_characters + 1].strip()

    def generate_text(self, prompt: str, chat_history: list = None,
                      max_output_tokens: int = None,
                      temperature: float = None) -> str:
        
        if not self.client or not self.generation_model_id:
            self.logger.error("AnthropicProvider: client or model not initialized")
            return None

        if chat_history is None:
            chat_history = []

        # Anthropic requires the 'system' prompt to be a top-level parameter, 
        # not part of the messages array. We must extract it from the chat_history.
        system_prompt = ""
        filtered_messages = []
        
        for msg in chat_history:
            if msg["role"] == self.enums.SYSTEM.value:
                system_prompt += msg["content"] + "\n\n"
            else:
                filtered_messages.append(msg)

        filtered_messages.append(self.construct_prompt(prompt, self.enums.USER.value))

        out_tokens = max_output_tokens if max_output_tokens is not None else self.defualt_generation_max_out_tokens
        temp = temperature if temperature is not None else self.default_generation_temperature

        try:
            response = self.client.messages.create(
                model=self.generation_model_id,
                system=system_prompt.strip() if system_prompt else None,
                messages=filtered_messages,
                max_tokens=out_tokens,
                temperature=temp
            )
            return response.content[0].text
            
        except Exception as e:
            self.logger.error(f"Anthropic generate_text failed: {e}")
            return None

    def construct_prompt(self, prompt: str, role: str):
        return {"role": role, "content": self.process_text(prompt)}

    def embed_text(self, text: str, document_type: str = None):
        raise NotImplementedError("AnthropicProvider does not support embeddings.")

    async def generate_structured_response(self, system_prompt: str, user_prompt: str, response_schema: dict):
        if not self.client or not self.generation_model_id:
            self.logger.error("AnthropicProvider: client or model not initialized")
            return None

        # Anthropic Claude 3 is excellent at strict JSON when using aggressive prompting
        schema_instruction = f"\n\nYou MUST return ONLY valid JSON matching this exact schema. Do not include markdown formatting, preambles, or explanations.\n{json.dumps(response_schema)}"
        
        messages = [
            {"role": "user", "content": user_prompt + schema_instruction}
        ]

        try:
            response = self.client.messages.create(
                model=self.generation_model_id,
                system=system_prompt,
                messages=messages,
                max_tokens=self.defualt_generation_max_out_tokens,
                temperature=self.default_generation_temperature
            )
            
            result = response.content[0].text
            print(f"\n=== RAW ANTHROPIC RESPONSE ===\n{result}\n========================\n", flush=True)
            return result

        except Exception as e:
            self.logger.error(f"AnthropicProvider structured generation error: {e}")
            return None