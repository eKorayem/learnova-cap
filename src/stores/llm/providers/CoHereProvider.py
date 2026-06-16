from ..LLMInterface import LLMInterface
from ..LLMEnums import CoHereEnums, DocumentTypeEnums
import cohere
import logging


class CoHereProvider(LLMInterface):
    def __init__(self, api_key: str, 
                    defualt_input_max_characters: int=1000, 
                    defualt_generation_max_out_tokens: int=1000,
                    default_generation_temperature: float=0.1):
        
        self.api_key = api_key
        
        self.defualt_input_max_characters = defualt_input_max_characters
        self.defualt_generation_max_out_tokens = defualt_generation_max_out_tokens
        self.default_generation_temperature = default_generation_temperature
        
        self.generation_model_id = None
        
        self.embedding_model_id = None
        self.embedding_size = None
        
        self.client = cohere.Client(api_key=self.api_key)
        
        self.enums = CoHereEnums

        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id
        
    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
        
        
    def process_text(self, text: str):
        return text[:self.defualt_input_max_characters+1].strip()
    
    def generate_text(self, prompt: str, chat_history: list=[],
                      max_output_tokens: int=None,
                      temperature: float = None):
        
        if not self.client:
            self.logger.error("CoHere client was NOT set")
            return None
        
        if not self.generation_model_id:
            self.logger.error("Generation model for CoHere was NOT set")
            return None
        
        max_output_tokens = max_output_tokens if max_output_tokens else self.defualt_generation_max_out_tokens
        temperature = temperature if temperature else self.default_generation_temperature
        
        response = self.client.chat(
            model=self.generation_model_id,
            chat_history=chat_history,
            message=self.process_text(prompt),
            temperature=temperature,
            max_tokens=max_output_tokens
        )
        
        if not response or not response.text:
            self.logger.error("No response from CoHere API")
            return None
        
        return response.text
    
    def construct_prompt(self, prompt: str, role: str):
        return {
            "role" : role, 
            "text" : self.process_text(prompt)
        }
    
    def embed_text(self, text: str, document_type: str=None):
        if not self.client:
            self.logger.error("CoHere client was NOT set")
            return None
        
        if not self.embedding_model_id:
            self.logger.error("Embedding model for CoHere was NOT set")
            return None
        
        input_type = CoHereEnums.DOCUMENT.value if document_type==DocumentTypeEnums.DOCUMENT else CoHereEnums.QUERY.value
        response = self.client.embed(
            texts=[self.process_text(text=text)],
            model=self.embedding_model_id,
            input_type= input_type,
            embedding_types=["float"]
            # output_dimension=self.,
        )
        
        if not response or not response.embeddings or len(response.embeddings)==0 or not response.embeddings.float:
            self.logger.error("Error while embedding text with CoHere")
            return None

        return response.embeddings[0]
    
    async def generate_structured_response(self, system_prompt: str, user_prompt: str, response_schema: dict):
        if not self.client or not self.generation_model_id:
            self.logger.error("CoHereProvider: client or model not initialized")
            return None

        import json
        # Cohere requires aggressive prompting to ensure clean JSON output
        schema_instruction = f"\n\nYou MUST return ONLY valid JSON matching this schema. Do not include markdown formatting or explanations.\n{json.dumps(response_schema)}"
        
        combined_prompt = f"System Rules: {system_prompt}\n\nUser Request: {user_prompt}{schema_instruction}"

        try:
            # Running Cohere's sync client in an async wrapper
            import asyncio
            response = await asyncio.to_thread(
                self.client.chat,
                model=self.generation_model_id,
                message=combined_prompt,
                temperature=self.default_generation_temperature,
                max_tokens=self.defualt_generation_max_out_tokens
            )

            if not response or not response.text:
                self.logger.error("CoHereProvider: empty response")
                return None

            # Clean the text just in case Cohere adds markdown
            cleaned_text = response.text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:].strip()
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3].strip()

            return cleaned_text

        except Exception as e:
            self.logger.error(f"CoHereProvider structured generation error: {e}")
            return None