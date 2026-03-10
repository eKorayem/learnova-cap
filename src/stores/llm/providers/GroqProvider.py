from ..LLMInterface import LLMInterface
from ..LLMEnums import GroqEnums, DocumentTypeEnums
import groq
import logging

class GroqProvider(LLMInterface):
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
        
        self.client = groq.Client(api_key=self.api_key)
        
        self.logger = logging.getLogger(__name__)
    
    
    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id
        
    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.logger.warning("Groq does not support embeddings. This method does nothing.")
        return None
        
    def process_text(self, text: str):
        return text[:self.defualt_input_max_characters+1].strip()

    def generate_text(self, prompt: str, chat_history: list=[],
                      max_output_tokens: int=None,
                      temperature: float = None):
        if not self.client:
            self.logger.error("Groq client was NOT set")
            return None
        
        if not self.generation_model_id:
            self.logger.error("Generation model for Groq was NOT set")
            return None
        
        max_output_tokens = max_output_tokens if max_output_tokens else self.defualt_generation_max_out_tokens
        temperature = temperature if temperature else self.default_generation_temperature
        
        messages = chat_history.copy()
        messages.append(
            self.construct_prompt(prompt=prompt, role=GroqEnums.USER.value)
        )
        
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.generation_model_id,
            max_tokens=max_output_tokens,
            temperature=temperature
        )
        
        if not response or not response.choices:
            self.logger.error("Error while generating text with Groq")
            return None

        return response.choices[0].message.content
    
    def embed_text(self, text: str, document_type: str=None):
        self.logger.warning("Groq does not provide embeddings. Returning None.")
        raise NotImplementedError("Groq does not support embeddings. Use LocalProvider instead.")
        
    
    
    
    def construct_prompt(self, prompt: str, role: str):
        return {
            "role" : role, 
            "content" : self.process_text(prompt)
        }