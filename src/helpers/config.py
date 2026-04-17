from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List

class Settings(BaseSettings):

    APP_NAME: str
    APP_VERSION: str

    FILE_ALLOWED_TYPES: list
    FILE_ALLOWED_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int

    MONGODB_URL: str
    MONGODB_DATABASE: str

    # ======================= LLM CONFIG =======================

    GENERATION_BACKEND: str
    EMBEDDING_BACKEND: str

    # Use Optional for anything that defaults to None
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_URL: Optional[str] = None
    COHERE_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None

    GENERATION_MODEL_ID: Optional[str] = None
    EMBEDDING_MODEL_ID: Optional[str] = None
    EMBEDDING_MODEL_SIZE: Optional[int] = None

    INPUT_DAFAULT_MAX_CHARACTERS: Optional[int] = None
    GENERATION_DAFAULT_MAX_TOKENS: Optional[int] = None
    GENERATION_DAFAULT_TEMPERATURE: Optional[float] = None

    JINA_API_KEY: Optional[str] = None
    JINA_API_URL: str = "https://api.jina.ai/v1"

    # ======================= Chunking Config =======================

    RAG_CHUNK_SIZE: int = 100
    RAG_OVERLAP_SIZE: int = 20
    STRUCTURE_CHUNK_SIZE: int = 1000
    STRUCTURE_OVERLAP_SIZE: int = 100

    # ======================= Question Generation Config =======================

    QUESTION_GENERATION_BACKEND: str = "OPENROUTER"
    QUESTION_GENERATION_MODEL_ID: str = "qwen/qwen3.6-plus-preview:free"
    QUESTION_OPENAI_API_KEY: Optional[str] = None    # OpenRouter API key
    QUESTION_CHUNK_SIZE: int = 1500        
    QUESTION_OVERLAP_SIZE: int = 150       

    # ======================= VectorDB CONFIG =======================

    VECTOR_DB_BACKEND: str
    VECTOR_DB_PATH: Optional[str] = None   # Made optional so Docker can ignore it
    VECTOR_DB_URL: Optional[str] = None 
    VECTOR_DB_DISTANCE_METHOD: Optional[str] = None

    # ============================ Template Configurations ============================

    DEFAULT_LAN: str = "en"
    PRIMARY_LAN: str = "en"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


def get_settings():
    return Settings()