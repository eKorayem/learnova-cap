from pydantic import Field
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

    # ======================= LLM CONFIG (RAG Chat) =======================

    GENERATION_BACKEND: str
    GENERATION_MODEL_ID: Optional[str] = None
    EMBEDDING_BACKEND: str

    # Use Optional for anything that defaults to None
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_URL: Optional[str] = None
    COHERE_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    NARAYA_API_KEY: Optional[str] = None

    EMBEDDING_MODEL_ID: Optional[str] = None
    EMBEDDING_MODEL_SIZE: Optional[int] = None

    INPUT_DAFAULT_MAX_CHARACTERS: Optional[int] = None
    GENERATION_DAFAULT_MAX_TOKENS: Optional[int] = None
    
    # Add these new fields:
    STRUCTURE_TEMPERATURE: float = Field(default=0.1, env="STRUCTURE_TEMPERATURE")
    EXTRACTION_TEMPERATURE: float = Field(default=0.1, env="EXTRACTION_TEMPERATURE")
    QUESTION_TEMPERATURE: float = Field(default=0.3, env="QUESTION_TEMPERATURE")
    RAG_CHAT_TEMPERATURE: float = Field(default=0.6, env="RAG_CHAT_TEMPERATURE")
    GENERATION_DAFAULT_TEMPERATURE: float = Field(default=0.1, env="GENERATION_DAFAULT_TEMPERATURE")

    
    JINA_API_KEY: Optional[str] = None
    JINA_API_URL: str = "https://api.jina.ai/v1"

    DEEPSEEK_API_KEY: Optional[str] = None

    # ======================= AWS Anthropic Config =======================
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION_NAME: str = "us-east-1" # Default region for Bedrock

    # ======================= Chunking Config =======================

    RAG_CHUNK_SIZE: int = 100
    RAG_OVERLAP_SIZE: int = 20
    STRUCTURE_CHUNK_SIZE: int = 1000
    STRUCTURE_OVERLAP_SIZE: int = 100

    # ======================= Structure Analysis Config =======================

    STRUCTURE_BACKEND: str = "OPENROUTER"
    STRUCTURE_MODEL_ID: str = "google/gemini-2.0-flash-exp:free"

    # ======================= Question Generation Config =======================

    QUESTION_GENERATION_BACKEND: str = "OPENROUTER"
    QUESTION_GENERATION_MODEL_ID: str = "qwen/qwen-2.5-72b-instruct:free"
    QUESTION_OPENAI_API_KEY: Optional[str] = None    # OpenRouter API key
    QUESTION_CHUNK_SIZE: int = 1500       
    QUESTION_OVERLAP_SIZE: int = 150      

    # ======================= Exam Grading Config =======================

    GRADING_BACKEND: str = "OPENROUTER"
    GRADING_MODEL_ID: str = "meta-llama/llama-3.3-70b-instruct:free"

    # ======================= VectorDB CONFIG =======================

    VECTOR_DB_BACKEND: str
    VECTOR_DB_PATH: Optional[str] = None   # Made optional so Docker can ignore it
    VECTOR_DB_URL: Optional[str] = None
    VECTOR_DB_DISTANCE_METHOD: Optional[str] = None

    # ============================ Template Configurations ============================

    DEFAULT_LAN: str = "en"
    PRIMARY_LAN: str = "en"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ============================ Security Configurations ============================
    AI_SHARED_SECRET: str

    # ============================ Webhook Configurations ============================
    LEARNOVA_BACKEND_URL: str = "https://tightness-pavestone-displease.ngrok-free.dev/"


def get_settings():
    return Settings()