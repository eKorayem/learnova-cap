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
    OPENROUTER_API_KEY: Optional[str] = None
    
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
    # Switched off the ":free" OpenRouter tier. Free-tier models are shared,
    # rate-limited, get renamed/deprecated without notice, and are less
    # reliable at strict JSON-schema compliance — all of which contributed
    # to the "sometimes works, sometimes doesn't" extraction problem.
    # Gemini 2.5 Flash-Lite has a 1M-token context window, so the full
    # reduced document text fits in one (or very few) batched calls.
    STRUCTURE_BACKEND: str = "OPENROUTER"
    STRUCTURE_MODEL_ID: str = "google/gemini-2.5-flash-lite"

    # Per-batch input budget for StructureController (see _split_into_batches).
    # 60,000 chars comfortably fits Flash-Lite's context with room for the
    # prompt/schema; raise this further since the model supports ~1M tokens,
    # or lower it if you switch STRUCTURE_BACKEND to a small-context model.
    STRUCTURE_MAX_INPUT_CHARS_PER_BATCH: int = 60000
    STRUCTURE_MAX_BATCHES: int = 12
    STRUCTURE_BATCH_SLEEP_SECONDS: float = 2

    # ======================= Question Generation Config =======================
    # Gemini 2.5 Flash: better instruction-following/reasoning than Lite,
    # still cheap, needed for generating plausible distractors/questions.
    QUESTION_GENERATION_BACKEND: str = "OPENROUTER"
    QUESTION_GENERATION_MODEL_ID: str = "google/gemini-2.5-flash"
    QUESTION_OPENAI_API_KEY: Optional[str] = None    # OpenRouter API key
    QUESTION_CHUNK_SIZE: int = 1500       
    QUESTION_OVERLAP_SIZE: int = 150      

    # ======================= Exam Grading Config =======================
    # Grading is low-volume (once per submission) and benefits most from
    # reasoning quality, so it's worth spending a little more here.
    GRADING_BACKEND: str = "OPENROUTER"
    GRADING_MODEL_ID: str = "google/gemini-2.5-flash"

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