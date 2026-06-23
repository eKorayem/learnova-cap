from enum import Enum

class LLMEnums(Enum):
    OPENAI = "OPENAI"
    COHERE = "COHERE"
    GROQ = "GROQ"
    SENTENCE_TRANSFORMER = "SENTENCE_TRANSFORMER"
    OPENROUTER = "OPENROUTER"         
    JINA = "JINA"  
    DEEPSEEK = "DEEPSEEK"  # <--- ADD THIS
    ANTHROPIC = "ANTHROPIC"

class OpenAIEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class CoHereEnums(Enum):
    SYSTEM = "SYSTEM"
    USER = "USER"
    ASSISTANT = "CHATBOT"

    DOCUMENT = "search_document"
    QUERY = "search_query"

class DocumentTypeEnums(Enum):
    DOCUMENT = "document"
    QUERY = "query"

class GroqEnums(Enum):
    SYSTEM = "system"
    USER = "user"

class AnthropicEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"