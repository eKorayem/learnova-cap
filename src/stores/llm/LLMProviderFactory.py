from .LLMEnums import LLMEnums
from .providers import OpenAIProvider, CoHereProvider, GroqProvider, SentenceTransformerProvider, OpenRouterProvider, JinaProvider, DeepSeekProvider


class LLMProviderFactory:
    def __init__(self, config: dict):
        self.config = config

    def create(self, provider: str):

        if provider == LLMEnums.OPENAI.value:
            return OpenAIProvider(
                api_key=self.config.OPENAI_API_KEY,
                api_url=self.config.OPENAI_API_URL,
                defualt_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                defualt_generation_max_out_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE
            )

        elif provider == LLMEnums.COHERE.value:
            return CoHereProvider(
                api_key=self.config.COHERE_API_KEY,
                defualt_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                defualt_generation_max_out_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE
            )

        elif provider == LLMEnums.GROQ.value:
            return GroqProvider(
                api_key=self.config.GROQ_API_KEY,
                defualt_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                defualt_generation_max_out_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE
            )

        elif provider == LLMEnums.SENTENCE_TRANSFORMER.value:
            return SentenceTransformerProvider(
                defualt_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                defualt_generation_max_out_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE
            )

        elif provider == LLMEnums.OPENROUTER.value:
            return OpenRouterProvider(
                api_key=self.config.QUESTION_OPENAI_API_KEY,
                defualt_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                defualt_generation_max_out_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE
            )

        elif provider == LLMEnums.JINA.value:
            return JinaProvider(
                api_key=self.config.JINA_API_KEY,
                defualt_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
            )
        

        elif provider == LLMEnums.DEEPSEEK.value:
            return DeepSeekProvider(
                api_key=self.config.DEEPSEEK_API_KEY,
                defualt_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                defualt_generation_max_out_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE
            )

        elif provider == LLMEnums.ANTHROPIC.value:
            from .providers.AnthropicProvider import AnthropicProvider
            return AnthropicProvider(
                aws_access_key=self.config.AWS_ACCESS_KEY_ID,
                aws_secret_key=self.config.AWS_SECRET_ACCESS_KEY,
                aws_region=self.config.AWS_REGION_NAME,
                defualt_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                defualt_generation_max_out_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")