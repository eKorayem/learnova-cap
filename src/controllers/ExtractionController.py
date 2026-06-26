import json
import logging
from typing import List, Optional
from stores.llm.LLMInterface import LLMInterface
from routes.schemas.extraction import TopicExtractionConfig

class ExtractionController:
    def __init__(self, generation_client: LLMInterface):
        self.generation_client = generation_client
        self.logger = logging.getLogger('uvicorn.error')

    async def extract_native_questions(self, document_text: str, topics: List[TopicExtractionConfig]) -> Optional[dict]:
        
        # Build a mapping of ID to Title for the LLM
        topics_str = "\n".join([f"- ID: {t.id} | Title: {t.title}" for t in topics])
        
        system_prompt = f"""You are an expert academic data extractor. Your ONLY job is to extract existing questions verbatim from the provided text.

CRITICAL RULES:
1. DO NOT invent questions. Extract them EXACTLY as they appear.
2. Categorize EVERY question into exactly ONE of the provided topic IDs based on context.
3. Classify the 'type' (multiple_choice, true_false, short_answer, essay).
4. Estimate 'difficulty' (easy, medium, hard).
5. For multiple_choice/true_false, format options exactly as: [{{"id":"A","text":"..."}}, {{"id":"B","text":"..."}}]
6. For multiple_choice/true_false, 'expected_answer' must be the letter "A", "B", "C", or "D".
7. For short_answer/essay, 'options' must be an empty array [], 'expected_answer' must be null, and you MUST generate a basic 'grading_rubric' based on the text.

AVAILABLE TOPICS:
{topics_str}

You must return ONLY valid JSON matching this schema exactly:
{{
  "extracted_questions": [
    {{
      "topic_id": 0,
      "question_text": "...",
      "type": "...",
      "difficulty": "...",
      "options": [{{"id": "A", "text": "..."}}],
      "expected_answer": "...",
      "explanation": null,
      "grading_rubric": {{"criteria": [{{"name": "...","description": "..."}}]}}
    }}
  ]
}}
"""

        user_prompt = f"DOCUMENT TEXT:\n\n{document_text}"

        try:
            # Enforce strict JSON output using the massive context model
            response = await self.generation_client.generate_structured_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_schema={} # OpenRouter/Groq handles the schema directly from the prompt instructions
            )

            if not response:
                return None

            return json.loads(response)

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM extraction output as JSON: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error during native question extraction: {e}")
            return None