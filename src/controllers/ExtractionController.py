import json
import logging
from typing import List, Optional
from stores.llm.LLMInterface import LLMInterface
from routes.schemas.extraction import TopicExtractionConfig
from helpers.config import get_settings

class ExtractionController:
    def __init__(self, generation_client: LLMInterface):
        self.generation_client = generation_client
        self.logger = logging.getLogger('uvicorn.error')

    async def extract_native_questions(self, document_text: str, topics: List[TopicExtractionConfig]) -> Optional[dict]:
        
        # Build a mapping of ID to Title for the LLM
        topics_str = "\n".join([f"- ID: {t.id} | Title: {t.title}" for t in topics])
        
        system_prompt = f"""You are an expert academic data extractor. Your ONLY job is to extract actual ACADEMIC ASSESSMENT QUESTIONS (quizzes, exercises, review questions, exam prep) verbatim from the provided text.

CRITICAL RULES:
1. DO NOT invent questions. Extract them EXACTLY as they appear.
2. IGNORE RHETORICAL & EXAMPLE QUESTIONS: You MUST NOT extract conversational filler (e.g., "Are you ready?"), rhetorical questions, or examples of questions used in the text (e.g., examples of password security questions like "What is your mother's maiden name?"). 
3. ONLY extract questions that are clearly intended as a formal test of the student's knowledge. If the text contains no formal assessment questions, return an empty array [].
4. Categorize EVERY valid question into exactly ONE of the provided topic IDs based on context.
5. Classify the 'type' (multiple_choice, true_false, short_answer, essay).
6. For multiple_choice: 
   - You MUST include an "options" array with exactly 4 items: [{{"id":"A","text":"..."}},{{"id":"B","text":"..."}},{{"id":"C","text":"..."}},{{"id":"D","text":"..."}}]
   - expected_answer must be "A", "B", "C", or "D"
   - grading_rubric must be null

7. For true_false: 
   - You MUST include an "options" array with exactly 2 items: [{{"id":"true","text":"True"}},{{"id":"false","text":"False"}}]
   - expected_answer must be "true" or "false"
   - grading_rubric must be null

8. For short_answer: 
   - Do NOT include an "options" array (or set it to null)
   - You MUST include a "grading_rubric" object exactly like this:
     {{"key_points":["point 1","point 2","point 3"]}}
   - "expected_answer" MUST contain a sample correct answer string. NEVER use null.

9. For essay: 
   - Do NOT include an "options" array (or set it to null)
   - You MUST include a "grading_rubric" object exactly like this:
     {{"criteria":[{{"name":"...","description":"..."}},{{"name":"...","description":"..."}}]}}
   - "expected_answer" MUST contain a sample correct answer string. NEVER use null.

10. You MUST assess the difficulty of the extracted question and set "difficulty" to exactly "easy", "medium", or "hard". NEVER use null.

AVAILABLE TOPICS:
{topics_str}

You must return ONLY valid JSON matching this schema exactly:
{{
  "extracted_questions": [
    {{
      "topic_id": 0,
      "question_text": "...",
      "type": "...",
      "difficulty": "medium",
      "options": [{{"id": "A", "text": "..."}}],
      "expected_answer": "Sample correct answer string goes here",
      "explanation": null,
      "grading_rubric": {{"key_points": ["..."]}}
    }}
  ]
}}
"""

        user_prompt = f"DOCUMENT TEXT:\n\n{document_text}"

        # Define the EXACT schema to prevent the '{}' hallucination bug
        expected_schema = {
            "extracted_questions": [
                {
                    "topic_id": 0,
                    "question_text": "string",
                    "type": "string",
                    "difficulty": "string",
                    "options": [{"id": "string", "text": "string"}],
                    "expected_answer": "string",
                    "explanation": "string",
                    "grading_rubric": {"key_points": ["string"]}
                }
            ]
        }

        try:
            settings = get_settings()
            
            response = await self.generation_client.generate_structured_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_schema=expected_schema,
                temperature=settings.EXTRACTION_TEMPERATURE
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