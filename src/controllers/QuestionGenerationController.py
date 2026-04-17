from .BaseController import BaseController
from models.ChunkModel import ChunkModel
from routes.schemas.question import (
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    QuestionResponse,
    QuestionConfig,
    TopicError,
    GradingRubric
)
from typing import List, Optional
import json
import logging


class QuestionGenerationController(BaseController):

    def __init__(self, generation_client):
        super().__init__()
        self.generation_client = generation_client
        self.logger = logging.getLogger(__name__)

    # =============================================================
    # PUBLIC ENTRY POINT
    # =============================================================

    async def generate_all(
        self,
        request: GenerateQuestionsRequest,
        chunk_model: ChunkModel
    ) -> GenerateQuestionsResponse:

        all_questions: List[QuestionResponse] = []
        errors: List[TopicError] = []

        # ── PARALLEL TOPIC PROCESSING ──────────────────────────────────────
        # All topics are processed simultaneously instead of one by one.
        # Each topic makes one LLM call — running them in parallel cuts
        # total time from (n_topics × llm_latency) to (~1 × llm_latency).
        async def process_topic(topic):
            try:
                topic_content = await self.get_topic_content(
                    chunk_model=chunk_model,
                    project_id=request.project_id,
                    topic_title=topic.topic_title
                )

                if not topic_content:
                    return None, TopicError(
                        topic_id=topic.topic_id,
                        topic_title=topic.topic_title,
                        reason="No content found for this topic in the document"
                    )

                questions = await self.generate_questions_for_topic(
                    topic_id=topic.topic_id,
                    topic_title=topic.topic_title,
                    topic_content=topic_content,
                    question_configs=topic.question_configs
                )

                if not questions:
                    return None, TopicError(
                        topic_id=topic.topic_id,
                        topic_title=topic.topic_title,
                        reason="LLM failed to generate questions for this topic"
                    )

                return questions, None

            except Exception as e:
                self.logger.error(f"Error processing topic '{topic.topic_title}': {e}")
                return None, TopicError(
                    topic_id=topic.topic_id,
                    topic_title=topic.topic_title,
                    reason=str(e)
                )

        # Run all topics in parallel
        import asyncio
        results = await asyncio.gather(
            *[process_topic(topic) for topic in request.topics]
        )

        # Collect results
        for questions, error in results:
            if error:
                errors.append(error)
            if questions:
                all_questions.extend(questions)
        # ───────────────────────────────────────────────────────────────────

        # 3. Determine status
        if len(errors) == 0:
            status = "completed"
        elif len(all_questions) == 0:
            status = "failed"
        else:
            status = "partial"

        return GenerateQuestionsResponse(
            request_id=request.request_id,
            course_id=request.course_id,
            project_id=request.project_id,
            status=status,
            errors=errors,
            questions=all_questions
        )

    # =============================================================
    # GET TOPIC CONTENT FROM MONGODB CHUNKS
    # =============================================================

    async def get_topic_content(
        self,
        chunk_model: ChunkModel,
        project_id: str,
        topic_title: str
    ) -> Optional[str]:

        # Fetch only question-type chunks for this project (up to 1000)
        chunks = await chunk_model.get_chunks_by_project_id(
            project_id=project_id,
            page_no=1,
            page_size=1000,
            chunk_type="question"
        )

        if not chunks:
            self.logger.warning(f"No chunks found for project {project_id}")
            return None

        # Sort by chunk_order
        chunks = sorted(chunks, key=lambda c: c.chunk_order)

        # Filter chunks relevant to this topic using keyword matching
        topic_keywords = [
            word.lower()
            for word in topic_title.split()
            if len(word) > 3  # skip short words like "and", "the", "of"
        ]

        relevant_chunks = []
        if topic_keywords:
            for chunk in chunks:
                chunk_lower = chunk.chunk_text.lower()
                if any(keyword in chunk_lower for keyword in topic_keywords):
                    relevant_chunks.append(chunk)

        # Fallback: use first 10 chunks if no relevant ones found
        if not relevant_chunks:
            self.logger.warning(
                f"No relevant chunks found for topic '{topic_title}', "
                f"falling back to first 10 chunks"
            )
            relevant_chunks = chunks[:10]

        # Join chunk texts
        full_text = "\n\n".join([c.chunk_text for c in relevant_chunks])

        # Truncate to max allowed characters for question generation
        max_chars = self.app_settings.QUESTION_CHUNK_SIZE * 10
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars]
            self.logger.info(f"Truncated topic content to {max_chars} characters")

        self.logger.info(
            f"Topic '{topic_title}': {len(relevant_chunks)} relevant chunks, "
            f"{len(full_text)} characters"
        )

        return full_text

    # =============================================================
    # GENERATE QUESTIONS FOR A SINGLE TOPIC
    # =============================================================

    async def generate_questions_for_topic(
        self,
        topic_id: int,
        topic_title: str,
        topic_content: str,
        question_configs: List[QuestionConfig]
    ) -> Optional[List[QuestionResponse]]:

        prompt = self._build_generation_prompt(
            topic_title=topic_title,
            topic_content=topic_content,
            question_configs=question_configs
        )

        try:
            response = self.generation_client.generate_text(
                prompt=prompt,
                chat_history=[],
                temperature=0.7,
                max_output_tokens=self.app_settings.GENERATION_DAFAULT_MAX_TOKENS
            )

            if not response:
                self.logger.error(f"Empty response from LLM for topic '{topic_title}'")
                return None

            raw_questions = self._parse_questions_response(response)

            if not raw_questions:
                return None

            # Map raw dicts to QuestionResponse objects
            questions = []
            for q in raw_questions:
                try:
                    grading_rubric = None
                    if q.get("grading_rubric"):
                        grading_rubric = GradingRubric(
                            type=q["grading_rubric"]["type"],
                            items=q["grading_rubric"]["items"]
                        )

                    questions.append(QuestionResponse(
                        topic_id=topic_id,
                        topic_title=topic_title,
                        type=q.get("type", ""),
                        difficulty=q.get("difficulty", ""),
                        question_text=q.get("question_text", ""),
                        explanation=q.get("explanation"),
                        options=q.get("options"),
                        expected_answer=q.get("expected_answer", ""),
                        grading_rubric=grading_rubric
                    ))
                except Exception as e:
                    self.logger.error(f"Error mapping question to schema: {e} | raw: {q}")
                    continue

            return questions if questions else None

        except Exception as e:
            self.logger.error(f"Error calling LLM for topic '{topic_title}': {e}")
            return None

    # =============================================================
    # BUILD PROMPT
    # =============================================================

    def _build_generation_prompt(
        self,
        topic_title: str,
        topic_content: str,
        question_configs: List[QuestionConfig]
    ) -> str:

        # Build the questions specification string
        questions_spec_lines = []
        for config in question_configs:
            questions_spec_lines.append(
                f"- {config.count} {config.type} question(s) at {config.difficulty} difficulty"
            )
        questions_spec = "\n".join(questions_spec_lines)

        prompt = f"""You are an educational question generator. Generate questions based on the provided content.

RULES:
1. Return ONLY a valid JSON array, no markdown, no explanation, no code blocks
2. Each question must follow this exact structure:
{{
  "type": "...",
  "difficulty": "...",
  "question_text": "...",
  "explanation": "...",
  "options": [...] or null,
  "expected_answer": "...",
  "grading_rubric": {{...}} or null
}}
3. For multiple_choice: options must be exactly 4 items:
   [{{"id":"A","text":"..."}},{{"id":"B","text":"..."}},{{"id":"C","text":"..."}},{{"id":"D","text":"..."}}]
   expected_answer must be the option id: "A", "B", "C", or "D"
   grading_rubric must be null

4. For true_false: options must be:
   [{{"id":"true","text":"True"}},{{"id":"false","text":"False"}}]
   expected_answer must be "true" or "false"
   grading_rubric must be null

5. For short_answer: options must be null
   grading_rubric must be:
   {{"type":"key_points","items":["point 1","point 2","point 3"]}}

6. For essay: options must be null
   grading_rubric must be:
   {{"type":"criteria","items":[{{"name":"...","description":"..."}},{{"name":"...","description":"..."}}]}}

7. Difficulty must exactly match the requested level: "easy", "medium", or "hard"
8. Base ALL questions strictly on the provided content — do not use outside knowledge
9. question_text must be a clear, complete question
10. explanation is required for all types — explain why the answer is correct

Topic: {topic_title}

Generate the following:
{questions_spec}

Content:
{topic_content}

JSON array output:"""

        return prompt

    # =============================================================
    # PARSE LLM RESPONSE
    # =============================================================

    def _parse_questions_response(self, response: str) -> Optional[list]:
        response = response.strip()

        self.logger.info(f"Raw LLM response (first 300 chars): {response[:300]}")

        # Remove markdown code blocks (reusing pattern from StructureController)
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            parts = response.split("```")
            if len(parts) >= 3:
                response = parts[1]
            else:
                response = response.replace("```", "")

        response = response.strip()

        # Extract JSON array between [ and ]
        start_idx = response.find("[")
        end_idx = response.rfind("]") + 1

        if start_idx == -1 or end_idx <= start_idx:
            self.logger.error("No JSON array found in LLM response")
            self.logger.error(f"Response was: {response[:500]}")
            return None

        response = response[start_idx:end_idx].strip()

        try:
            questions = json.loads(response)

            if not isinstance(questions, list):
                self.logger.error("Parsed JSON is not a list")
                return None

            self.logger.info(f"Successfully parsed {len(questions)} questions")
            return questions

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parse failed: {e}")
            self.logger.error(f"Attempted to parse: {response[:500]}")
            return None