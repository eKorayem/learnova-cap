from .BaseController import BaseController
from models.ChunkModel import ChunkModel
from routes.schemas.question import (
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    QuestionResponse,
    QuestionConfig,
    TopicError
)
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
import json
import logging
import asyncio


# =============================================================
# STRICT INTERNAL LLM OUTPUT SCHEMAS
# =============================================================

class GeneratedQuestionOption(BaseModel):
    id: str
    text: str

    class Config:
        extra = "forbid"


class GeneratedQuestion(BaseModel):
    """
    Internal schema intentionally mirrors QuestionResponse fields exactly,
    while using a stricter option structure for JSON schema guidance.
    """
    topic_id: int
    topic_title: str
    type: str
    difficulty: str
    question_text: str
    explanation: Optional[str] = None
    options: Optional[List[GeneratedQuestionOption]] = None
    expected_answer: str
    grading_rubric: Optional[Dict[str, Any]] = None

    class Config:
        extra = "forbid"


class GeneratedQuestionsPayload(BaseModel):
    """
    JSON object wrapper required for native JSON object mode.
    Do not ask the model to return a bare array.
    """
    questions: List[GeneratedQuestion] = Field(default_factory=list)

    class Config:
        extra = "forbid"


class QuestionGenerationController(BaseController):

    VALID_TYPES = {"multiple_choice", "true_false", "short_answer", "essay"}
    VALID_DIFFICULTIES = {"easy", "medium", "hard"}
    MCQ_OPTION_IDS = ["A", "B", "C", "D"]

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
                        reason="LLM failed to generate valid structured questions for this topic"
                    )

                return questions, None

            except Exception as e:
                self.logger.error(f"Error processing topic '{topic.topic_title}': {e}")
                return None, TopicError(
                    topic_id=topic.topic_id,
                    topic_title=topic.topic_title,
                    reason=str(e)
                )

        results = await asyncio.gather(
            *[process_topic(topic) for topic in request.topics]
        )

        for questions, error in results:
            if error:
                errors.append(error)
            if questions:
                all_questions.extend(questions)

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

        chunks = await chunk_model.get_chunks_by_project_id(
            project_id=project_id,
            page_no=1,
            page_size=1000,
            chunk_type="question"
        )

        if not chunks:
            self.logger.warning(f"No question chunks found for project {project_id}")
            return None

        chunks = sorted(chunks, key=lambda c: c.chunk_order)

        topic_keywords = [
            word.lower()
            for word in topic_title.split()
            if len(word) > 3
        ]

        relevant_chunks = []
        if topic_keywords:
            scored_chunks = []
            for chunk in chunks:
                chunk_lower = chunk.chunk_text.lower()
                score = sum(1 for keyword in topic_keywords if keyword in chunk_lower)
                if score > 0:
                    scored_chunks.append((score, chunk))

            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            relevant_chunks = [item[1] for item in scored_chunks[:20]]

        if not relevant_chunks:
            self.logger.warning(
                f"No relevant chunks found for topic '{topic_title}', "
                f"falling back to first 10 chunks"
            )
            relevant_chunks = chunks[:10]
        else:
            relevant_chunks = sorted(relevant_chunks, key=lambda c: c.chunk_order)

        full_text = "\n\n".join([c.chunk_text for c in relevant_chunks])

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

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            topic_id=topic_id,
            topic_title=topic_title,
            topic_content=topic_content,
            question_configs=question_configs
        )

        try:
            response_schema = self._pydantic_schema(GeneratedQuestionsPayload)

            llm_response = await self.generation_client.generate_structured_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_schema=response_schema,
                temperature=self.app_settings.QUESTION_TEMPERATURE,
            )

            if not llm_response:
                self.logger.error(f"Empty structured response from LLM for topic '{topic_title}'")
                return None

            parsed_payload = self._load_structured_payload(llm_response)
            if not parsed_payload:
                self.logger.error(f"Invalid JSON object from LLM for topic '{topic_title}'")
                return None

            sanitized_questions = []
            for raw_question in parsed_payload.questions:
                sanitized = self._sanitize_question(
                    raw_question=self._model_dump(raw_question),
                    topic_id=topic_id,
                    topic_title=topic_title
                )
                if sanitized:
                    sanitized_questions.append(sanitized)

            sanitized_questions = self._enforce_requested_counts(
                questions=sanitized_questions,
                question_configs=question_configs
            )

            if not sanitized_questions:
                self.logger.error(f"No valid questions survived sanitization for topic '{topic_title}'")
                return None

            self.logger.info(
                f"Generated {len(sanitized_questions)} valid questions "
                f"for topic '{topic_title}'"
            )
            return sanitized_questions

        except Exception as e:
            self.logger.error(f"Error generating structured questions for topic '{topic_title}': {e}")
            return None

    # =============================================================
    # PROMPTS
    # =============================================================

    def _build_system_prompt(self) -> str:
        return """You are an expert university assessment designer.

You must generate assessment questions using ONLY the provided course content.
You must return ONLY a valid JSON object that conforms to the supplied schema.
Do not return markdown, code fences, comments, prose, or a bare JSON array.

ABSOLUTE OUTPUT CONTRACT:
- The root must be an object with exactly one key: "questions".
- "questions" must be an array.
- Every question object must contain these fields:
  topic_id, topic_title, type, difficulty, question_text, explanation, options, expected_answer, grading_rubric.
- Do not invent extra fields.
- Do not use null for expected_answer. It must always be a non-empty string.
- For objective questions, options must be an array and grading_rubric must be null.
- For subjective questions, options must be null and grading_rubric must be an object.

QUESTION TYPE RULES:
1. multiple_choice:
   - options must contain exactly four objects with IDs "A", "B", "C", "D".
   - expected_answer must be exactly one of "A", "B", "C", "D".
   - grading_rubric must be null.
   - explanation must explain why the correct answer is correct and why common confusions are wrong.

2. true_false:
   - options must be exactly:
     [{"id":"true","text":"True"},{"id":"false","text":"False"}]
   - expected_answer must be exactly "true" or "false".
   - grading_rubric must be null.
   - explanation must justify the truth value using the content.

3. short_answer:
   - options must be null.
   - expected_answer must be a concise sample correct answer.
   - grading_rubric must be exactly shaped as:
     {"key_points":["atomic required concept 1","atomic required concept 2","atomic required concept 3"]}
   - key_points must be technical, concept-driven, and independently gradable.
   - Do not include vague key points such as "explains clearly" or "mentions the topic".

4. essay:
   - options must be null.
   - expected_answer must be a rigorous model answer paragraph.
   - grading_rubric must be exactly shaped as:
     {"criteria":[{"name":"criterion name","description":"specific technical expectations"}]}
   - Criteria must evaluate conceptual correctness, causal reasoning, evidence, distinctions, and limitations where applicable.

DISTRACTOR ENGINEERING REQUIREMENTS FOR MULTIPLE CHOICE:
- Distractors must be highly plausible to a student who partially understands the material.
- Distractors must be non-obvious and content-related.
- Avoid absurd, humorous, or trivially false choices.
- Avoid choices that differ only by wording.
- Avoid "all of the above", "none of the above", and negative trick wording unless explicitly requested by the content.
- Each distractor should reflect a realistic misconception, overgeneralization, reversed causal relationship, or confused neighboring concept.
- Exactly one option must be correct.

FEW-SHOT QUALITY STANDARD:

Bad multiple-choice distractors:
Question: What is encapsulation?
A. Bundling data and methods together.
B. A type of fruit.
C. The capital of France.
D. A random number.

Good multiple-choice distractors:
Question: What is encapsulation?
A. Bundling data with the methods that operate on it while controlling access.
B. Splitting a program into unrelated files to reduce file size.
C. Making all internal fields public so other classes can reuse them directly.
D. Replacing inheritance with repeated code in each class.

Why the good version is better:
- B, C, and D are plausible programming misconceptions.
- Only A captures the technical concept precisely.

SHORT ANSWER RUBRIC STANDARD:
Bad key_points:
["Good explanation", "Mentions examples", "Clear writing"]

Good key_points:
["Defines the concept using the core technical property", "Explains the mechanism or process that produces the result", "Identifies a limitation, exception, or common misconception"]

ESSAY RUBRIC STANDARD:
Bad criterion:
{"name":"Quality","description":"Answer is good and detailed"}

Good criterion:
{"name":"Conceptual accuracy","description":"Accurately explains the central concepts and distinguishes them from related but incorrect alternatives."}

STRICTNESS & ESCAPE HATCH:
- Base every question EXCLUSIVELY on the provided topic content.
- If the content does not support a requested question type, generate the closest valid question from supported content.
- CRITICAL QUOTA RULE: If the provided content is too short or lacks enough detail to generate the requested number of unique, high-quality questions without repeating concepts or hallucinating outside knowledge, DO NOT force it. 
- Generate ONLY as many valid questions as the text naturally supports (even if that means returning 3 questions when 10 were requested). Quality and strict reliance on the text are vastly more important than hitting the exact quota.
- Difficulty must exactly match the requested difficulty: "easy", "medium", or "hard".
"""

    def _build_user_prompt(
        self,
        topic_id: int,
        topic_title: str,
        topic_content: str,
        question_configs: List[QuestionConfig]
    ) -> str:

        questions_spec_lines = []
        total_requested = 0

        for config in question_configs:
            q_type = self._normalize_type(config.type)
            difficulty = self._normalize_difficulty(config.difficulty)
            count = max(int(config.count), 0)
            total_requested += count
            questions_spec_lines.append(
                f"- Generate exactly {count} question(s): type={q_type}, difficulty={difficulty}"
            )

        questions_spec = "\n".join(questions_spec_lines)

        return f"""TOPIC METADATA:
topic_id: {topic_id}
topic_title: {topic_title}

REQUESTED QUESTION MIX:
{questions_spec}

TOTAL QUESTIONS REQUIRED:
{total_requested}

CONTENT TO USE:
{topic_content}

Return a JSON object with exactly this root shape:
{{
  "questions": [
    {{
      "topic_id": {topic_id},
      "topic_title": "{topic_title}",
      "type": "multiple_choice",
      "difficulty": "medium",
      "question_text": "A complete question grounded in the content.",
      "explanation": "Why the answer is correct.",
      "options": [
        {{"id":"A","text":"Option A"}},
        {{"id":"B","text":"Option B"}},
        {{"id":"C","text":"Option C"}},
        {{"id":"D","text":"Option D"}}
      ],
      "expected_answer": "A",
      "grading_rubric": null
    }}
  ]
}}
"""

    # =============================================================
    # STRUCTURED RESPONSE PARSING
    # =============================================================

    def _load_structured_payload(self, response: str) -> Optional[GeneratedQuestionsPayload]:
        """
        Native JSON object mode should already return a clean object.
        No regex extraction, no markdown stripping, no bracket chopping.
        """
        try:
            data = json.loads(response)
            if not isinstance(data, dict):
                self.logger.error("Structured LLM response root was not a JSON object")
                return None
            return GeneratedQuestionsPayload(**data)
        except json.JSONDecodeError as e:
            self.logger.error(f"Structured LLM response was not valid JSON: {e}")
            self.logger.error(f"Raw response preview: {response[:500]}")
            return None
        except Exception as e:
            self.logger.error(f"Structured LLM response failed schema validation: {e}")
            self.logger.error(f"Raw response preview: {response[:500]}")
            return None

    # =============================================================
    # SANITIZATION AND CALLBACK 422 PROTECTION
    # =============================================================

    def _sanitize_question(
        self,
        raw_question: Dict[str, Any],
        topic_id: int,
        topic_title: str
    ) -> Optional[QuestionResponse]:

        q_type = self._normalize_type(raw_question.get("type"))
        difficulty = self._normalize_difficulty(raw_question.get("difficulty"))

        if q_type not in self.VALID_TYPES:
            self.logger.warning(f"Dropping question with invalid type: {raw_question.get('type')}")
            return None

        question_text = str(raw_question.get("question_text") or "").strip()
        if not question_text:
            self.logger.warning("Dropping question with empty question_text")
            return None

        explanation = raw_question.get("explanation")
        explanation = str(explanation).strip() if explanation is not None and str(explanation).strip() else None

        expected_answer = str(raw_question.get("expected_answer") or "").strip()
        grading_rubric = raw_question.get("grading_rubric")
        options = raw_question.get("options")

        if q_type == "multiple_choice":
            sanitized_options = self._sanitize_mcq_options(options)
            if not sanitized_options:
                self.logger.warning(f"Dropping malformed MCQ: {question_text[:120]}")
                return None

            expected_answer = expected_answer.upper()
            if expected_answer not in self.MCQ_OPTION_IDS:
                expected_answer = self._infer_option_id_from_text(expected_answer, sanitized_options)

            if expected_answer not in self.MCQ_OPTION_IDS:
                self.logger.warning(f"Dropping MCQ with invalid expected_answer: {question_text[:120]}")
                return None

            return QuestionResponse(
                topic_id=topic_id,
                topic_title=topic_title,
                type=q_type,
                difficulty=difficulty,
                question_text=question_text,
                explanation=explanation or "The correct option follows directly from the provided course content.",
                options=sanitized_options,
                expected_answer=expected_answer,
                grading_rubric=None
            )

        if q_type == "true_false":
            expected_answer = expected_answer.lower()
            if expected_answer in {"t", "true", "yes"}:
                expected_answer = "true"
            elif expected_answer in {"f", "false", "no"}:
                expected_answer = "false"
            else:
                self.logger.warning(f"Dropping true_false question with invalid expected_answer: {question_text[:120]}")
                return None

            return QuestionResponse(
                topic_id=topic_id,
                topic_title=topic_title,
                type=q_type,
                difficulty=difficulty,
                question_text=question_text,
                explanation=explanation or "The truth value follows from the provided course content.",
                options=[
                    {"id": "true", "text": "True"},
                    {"id": "false", "text": "False"}
                ],
                expected_answer=expected_answer,
                grading_rubric=None
            )

        if q_type == "short_answer":
            if not expected_answer:
                expected_answer = "A correct response should address all required key points in the grading rubric."

            sanitized_rubric = self._sanitize_short_answer_rubric(grading_rubric, expected_answer)

            return QuestionResponse(
                topic_id=topic_id,
                topic_title=topic_title,
                type=q_type,
                difficulty=difficulty,
                question_text=question_text,
                explanation=explanation or "A complete answer must include the core concepts identified in the rubric.",
                options=None,
                expected_answer=expected_answer,
                grading_rubric=sanitized_rubric
            )

        if q_type == "essay":
            if not expected_answer:
                expected_answer = "A correct essay should develop a rigorous, concept-driven response aligned with the grading criteria."

            sanitized_rubric = self._sanitize_essay_rubric(grading_rubric, expected_answer)

            return QuestionResponse(
                topic_id=topic_id,
                topic_title=topic_title,
                type=q_type,
                difficulty=difficulty,
                question_text=question_text,
                explanation=explanation or "A strong essay must satisfy the conceptual criteria in the rubric.",
                options=None,
                expected_answer=expected_answer,
                grading_rubric=sanitized_rubric
            )

        return None

    def _sanitize_mcq_options(self, options: Any) -> Optional[List[Dict[str, str]]]:
        if not isinstance(options, list) or len(options) != 4:
            return None

        normalized = []
        seen_ids = set()

        for idx, opt in enumerate(options):
            if not isinstance(opt, dict):
                return None

            opt_id = str(opt.get("id") or "").strip().upper()
            opt_text = str(opt.get("text") or "").strip()

            if not opt_id:
                opt_id = self.MCQ_OPTION_IDS[idx]

            if opt_id not in self.MCQ_OPTION_IDS:
                return None

            if opt_id in seen_ids:
                return None

            if not opt_text:
                return None

            seen_ids.add(opt_id)
            normalized.append({"id": opt_id, "text": opt_text})

        normalized.sort(key=lambda item: self.MCQ_OPTION_IDS.index(item["id"]))

        if [item["id"] for item in normalized] != self.MCQ_OPTION_IDS:
            return None

        return normalized

    def _infer_option_id_from_text(self, expected_answer: str, options: List[Dict[str, str]]) -> str:
        expected_normalized = str(expected_answer or "").strip().lower()
        for option in options:
            if option["text"].strip().lower() == expected_normalized:
                return option["id"]
        return expected_answer

    def _sanitize_short_answer_rubric(self, rubric: Any, expected_answer: str) -> Dict[str, List[str]]:
        key_points = []

        if isinstance(rubric, dict) and isinstance(rubric.get("key_points"), list):
            key_points = [
                str(point).strip()
                for point in rubric.get("key_points", [])
                if str(point).strip()
            ]

        if not key_points:
            key_points = [
                "Defines the central concept accurately.",
                "Explains the relevant mechanism, relationship, or process.",
                "Uses course-specific terminology or distinctions correctly."
            ]

        return {"key_points": key_points[:6]}

    def _sanitize_essay_rubric(self, rubric: Any, expected_answer: str) -> Dict[str, List[Dict[str, str]]]:
        criteria = []

        if isinstance(rubric, dict) and isinstance(rubric.get("criteria"), list):
            for criterion in rubric.get("criteria", []):
                if not isinstance(criterion, dict):
                    continue

                name = str(criterion.get("name") or "").strip()
                description = str(criterion.get("description") or "").strip()

                if name and description:
                    criteria.append({
                        "name": name,
                        "description": description
                    })

        if not criteria:
            criteria = [
                {
                    "name": "Conceptual accuracy",
                    "description": "Accurately explains the central concepts and avoids related misconceptions."
                },
                {
                    "name": "Reasoning and relationships",
                    "description": "Explains mechanisms, causal relationships, comparisons, or implications using course content."
                },
                {
                    "name": "Completeness",
                    "description": "Addresses all major parts of the prompt with specific, relevant details."
                }
            ]

        return {"criteria": criteria[:6]}

    def _enforce_requested_counts(
        self,
        questions: List[QuestionResponse],
        question_configs: List[QuestionConfig]
    ) -> List[QuestionResponse]:

        selected: List[QuestionResponse] = []
        used_indexes = set()

        for config in question_configs:
            target_type = self._normalize_type(config.type)
            target_difficulty = self._normalize_difficulty(config.difficulty)
            target_count = max(int(config.count), 0)

            matched = []
            for idx, question in enumerate(questions):
                if idx in used_indexes:
                    continue

                if question.type == target_type and question.difficulty == target_difficulty:
                    matched.append((idx, question))

                if len(matched) >= target_count:
                    break

            for idx, question in matched:
                used_indexes.add(idx)
                selected.append(question)

        return selected

    # =============================================================
    # NORMALIZATION HELPERS
    # =============================================================

    def _normalize_type(self, value: Any) -> str:
        normalized = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")

        if normalized in {"mcq", "multiplechoice", "multiple_choice_question"}:
            return "multiple_choice"
        if "multiple" in normalized:
            return "multiple_choice"
        if normalized in {"tf", "truefalse", "true_false_question"}:
            return "true_false"
        if "true" in normalized and "false" in normalized:
            return "true_false"
        if "essay" in normalized:
            return "essay"
        if "short" in normalized:
            return "short_answer"

        return normalized

    def _normalize_difficulty(self, value: Any) -> str:
        normalized = str(value or "medium").strip().lower()
        return normalized if normalized in self.VALID_DIFFICULTIES else "medium"

    def _pydantic_schema(self, model_cls) -> Dict[str, Any]:
        if hasattr(model_cls, "model_json_schema"):
            return model_cls.model_json_schema()
        return model_cls.schema()

    def _model_dump(self, model: BaseModel) -> Dict[str, Any]:
        if hasattr(model, "model_dump"):
            return model.model_dump()
        return model.dict()