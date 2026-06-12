from pydantic import BaseModel
from typing import List, Optional, Any, Dict

# =============================================================
# REQUEST SCHEMAS (Backend → AI System)
# =============================================================

class QuestionConfig(BaseModel):
    type: str           # "multiple_choice", "true_false", "short_answer", "essay"
    difficulty: str     # "easy", "medium", "hard"
    count: int


class TopicQuestionRequest(BaseModel):
    topic_id: int
    topic_title: str
    question_configs: List[QuestionConfig]


class GenerateQuestionsRequest(BaseModel):
    request_id: str
    course_id: int
    project_id: str
    topics: List[TopicQuestionRequest]


# =============================================================
# RESPONSE SCHEMAS (AI System → Backend)
# =============================================================


class QuestionResponse(BaseModel):
    topic_id: int
    topic_title: str
    type: str
    difficulty: str
    question_text: str
    explanation: Optional[str] = None
    options: Optional[List[dict]] = None   # [{"id": "A", "text": "..."}]
    expected_answer: str
    # Changed from GradingRubric to Dict to accept the backend's dynamic shapes
    grading_rubric: Optional[Dict[str, Any]] = None

class TopicError(BaseModel):
    topic_id: Optional[int] = None
    topic_title: Optional[str] = None
    reason: str


class GenerateQuestionsResponse(BaseModel):
    request_id: str
    course_id: int
    project_id: str
    status: str                         # "completed", "partial", "failed"
    errors: List[TopicError] = []
    questions: List[QuestionResponse] = []


class QuestionGenerationBody(BaseModel):
    # Removed material_id completely!
    topics: List[TopicQuestionRequest]

class QuestionWebhookPayload(BaseModel):
    request_id: str
    timestamp: str
    operation_type: str
    course_id: int
    body: QuestionGenerationBody

