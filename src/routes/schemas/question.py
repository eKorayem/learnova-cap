from pydantic import BaseModel
from typing import List, Optional, Any


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

class GradingRubric(BaseModel):
    type: str           # "key_points" or "criteria"
    items: List[Any]
    # for key_points: items is List[str]
    # for criteria:   items is List[dict] with "name" and "description" keys


class QuestionResponse(BaseModel):
    topic_id: int
    topic_title: str
    type: str
    difficulty: str
    question_text: str
    explanation: Optional[str] = None
    options: Optional[List[dict]] = None   # [{"id": "A", "text": "..."}]
    expected_answer: str
    grading_rubric: Optional[GradingRubric] = None


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