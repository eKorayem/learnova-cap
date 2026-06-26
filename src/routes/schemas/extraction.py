from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

# ==========================================
# OUTGOING SCHEMAS (AI -> Backend)
# ==========================================
class ExtractedOption(BaseModel):
    id: str
    text: str

class ExtractedQuestion(BaseModel):
    topic_id: int
    question_text: str
    type: str
    difficulty: str
    options: Optional[List[ExtractedOption]] = []
    expected_answer: Optional[str] = None
    explanation: Optional[str] = None
    grading_rubric: Optional[Dict[str, Any]] = None

# ==========================================
# INCOMING SCHEMAS (Backend -> AI)
# ==========================================
class TopicExtractionConfig(BaseModel):
    id: int
    title: str

class ExtractionRequestBody(BaseModel):
    module_id: int
    material_id: int
    topics: List[TopicExtractionConfig]

class ExtractionWebhookPayload(BaseModel):
    request_id: str
    timestamp: str
    operation_type: str
    course_id: int
    body: ExtractionRequestBody