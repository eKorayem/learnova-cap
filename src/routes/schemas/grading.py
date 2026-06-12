from pydantic import BaseModel
from typing import List, Optional, Any, Dict

# ==========================================
# INCOMING REQUEST SCHEMAS (From Backend)
# ==========================================

class ExamQuestion(BaseModel):
    exam_question_id: int
    question_text: str
    type: str
    expected_answer: str
    grading_rubric: Optional[Dict[str, Any]] = None
    max_score: float
    student_answer: str

class GradingRequestBody(BaseModel):
    attempt_id: int
    exam_id: int
    questions: List[ExamQuestion]

class GradingWebhookPayload(BaseModel):
    request_id: str
    timestamp: str
    operation_type: str
    course_id: int
    body: GradingRequestBody

# ==========================================
# OUTGOING RESPONSE SCHEMAS (To Backend)
# ==========================================

class GradedResult(BaseModel):
    exam_question_id: int
    points_earned: float
    feedback: str
    
class GradingAIInternalResult(BaseModel):
    """This is the schema we force the AI to return internally"""
    reasoning_process: str # The Chain of Thought (not sent to backend)
    points_earned: float
    feedback: str # What the student actually sees