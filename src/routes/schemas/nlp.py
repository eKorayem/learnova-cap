from pydantic import BaseModel
from typing import List, Optional

class PushRequest(BaseModel):
    do_reset: Optional[int] = 0

class SearchRequest(BaseModel):
    text: str
    limit: Optional[int] = 5

# ==========================================
# RAG CHAT WEBHOOK SCHEMAS
# ==========================================

class ChatHistoryItem(BaseModel):
    role: str
    content: str

class RagChatBody(BaseModel):
    session_id: int
    message_id: int
    user_role: str
    message: str
    history: List[ChatHistoryItem] = []

class RagChatWebhookPayload(BaseModel):
    request_id: str
    timestamp: str
    operation_type: str
    course_id: int
    body: RagChatBody