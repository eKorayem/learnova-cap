from pydantic import BaseModel, Field
from typing import List, Optional, Any

# ==========================================
# INCOMING REQUEST SCHEMAS (From Backend)
# ==========================================

class TargetSection(BaseModel):
    topic_id: int
    topic_title: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None

class SummaryRequestBody(BaseModel):
    material_id: Optional[int] = None  
    target_sections: List[TargetSection]
    summary_type: str = "detailed" # e.g., "brief", "detailed", "executive"

class SummaryWebhookPayload(BaseModel):
    request_id: str
    timestamp: str
    operation_type: str
    course_id: int
    body: SummaryRequestBody

# ==========================================
# INTERNAL & OUTGOING SCHEMAS (AI -> Backend)
# ==========================================

class SummarySection(BaseModel):
    topic_title: str
    content: str = Field(description="A cohesive paragraph summarizing the section.")
    key_points: List[str] = Field(description="3 to 5 critical bullet points extracted from this section.")

class SummaryPayload(BaseModel):
    title: str = Field(description="The overarching cohesive title for the summary document.")
    executive_summary: str = Field(description="A high-level paragraph summarizing the entire compiled text.")
    sections: List[SummarySection] = Field(description="The detailed summaries broken down by the requested sections.")