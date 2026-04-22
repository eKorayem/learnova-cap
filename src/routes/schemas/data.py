from pydantic import BaseModel
from typing import Optional


class ProcessRequest(BaseModel):
    file_id: str = None
    chunk_size: Optional[int] = None
    overlap_size: Optional[int] = None
    do_reset: Optional[int] = 0

# --- NEW: Webhook Ingestion Schemas ---
class MaterialMetadata(BaseModel):
    title: str
    type: str
    file_name: str
    mime_type: str
    signed_download_url: str

class ExtractionConfig(BaseModel):
    extract_topics: bool
    extract_learning_outcomes: bool
    allow_subtopics: bool

class DocumentIngestionBody(BaseModel):
    module_id: int
    material_id: int
    material: MaterialMetadata
    extraction_config: ExtractionConfig

class DocumentWebhookPayload(BaseModel):
    request_id: str
    timestamp: str
    operation_type: str
    course_id: int
    body: DocumentIngestionBody