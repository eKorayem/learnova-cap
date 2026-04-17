from pydantic import BaseModel
from typing import List, Optional


# =============================================================
# SCHEMAS FOR STRUCTURE ANALYSIS (NORMALIZED FORMAT)
# =============================================================
# These schemas define the request/response format for the
# structure analysis endpoint. The response is normalized to a
# flat list of topics suitable for backend consumption.
# =============================================================


class AnalyzeMaterialStructureRequest(BaseModel):
    """
    Request schema for structure analysis endpoint.

    Note: In this flow, project_id (path parameter) corresponds
    logically to material_id from the backend system.
    """
    request_id: str
    course_id: int
    module_id: int
    material_id: int
    lecture_id: Optional[str] = None
    max_topics: Optional[int] = None
    use_all_chunks: Optional[bool] = False


class NormalizedTopicResponse(BaseModel):
    """
    A single topic in the normalized flat structure.
    Each topic or subtitle becomes one row in the final list.
    """
    temp_id: str
    title: str
    description: str
    order_index: int
    parent_temp_id: Optional[str] = None


class AnalyzeMaterialStructureResponse(BaseModel):
    """
    Response schema for structure analysis endpoint.
    Returns a flat list of topics with parent-child relationships
    expressed via temp_id references.
    """
    request_id: str
    course_id: int
    module_id: int
    material_id: int
    status: str  # "completed" or "failed"
    topics: List[NormalizedTopicResponse]


# =============================================================
# LEGACY SCHEMAS (kept for backward compatibility if needed)
# =============================================================

class SubtitleResponse(BaseModel):
    title: str
    order: int


class TopicResponse(BaseModel):
    title: str
    order: int
    subtitles: List[SubtitleResponse]


class AnalyzeStructureResponse(BaseModel):
    project_id: str
    lecture_id: str
    topics: List[TopicResponse]