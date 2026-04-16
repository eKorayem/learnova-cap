from pydantic import BaseModel
from typing import List, Optional


class AnalyzeStructureRequest(BaseModel):
    project_id: str
    lecture_id: Optional[str] = None
    max_topics: Optional[int] = None
    use_all_chunks: Optional[bool] = False  # Use all chunks for better accuracy (slower)


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