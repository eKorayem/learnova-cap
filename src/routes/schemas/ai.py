from pydantic import BaseModel
from typing import List, Optional


class AnalyzeStructureRequest(BaseModel):
    project_id: str
    lecture_id: Optional[str] = None  # ← Make it optional
    max_topics: Optional[int] = None  # Limit number of topics


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