from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

# ==========================================
# INCOMING REQUEST SCHEMAS (From Backend)
# ==========================================

class TargetSection(BaseModel):
    topic_id: int
    topic_title: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None

class PresentationRequestBody(BaseModel):
    material_id: Optional[int] = None  
    target_sections: List[TargetSection]
    slide_count: int

class PresentationWebhookPayload(BaseModel):
    request_id: str
    timestamp: str
    operation_type: str
    course_id: int
    body: PresentationRequestBody


# ==========================================
# INTERNAL & OUTGOING SCHEMAS (AI -> Backend)
# ==========================================

class Visual(BaseModel):
    src: str = "assets/placeholder.webp"
    fit: str = "cover"
    caption: Optional[str] = None
    position: Optional[str] = None
    alt: Optional[str] = None

class PresentationCard(BaseModel):
    heading: str
    body: str
    icon: Optional[str] = None

class ComparisonSide(BaseModel):
    label: str
    title: str
    subtitle: str
    points: List[str]

class Comparison(BaseModel):
    left: ComparisonSide
    right: ComparisonSide

class ProcessStep(BaseModel):
    title: str
    body: str

class TimelineEvent(BaseModel):
    date: str
    title: str
    body: str

class DiagramNode(BaseModel):
    id: str
    level: int
    title: str
    body: str

class DiagramEdge(BaseModel):
    from_node: str = Field(alias="from") # 'from' is a Python reserved keyword
    to: str

class Diagram(BaseModel):
    nodes: List[DiagramNode]
    edges: List[DiagramEdge]

class Table(BaseModel):
    headers: List[str]
    rows: List[List[str]]

class EquationStep(BaseModel):
    label: str
    latex: str
    render_mode: str = "svg"
    explanation: str

class Equation(BaseModel):
    label: str
    latex: str
    render_mode: str = "svg"
    explanation: Optional[str] = None
    steps: Optional[List[EquationStep]] = None

class SummaryObject(BaseModel):
    points: List[str]
    takeaway: str
    next: Optional[str] = None

class SlideContent(BaseModel):
    lead: Optional[str] = None
    body: Optional[str] = None
    bullets: Optional[List[str]] = None
    key_term: Optional[str] = None
    definition: Optional[str] = None
    example: Optional[str] = None

class GeneratedSlide(BaseModel):
    slide_number: int
    layout_type: str
    kicker: str
    title: str
    subtitle: Optional[str] = None
    
    # Metadata fields
    course: Optional[str] = None
    instructor: Optional[str] = None
    term: Optional[str] = None
    section_number: Optional[str] = None
    descriptor: Optional[str] = None
    overview: Optional[str] = None
    outcome: Optional[str] = None
    
    # Top-Level Data Objects
    content: Optional[SlideContent] = None
    visual: Optional[Visual] = None
    objectives: Optional[List[str]] = None
    cards: Optional[List[PresentationCard]] = None
    comparison: Optional[Comparison] = None
    process: Optional[List[ProcessStep]] = None
    timeline: Optional[List[TimelineEvent]] = None
    diagram: Optional[Diagram] = None
    table: Optional[Table] = None
    equation: Optional[Equation] = None
    summary: Optional[SummaryObject] = None
    references: Optional[List[str]] = None

    class Config:
        populate_by_name = True

class PresentationPayload(BaseModel):
    title: str = Field(description="The overarching cohesive title for the entire slide deck presentation.")
    slides: List[GeneratedSlide]