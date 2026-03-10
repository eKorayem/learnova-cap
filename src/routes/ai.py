import logging
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from routes.schemas.ai import (
    AnalyzeStructureRequest, 
    AnalyzeStructureResponse,
    TopicResponse,
    SubtitleResponse
)
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from controllers import StructureController
from models.enums.ResponseEnums import ResponseSignal

logger = logging.getLogger('uvicorn.error')

ai_router = APIRouter(
    prefix="/api/v1/ai",
    tags=["api_v1", "ai"]
)


@ai_router.post("/analyze-structure")
async def analyze_structure(
    request: Request,
    analyze_request: AnalyzeStructureRequest
):
    """
    Analyze PDF structure and extract topics/subtitles
    """
    
    # Get models
    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )
    
    chunk_model = await ChunkModel.create_instance(
        db_client=request.app.db_client
    )
    
    # Check project exists
    project = await project_model.get_project_or_create_one(
        project_id=analyze_request.project_id
    )
    
    if not project:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "signal": "project_not_found",
                "message": f"Project {analyze_request.project_id} not found"
            }
        )
    
    # Analyze structure
    structure_controller = StructureController(
        generation_client=request.app.generation_client
    )
    
    structure = await structure_controller.analyze_lecture_structure(
        chunk_model=chunk_model,
        project_id=project.project_id,
        max_topics=analyze_request.max_topics
    )
    
    if not structure:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "signal": "structure_analysis_failed",
                "message": "Failed to analyze document structure"
            }
        )
    
    # Format response
    topics_response = [
        TopicResponse(
            title=topic["title"],
            order=topic["order"],
            subtitles=[
                SubtitleResponse(
                    title=sub["title"],
                    order=sub["order"]
                )
                for sub in topic.get("subtitles", [])
            ]
        )
        for topic in structure.get("topics", [])
    ]
    
    return JSONResponse(
        content={
            "signal": "structure_analysis_success",
            "project_id": analyze_request.project_id,
            "lecture_id": analyze_request.lecture_id,
            "topics": [topic.dict() for topic in topics_response]
        }
    )