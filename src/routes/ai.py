import logging
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from routes.schemas.ai import (
    AnalyzeStructureRequest,
    AnalyzeStructureResponse,
    TopicResponse,
    SubtitleResponse
)
from routes.schemas.data import ProcessRequest
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from models.AssetModel import AssetModel
from models.db_schemas import DataChunk
from controllers import StructureController, ProcessController
from models.enums.ResponseEnums import ResponseSignal
from models.enums.AssetTypeEnum import AssetTypeEnum

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
        max_topics=analyze_request.max_topics,
        use_all_chunks=analyze_request.use_all_chunks
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


@ai_router.post("/process-with-structure/{project_id}")
async def process_with_structure(
    request: Request,
    project_id: str,
    process_request: ProcessRequest
):
    """
    Process documents with structure-aware chunking.
    Uses larger chunk sizes to preserve topic context for better structure analysis.
    """

    asset_model = await AssetModel.create_instance(db_client=request.app.db_client)
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    # Get assets
    project_files_ids = {}
    if process_request.file_id:
        asset_record = await asset_model.get_asset_record(
            asset_project_id=project.id,
            asset_name=process_request.file_id
        )
        if asset_record is None:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"signal": ResponseSignal.FILE_ID_ERROR.value}
            )
        project_files_ids = {asset_record.id: asset_record.asset_name}
    else:
        project_files = await asset_model.get_all_projects_assets(
            asset_project_id=project.id,
            asset_type=AssetTypeEnum.FILE.value
        )
        project_files_ids = {record.id: record.asset_name for record in project_files}

    if len(project_files_ids) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.NO_FILES_ERROR.value}
        )

    process_controller = ProcessController(project_id=project_id)
    chunk_model = await ChunkModel.create_instance(db_client=request.app.db_client)

    if process_request.do_reset:
        _ = await chunk_model.delete_chunks_by_project_id(project_id=project.project_id)

    # Use larger chunks for structure preservation (from config)
    structure_chunk_size = process_request.chunk_size or request.app.settings.STRUCTURE_CHUNK_SIZE
    structure_overlap_size = process_request.overlap_size or request.app.settings.STRUCTURE_OVERLAP_SIZE

    no_records = 0
    no_files = 0

    for asset_id, file_id in project_files_ids.items():
        file_content = process_controller.get_file_content(file_id=file_id)

        if file_content is None:
            logger.error(f"Error while processing {file_id}")
            continue

        file_chunks = process_controller.process_file_content(
            file_id=file_id,
            file_content=file_content,
            chunk_size=structure_chunk_size,
            overlap_size=structure_overlap_size,
        )

        if file_chunks is None or len(file_chunks) == 0:
            continue

        file_chunks_records = [
            DataChunk(
                chunk_text=chunk.page_content,
                chunk_metadata=chunk.metadata,
                chunk_order=i + 1,
                project_id=project.project_id,
                chunk_project_id=project.id,
                chunk_asset_id=asset_id
            )
            for i, chunk in enumerate(file_chunks)
        ]

        no_records += await chunk_model.insert_many_chunks(chunks=file_chunks_records)
        no_files += 1

    return JSONResponse(
        content={
            "signal": ResponseSignal.PROCESSING_SUCCESS.value,
            "inserted_chunks": no_records,
            "processed_files": no_files,
            "chunk_size_used": structure_chunk_size,
            "overlap_size_used": structure_overlap_size,
        }
    )