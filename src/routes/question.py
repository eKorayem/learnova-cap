import logging
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from routes.schemas.question import (
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    TopicError
)
from routes.schemas.data import ProcessRequest
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from models.AssetModel import AssetModel
from models.db_schemas import DataChunk
from controllers.QuestionGenerationController import QuestionGenerationController
from controllers.ProcessController import ProcessController
from models.enums.ResponseEnums import ResponseSignal
from models.enums.AssetTypeEnum import AssetTypeEnum

logger = logging.getLogger('uvicorn.error')

question_router = APIRouter(
    prefix="/api/v1/courses",
    tags=["api_v1", "courses"]
)


# =============================================================
# STEP 1: Process document with question chunk size
# Call this ONCE per document before generating questions
# =============================================================

@question_router.post("/{project_id}/documents/chunks/questions")
async def process_for_questions(
    request: Request,
    project_id: str,
    process_request: ProcessRequest
):
    """
    Chunk the document using QUESTION_CHUNK_SIZE (larger chunks = more
    context for the LLM when generating questions).
    Must be called before /generate.
    """

    asset_model = await AssetModel.create_instance(db_client=request.app.db_client)
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

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
        # Only delete question chunks — leave rag/structure chunks untouched
        _ = await chunk_model.delete_chunks_by_project_id(
            project_id=project.project_id,
            chunk_type="question"
        )

    question_chunk_size = process_request.chunk_size or request.app.settings.QUESTION_CHUNK_SIZE
    question_overlap_size = process_request.overlap_size or request.app.settings.QUESTION_OVERLAP_SIZE

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
            chunk_size=question_chunk_size,
            overlap_size=question_overlap_size,
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
                chunk_asset_id=asset_id,
                chunk_type="question"       # ← tagged
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
            "chunk_size_used": question_chunk_size,
            "overlap_size_used": question_overlap_size,
            "chunk_type": "question"
        }
    )


# =============================================================
# STEP 2: Generate questions using question chunks
# =============================================================

@question_router.post("/{project_id}/questions/generate", response_model=GenerateQuestionsResponse)
async def generate_questions(
    request: Request,
    project_id: str,
    generate_request: GenerateQuestionsRequest
):
    """
    Generate questions for a course based on topic configurations.
    Reads only chunks with chunk_type='question'.
    """

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    if not project:
        print("NOT PROJECT FOUND")
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=GenerateQuestionsResponse(
                request_id=generate_request.request_id,
                course_id=generate_request.course_id,
                project_id=project_id,
                status="failed",
                errors=[
                    TopicError(
                        topic_id=None,
                        topic_title=None,
                        reason=f"Project {project_id} not found in the AI system"
                    )
                ],
                questions=[]
            ).dict()
        )

    chunk_model = await ChunkModel.create_instance(
        db_client=request.app.db_client
    )

    controller = QuestionGenerationController(
        generation_client=request.app.question_generation_client
    )

    result = await controller.generate_all(
        request=generate_request,
        chunk_model=chunk_model
    )

    print(result)
    if result.status == "failed":
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=result.dict()
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=result.dict()
    )