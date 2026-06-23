import logging
from fastapi import APIRouter, Request, status, BackgroundTasks, Depends
from fastapi.responses import JSONResponse

from routes.schemas.question import (
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    TopicError,
    QuestionWebhookPayload
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

from core.security.dependencies import verify_backend_signature
from core.security.callback import send_webhook_callback

import json

logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.INFO) # Turned the lights on for debugging!

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
# STEP 2: Generate questions using question chunks (WEBHOOK)
# =============================================================

async def _generate_questions_background(
    app, project, payload: QuestionWebhookPayload
):
    """Background task to generate questions and send the callback."""
    logger.info(f"Background Task: Generating questions for course {project.project_id}")
    
    chunk_model = await ChunkModel.create_instance(db_client=app.db_client)
    controller = QuestionGenerationController(generation_client=app.qgen_client) # <-- Updated!

    # Adapt the payload to fit the controller's existing logic
    adapted_request = GenerateQuestionsRequest(
        request_id=payload.request_id,
        course_id=payload.course_id,
        project_id=project.project_id, # Using course_id as the master project_id
        topics=payload.body.topics
    )

    result = await controller.generate_all(
        request=adapted_request,
        chunk_model=chunk_model
    )

    # ==========================================
    # FORMAT DATA EXACTLY AS BACKEND EXPECTS
    # ==========================================
    formatted_questions = []
    
    logger.info(f"============= Result ===================:\n{result}")

    for q in result.questions:
        q_dict = q.dict()
        # The backend strict validation rejects this extra field, so we remove it
        q_dict.pop("topic_title", None) 
        formatted_questions.append(q_dict)

    # DEBUG BLOCK - Updated to show what we are ACTUALLY sending now
    callback_payload = {
        "questions": formatted_questions
    }
    logger.info(f"SENDING THIS TO BACKEND:\n{json.dumps(callback_payload, indent=2)}")

    # Send the Webhook Callback
    if result.status == "completed" or result.status == "partial":
        await send_webhook_callback(
            request_id=payload.request_id,
            course_id=payload.course_id,
            operation_type=payload.operation_type,
            status="success", # Changed from "success" to match backend expectation
            message="Questions generated successfully.",
            data={
                "questions": formatted_questions
                # REMOVED "errors" array completely!
            }
        )
    else:
        await send_webhook_callback(
            request_id=payload.request_id,
            course_id=payload.course_id,
            operation_type=payload.operation_type,
            status="failed",
            message="Failed to generate questions. Check AI logs.",
            # REMOVED material_id to prevent crashes!
            data={
                "questions": [] # Send empty questions on failure so his schema doesn't break
            }
        )


@question_router.post("/questions/generate")
async def generate_questions_webhook(
    request: Request,
    payload: QuestionWebhookPayload,
    background_tasks: BackgroundTasks,
    secure_request_id: str = Depends(verify_backend_signature) # THE GATEKEEPER
):
    """
    Webhook Endpoint: Generate questions asynchronously and return via callback.
    """
    # Use the course_id as the master project_id to search across all files in the course
    project_id = str(payload.course_id)

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    if not project:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"message": f"Course ID {project_id} not found in the AI system."}
        )

    # Toss it to the background so we don't timeout the HTTP request!
    background_tasks.add_task(
        _generate_questions_background,
        app=request.app, project=project, payload=payload
    )

    return JSONResponse(
        content={
            "status": "processing_started",
            "request_id": secure_request_id,
            "message": "Question generation queued successfully."
        }
    )