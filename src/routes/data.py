import os
import logging
import aiofiles
import httpx
from bson import ObjectId
from fastapi import APIRouter, Depends, UploadFile, status, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from helpers.config import get_settings, Settings
from core.security.dependencies import verify_backend_signature

from controllers import DataController, ProjectController, ProcessController, NLPController
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from models.AssetModel import AssetModel
from models.db_schemas import DataChunk, Asset
from models.enums.AssetTypeEnum import AssetTypeEnum
from models import ResponseSignal

from .schemas.data import ProcessRequest, DocumentWebhookPayload

from core.security.callback import send_webhook_callback

load_dotenv(".env")
logger = logging.getLogger('uvicorn.error')

data_router = APIRouter(
    prefix="/api/v1/courses",
    tags=["api_v1", "courses"]
)

# =============================================================
# 1. ASYNC DOWNLOADER (Helper for Webhook)
# =============================================================
async def _download_and_process_background(
    app, project, payload: DocumentWebhookPayload
):
    """
    Downloads the file asynchronously from the signed URL, registers it as an asset,
    and then triggers the master processing pipeline.
    """
    logger.info(f"Background Task: Downloading material {payload.body.material_id} for project {project.project_id}")
    
    data_controller = DataController()
    file_name = payload.body.material.file_name
    file_path, file_id = data_controller.generate_unique_file_path(
        orig_file_name=file_name,
        project_id=project.project_id
    )

    download_url = payload.body.material.signed_download_url
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("GET", download_url) as response:
                response.raise_for_status()
                async with aiofiles.open(file_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=app.settings.FILE_DEFAULT_CHUNK_SIZE):
                        await f.write(chunk)
    except Exception as e:
        logger.error(f"Failed to download file from {download_url}: {e}")
        # TODO: Send failure callback to Learnova backend here
        return

    asset_model = await AssetModel.create_instance(db_client=app.db_client)
    asset_resource = Asset(
        asset_project_id=project.id,
        asset_type=AssetTypeEnum.FILE.value,
        asset_name=file_id,
        asset_size=os.path.getsize(file_path)
    )
    asset_record = await asset_model.create_asset(asset=asset_resource)
    
    logger.info(f"Successfully downloaded and saved asset: {file_id}")

    project_files_ids = {asset_record.id: asset_record.asset_name}
    await _master_process_background(
        app=app,
        project=project,
        project_files_ids=project_files_ids,
        do_reset=0,
        request_id=payload.request_id,
        course_id=payload.course_id
    )


# =============================================================
# 2. THE DEVELOPER BACKDOOR (Local File Upload - No Signature Required)
# =============================================================
@data_router.post("/{project_id}/documents")
async def upload_data(
        request: Request,
        project_id: str,
        file: UploadFile,
        background_tasks: BackgroundTasks,
        app_settings: Settings = Depends(get_settings)
):
    """
    Legacy direct-upload endpoint for local developer testing.
    """
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    data_controller = DataController()
    is_valid, result_signal = data_controller.validate_uploaded_file(file=file)

    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"Signal": result_signal}
        )

    file_path, file_id = data_controller.generate_unique_file_path(
        orig_file_name=file.filename, project_id=project_id
    )

    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
    except Exception as e:
        logger.error(f"Error while uploading file: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"Signal": ResponseSignal.FILE_UPLOAD_FAILED.value}
        )

    asset_model = await AssetModel.create_instance(db_client=request.app.db_client)
    asset_resource = Asset(
        asset_project_id=project.id, asset_type="file",
        asset_name=file_id, asset_size=os.path.getsize(file_path)
    )
    asset_record = await asset_model.create_asset(asset=asset_resource)
    
    project_files_ids = {asset_record.id: asset_record.asset_name}

    background_tasks.add_task(
        _master_process_background,
        app=request.app, project=project, project_files_ids=project_files_ids, do_reset=0
    )
    
    return JSONResponse(
        content={
            "signal": ResponseSignal.FILE_UPLOAD_SUCCESS.value,
            "file_id": str(asset_record.id),
            "status": "processing_started",
            "message": "File uploaded and background processing has started automatically."
        }
    )


# =============================================================
# 3. PRODUCTION WEBHOOK (Cloud URL - Cryptographically Secured)
# =============================================================
@data_router.post("/documents/ingest")
async def ingest_document_webhook(
    request: Request,
    payload: DocumentWebhookPayload,
    background_tasks: BackgroundTasks,
    request_id: str = Depends(verify_backend_signature) # <-- THE GATEKEEPER
):
    """
    Production webhook endpoint to receive a document download URL from the main Learnova backend.
    """
    project_id = str(payload.body.material_id)
    
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    background_tasks.add_task(
        _download_and_process_background, 
        app=request.app, project=project, payload=payload
    )
    
    return JSONResponse(
        content={
            "status": "processing_started",
            "request_id": request_id,
            "message": "Document ingestion queued successfully."
        }
    )


# =============================================================
# 4. CHUNKING & DB LOGIC 
# =============================================================

@data_router.post("/{project_id}/documents/chunks")
async def process_endpoint(request: Request, project_id: str, process_request: ProcessRequest):
    chunk_size = process_request.chunk_size or request.app.settings.RAG_CHUNK_SIZE
    overlap_size = process_request.overlap_size or request.app.settings.RAG_OVERLAP_SIZE
    do_reset = process_request.do_reset

    asset_model = await AssetModel.create_instance(db_client=request.app.db_client)
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    project_files_ids = {}
    if process_request.file_id:
        asset_record = await asset_model.get_asset_record(
            asset_project_id=project.id, asset_name=process_request.file_id
        )
        if asset_record is None:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"signal": ResponseSignal.FILE_ID_ERROR.value}
            )
        project_files_ids = {asset_record.id: asset_record.asset_name}
    else:
        project_files = await asset_model.get_all_projects_assets(
            asset_project_id=project.id, asset_type=AssetTypeEnum.FILE.value
        )
        project_files_ids = {record.id: record.asset_name for record in project_files}

    if len(project_files_ids) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.NO_FILES_ERROR.value}
        )

    process_controller = ProcessController(project_id=project_id)
    no_records, no_files = 0, 0
    chunk_model = await ChunkModel.create_instance(db_client=request.app.db_client)

    if do_reset == 1:
        _ = await chunk_model.delete_chunks_by_project_id(
            project_id=project.project_id, chunk_type="rag"
        )

    for asset_id, file_id in project_files_ids.items():
        file_content = process_controller.get_file_content(file_id=file_id)
        if file_content is None:
            logger.error(f"Error while processing {file_id}")
            continue

        file_chunks = process_controller.process_file_content(
            file_id=file_id, file_content=file_content,
            chunk_size=chunk_size, overlap_size=overlap_size,
        )

        if file_chunks is None or len(file_chunks) == 0:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"signal": ResponseSignal.PROCESSING_FAILD.value}
            )

        file_chunks_records = [
            DataChunk(
                chunk_text=chunk.page_content, chunk_metadata=chunk.metadata, chunk_order=i + 1,
                project_id=project.project_id, chunk_project_id=project.id,
                chunk_asset_id=asset_id, chunk_type="rag"
            ) for i, chunk in enumerate(file_chunks)
        ]

        no_records += await chunk_model.insert_many_chunks(chunks=file_chunks_records)
        no_files += 1

    return JSONResponse(
        content={
            "signal": ResponseSignal.PROCESSING_SUCCESS.value,
            "inserted_chunks": no_records,
            "processed_files": no_files,
        }
    )

async def _master_process_background(
    app, project, project_files_ids: dict, do_reset: int,
    request_id: str = None, course_id: int = None # <-- Added optional parameters,
):
    logger.info(f"Starting background processing for project: {project.project_id}")
    try:
        db_client = app.db_client
        chunk_model = await ChunkModel.create_instance(db_client=db_client)

        if do_reset:
            logger.info(f"Clearing old chunks for project {project.project_id}...")
            await chunk_model.delete_chunks_by_project_id(project_id=project.project_id)

        process_controller = ProcessController(project_id=project.project_id)
        settings = app.settings
        
        all_chunk_records, rag_records = [], []

        for asset_id, file_name in project_files_ids.items():
            logger.info(f"Processing file: {file_name}")
            file_content = process_controller.get_file_content(file_id=file_name)

            if not file_content:
                logger.error(f"Could not read file content for {file_name}")
                continue

            struct_chunks = process_controller.process_file_content(
                file_id=file_name, file_content=file_content,
                chunk_size=settings.STRUCTURE_CHUNK_SIZE, overlap_size=settings.STRUCTURE_OVERLAP_SIZE
            )
            if struct_chunks:
                all_chunk_records.extend([
                    DataChunk(
                        chunk_text=c.page_content, chunk_metadata=c.metadata, chunk_order=i+1,
                        project_id=project.project_id, chunk_project_id=project.id,
                        chunk_asset_id=asset_id, chunk_type="structure"
                    ) for i, c in enumerate(struct_chunks)
                ])

            quest_chunks = process_controller.process_file_content(
                file_id=file_name, file_content=file_content,
                chunk_size=settings.QUESTION_CHUNK_SIZE, overlap_size=settings.QUESTION_OVERLAP_SIZE
            )
            if quest_chunks:
                all_chunk_records.extend([
                    DataChunk(
                        chunk_text=c.page_content, chunk_metadata=c.metadata, chunk_order=i+1,
                        project_id=project.project_id, chunk_project_id=project.id,
                        chunk_asset_id=asset_id, chunk_type="question"
                    ) for i, c in enumerate(quest_chunks)
                ])

            rag_chunks = process_controller.process_file_content(
                file_id=file_name, file_content=file_content,
                chunk_size=settings.RAG_CHUNK_SIZE, overlap_size=settings.RAG_OVERLAP_SIZE
            )
            if rag_chunks:
                file_rag_records = [
                    DataChunk(
                        chunk_text=c.page_content, chunk_metadata=c.metadata, chunk_order=i+1,
                        project_id=project.project_id, chunk_project_id=project.id,
                        chunk_asset_id=asset_id, chunk_type="rag"
                    ) for i, c in enumerate(rag_chunks)
                ]
                rag_records.extend(file_rag_records)
                all_chunk_records.extend(file_rag_records)

        if all_chunk_records:
            logger.info(f"Saving {len(all_chunk_records)} total chunks to MongoDB...")
            await chunk_model.insert_many_chunks(chunks=all_chunk_records, batch_size=500)

        if rag_records:
            logger.info(f"Embedding {len(rag_records)} RAG vectors into Qdrant...")
            nlp_controller = NLPController(
                vectordb_client=app.vectordb_client, generation_client=app.generation_client,
                embedding_client=app.embedding_client, template_parser=app.template_parser
            )
            chunks_ids = list(range(len(rag_records)))
            await nlp_controller.index_into_vector_db(
                project=project, chunks=rag_records, chunks_ids=chunks_ids, do_reset=do_reset
            )

        logger.info(f"✅ Master processing COMPLETE for project: {project.project_id}")
        # --- NEW CALLBACK TRIGGER ---
        if request_id and course_id:
            await send_webhook_callback(
                request_id=request_id,
                course_id=course_id,
                operation_type="document_ingestion",
                status="success",
                message=f"Document successfully chunked and vectorized."
            )

    except Exception as e:
        logger.error(f"Error in master processing for {project.project_id}: {e}")
        # Send failure callback if it crashes!
        if request_id and course_id:
            await send_webhook_callback(
                request_id=request_id, course_id=course_id,
                operation_type="document_ingestion", status="failed", message=str(e)
            )

@data_router.post("/{project_id}/process", status_code=status.HTTP_202_ACCEPTED)
async def process_document_master(
    request: Request, project_id: str,
    process_request: ProcessRequest, background_tasks: BackgroundTasks
):
    asset_model = await AssetModel.create_instance(db_client=request.app.db_client)
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)
    
    project_files_ids = {}
    if process_request.file_id:
        asset_record = await asset_model.get_asset_record(
            asset_project_id=project.id, asset_name=process_request.file_id
        )
        if not asset_record:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"signal": ResponseSignal.FILE_ID_ERROR.value, "message": "File not found."}
            )
        project_files_ids = {asset_record.id: asset_record.asset_name}
    else:
        project_files = await asset_model.get_all_projects_assets(
            asset_project_id=project.id, asset_type=AssetTypeEnum.FILE.value
        )
        project_files_ids = {record.id: record.asset_name for record in project_files}

    if not project_files_ids:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.NO_FILES_ERROR.value, "message": "No files found to process."}
        )

    background_tasks.add_task(
        _master_process_background,
        app=request.app, project=project, project_files_ids=project_files_ids, do_reset=process_request.do_reset
    )

    return {
        "signal": ResponseSignal.PROCESSING_SUCCESS.value,
        "status": "processing_started",
        "message": f"Processing {len(project_files_ids)} document(s) in the background."
    }

@data_router.post("/test-secure-endpoint")
async def secure_test(
    request: Request,
    request_id: str = Depends(verify_backend_signature) 
):
    return {"message": f"Security passed! Processing request: {request_id}"}