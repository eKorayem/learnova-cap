import logging
from fastapi import APIRouter, Request, BackgroundTasks, status, Depends
from fastapi.responses import JSONResponse

from routes.schemas.extraction import ExtractionWebhookPayload
from models.ChunkModel import ChunkModel
from models.ProjectModel import ProjectModel
from models.AssetModel import AssetModel
from models.enums.AssetTypeEnum import AssetTypeEnum
from controllers.ExtractionController import ExtractionController
from core.security.dependencies import verify_backend_signature
from core.security.callback import send_webhook_callback

logger = logging.getLogger('uvicorn.error')

extraction_router = APIRouter(
    prefix="/api/v1/courses",
    tags=["api_v1", "extraction"]
)

async def _extract_questions_background(app, project, payload: ExtractionWebhookPayload):
    logger.info(f"Background Task: Extracting native questions for material {payload.body.material_id}")
    
    try:
        # 1. Fetch chunks using the structure chunks (largest context size, best for reading)
        chunk_model = await ChunkModel.create_instance(db_client=app.db_client)
        asset_model = await AssetModel.create_instance(db_client=app.db_client)
        
        # Find the specific asset for this material
        project_files = await asset_model.get_all_projects_assets(
            asset_project_id=project.id,
            asset_type=AssetTypeEnum.FILE.value
        )
        
        target_asset_id = None
        for asset in project_files:
            if asset.asset_config and asset.asset_config.get("material_id") == payload.body.material_id:
                target_asset_id = asset.id
                break

        chunks = await chunk_model.get_chunks_by_project_id(
            project_id=project.project_id,
            page_no=1,
            page_size=3000,
            chunk_type="structure",
            asset_id=target_asset_id
        )

        if not chunks:
            raise ValueError("No text chunks found for this material.")

        # Stitch document back together in order
        chunks = sorted(chunks, key=lambda c: c.chunk_order)
        full_text = "\n\n".join([c.chunk_text for c in chunks])

        # 2. Run the Extraction Controller
        controller = ExtractionController(generation_client=app.structure_client)
        result = await controller.extract_native_questions(
            document_text=full_text,
            topics=payload.body.topics  # Pass the list of Topic objects
        )

        if not result or "extracted_questions" not in result:
            raise ValueError("LLM failed to return a valid extraction schema.")
        
    
        # ==========================================
        # PYTHON SAFETY NET: Fix nulls
        # ==========================================
        for q in result["extracted_questions"]:
            # Fix null expected_answer
            if q.get("type") == "short_answer" and not q.get("expected_answer"):
                q["expected_answer"] = "Please refer to the grading rubric for criteria."
            if not q.get("expected_answer"):
                q["expected_answer"] = "N/A"
            
            # Fix null difficulty (Backend 422 Fix)
            if not q.get("difficulty"):
                q["difficulty"] = "medium"
        # ==========================================

        # 3. Send Success Callback
        await send_webhook_callback(
            request_id=payload.request_id,
            course_id=payload.course_id,
            operation_type=payload.operation_type,
            status="success",
            message=f"Successfully extracted {len(result['extracted_questions'])} native questions.",
            module_id=payload.body.module_id,
            material_id=payload.body.material_id,
            data={"extracted_questions": result["extracted_questions"]}  # <--- CHANGED THIS KEY
        )

    except Exception as e:
        logger.error(f"Native extraction failed: {e}")
        # Send Failure Callback
        await send_webhook_callback(
            request_id=payload.request_id,
            course_id=payload.course_id,
            operation_type=payload.operation_type,
            status="failed",
            message=f"Extraction failed: {str(e)[:100]}",
            module_id=payload.body.module_id,
            material_id=payload.body.material_id,
            data={"extracted_questions": []}  # <--- CHANGED THIS KEY
        )

@extraction_router.post("/extraction/questions")
async def extract_questions_endpoint(
    request: Request,
    payload: ExtractionWebhookPayload,
    background_tasks: BackgroundTasks,
    secure_request_id: str = Depends(verify_backend_signature)
):
    project_id = str(payload.course_id)
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    background_tasks.add_task(
        _extract_questions_background,
        app=request.app, project=project, payload=payload
    )
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "processing_started",
            "request_id": secure_request_id,
            "message": "Native question extraction queued successfully."
        }
    )