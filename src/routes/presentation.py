import logging
import json
from fastapi import APIRouter, Request, status, BackgroundTasks, Depends
from fastapi.responses import JSONResponse

from routes.schemas.presentation import PresentationWebhookPayload
from models.ChunkModel import ChunkModel
from models.ProjectModel import ProjectModel
from models.AssetModel import AssetModel
from models.enums.AssetTypeEnum import AssetTypeEnum
from controllers.PresentationController import PresentationController
from core.security.dependencies import verify_backend_signature
from core.security.callback import send_webhook_callback

logger = logging.getLogger('uvicorn.error')

presentation_router = APIRouter(
    prefix="/api/v1/courses",
    tags=["api_v1", "presentation"]
)

async def _generate_presentation_background(app, project, payload: PresentationWebhookPayload):
    """Background task to generate presentation slides and send the callback."""
    logger.info(f"Background Task: Generating presentation for course {project.project_id}")
    
    try:
        chunk_model = await ChunkModel.create_instance(db_client=app.db_client)
        asset_model = await AssetModel.create_instance(db_client=app.db_client)
        
        # 1. Target the specific PDF using material_id
        target_asset_id = None
        if payload.body.material_id:
            project_files = await asset_model.get_all_projects_assets(
                asset_project_id=project.id,
                asset_type=AssetTypeEnum.FILE.value
            )
            for asset in project_files:
                if asset.asset_config and asset.asset_config.get("material_id") == payload.body.material_id:
                    target_asset_id = asset.id
                    break

        # 2. Instantiate the controller
        controller = PresentationController(generation_client=app.generation_client)

        # 3. Extract target sections as a list of dicts
        target_sections = [section.dict() for section in payload.body.target_sections]

        # 4. Run the presentation generation engine
        result_payload = await controller.generate_presentation(
            chunk_model=chunk_model,
            project_id=project.project_id,
            target_sections=target_sections,
            slide_count=payload.body.slide_count,
            asset_id=target_asset_id
        )

        # 5. Handle the callback
        if result_payload and "slides" in result_payload and len(result_payload["slides"]) > 0:
            logger.info(f"Presentation generated successfully: {len(result_payload['slides'])} slides.")
            
            await send_webhook_callback(
                request_id=payload.request_id,
                course_id=payload.course_id,
                operation_type=payload.operation_type,
                status="success", 
                message="Presentation generated successfully.",
                data=result_payload
            )
        else:
            raise ValueError("LLM returned empty or invalid slide data.")

    except Exception as e:
        logger.error(f"Presentation generation failed: {e}")
        await send_webhook_callback(
            request_id=payload.request_id,
            course_id=payload.course_id,
            operation_type=payload.operation_type,
            status="failed",
            message=f"Presentation generation failed: {str(e)[:100]}",
            data={"slides": []}
        )

@presentation_router.post("/presentation/generate")
async def generate_presentation_webhook(
    request: Request,
    payload: PresentationWebhookPayload,
    background_tasks: BackgroundTasks,
    secure_request_id: str = Depends(verify_backend_signature)
):
    """
    Webhook Endpoint: Generate a structured slide deck asynchronously.
    """
    project_id = str(payload.course_id)

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    if not project:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"message": f"Course ID {project_id} not found in the AI system."}
        )

    # Offload to background task
    background_tasks.add_task(
        _generate_presentation_background,
        app=request.app, project=project, payload=payload
    )

    return JSONResponse(
        content={
            "status": "processing_started",
            "request_id": secure_request_id,
            "message": "Presentation generation queued successfully."
        }
    )