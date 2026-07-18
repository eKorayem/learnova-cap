import logging
import json
from fastapi import APIRouter, Request, status, BackgroundTasks, Depends
from fastapi.responses import JSONResponse

from routes.schemas.summary import SummaryWebhookPayload
from models.ChunkModel import ChunkModel
from models.ProjectModel import ProjectModel
from models.AssetModel import AssetModel
from models.enums.AssetTypeEnum import AssetTypeEnum
from controllers.SummaryController import SummaryController
from core.security.dependencies import verify_backend_signature
from core.security.callback import send_webhook_callback

logger = logging.getLogger('uvicorn.error')

summary_router = APIRouter(
    prefix="/api/v1/courses",
    tags=["api_v1", "summary"]
)

async def _generate_summary_background(app, project, payload: SummaryWebhookPayload):
    logger.info(f"Background Task: Generating summary for course {project.project_id}")
    
    try:
        chunk_model = await ChunkModel.create_instance(db_client=app.db_client)
        asset_model = await AssetModel.create_instance(db_client=app.db_client)
        
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

        # Assuming you instantiated summary_client in main.py
        controller = SummaryController(generation_client=app.summary_client)

        target_sections = [section.dict() for section in payload.body.target_sections]

        result_payload = await controller.generate_summary(
            chunk_model=chunk_model,
            project_id=project.project_id,
            target_sections=target_sections,
            summary_type=payload.body.summary_type,
            asset_id=target_asset_id
        )

        if result_payload and "sections" in result_payload and len(result_payload["sections"]) > 0:
            logger.info("Summary generated successfully.")
            await send_webhook_callback(
                request_id=payload.request_id,
                course_id=payload.course_id,
                operation_type=payload.operation_type,
                status="success", 
                message="Summary generated successfully.",
                data=result_payload
            )
        else:
            raise ValueError("LLM returned empty or invalid summary data.")

    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        await send_webhook_callback(
            request_id=payload.request_id,
            course_id=payload.course_id,
            operation_type=payload.operation_type,
            status="failed",
            message=f"Summary generation failed: {str(e)[:100]}",
            data={"title": "", "executive_summary": "", "sections": []}
        )

@summary_router.post("/summary/generate")
async def generate_summary_webhook(
    request: Request,
    payload: SummaryWebhookPayload,
    background_tasks: BackgroundTasks,
    secure_request_id: str = Depends(verify_backend_signature)
):
    project_id = str(payload.course_id)

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    if not project:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"message": f"Course ID {project_id} not found."}
        )

    background_tasks.add_task(
        _generate_summary_background,
        app=request.app, project=project, payload=payload
    )

    return JSONResponse(
        content={
            "status": "processing_started",
            "request_id": secure_request_id,
            "message": "Summary generation queued successfully."
        }
    )