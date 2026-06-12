import logging
from fastapi import APIRouter, Request, status, BackgroundTasks, Depends
from fastapi.responses import JSONResponse

from routes.schemas.grading import GradingWebhookPayload
from controllers.GradingController import GradingController
from core.security.dependencies import verify_backend_signature
from core.security.callback import send_webhook_callback

logger = logging.getLogger('uvicorn.error')

grading_router = APIRouter(
    prefix="/api/v1/courses",
    tags=["api_v1", "grading"]
)

async def _grade_exam_background(
    app, payload: GradingWebhookPayload
):
    """Background task to grade the exam and send the callback."""
    logger.info(f"Background Task: Grading exam {payload.body.exam_id} (Attempt: {payload.body.attempt_id})")
    
    # Initialize the controller using the app's LLM client
    controller = GradingController(generation_client=app.question_generation_client)

    # Run the grading engine
    graded_results = await controller.evaluate_exam(request_data=payload.body)

    # Format exactly as the backend expects
    callback_payload = {
        "attempt_id": payload.body.attempt_id,
        "exam_id": payload.body.exam_id,
        "results": [result.dict() for result in graded_results]
    }

    if len(graded_results) > 0:
        await send_webhook_callback(
            request_id=payload.request_id,
            course_id=payload.course_id,
            operation_type=payload.operation_type,
            status="success", # Will be translated to "completed" by your utility
            message="Exam graded successfully.",
            data=callback_payload
        )
    else:
        await send_webhook_callback(
            request_id=payload.request_id,
            course_id=payload.course_id,
            operation_type=payload.operation_type,
            status="failed",
            message="Exam grading failed. Check AI logs.",
            data={"attempt_id": payload.body.attempt_id, "exam_id": payload.body.exam_id, "results": []}
        )

@grading_router.post("/grading/evaluate")
async def evaluate_exam_webhook(
    request: Request,
    payload: GradingWebhookPayload,
    background_tasks: BackgroundTasks,
    secure_request_id: str = Depends(verify_backend_signature) # THE GATEKEEPER
):
    """
    Webhook Endpoint: Grade subjective exam questions asynchronously.
    """
    
    # Toss it to the background so we don't timeout the HTTP request!
    background_tasks.add_task(
        _grade_exam_background,
        app=request.app, payload=payload
    )

    return JSONResponse(
        content={
            "status": "processing_started",
            "request_id": secure_request_id,
            "message": "Exam grading queued successfully."
        }
    )