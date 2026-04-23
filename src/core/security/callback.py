import httpx
import logging
from core.security.ai_protocol import build_ai_request_envelope, build_ai_request_headers, serialize_json_for_signing
from core.security.ai_signature import get_current_timestamp, create_signature_from_bytes
from helpers.config import get_settings

logger = logging.getLogger(__name__)

async def send_webhook_callback(
    request_id: str, 
    course_id: int, 
    operation_type: str, 
    status: str, 
    message: str,
    module_id: int = None,     # <--- ADDED
    material_id: int = None,   # <--- ADDED
    data: dict = None
):
    """
    The Mailman: Signs and sends the callback payload back to the Learnova backend.
    """
    settings = get_settings()
    callback_path = "/ai/callback"
    callback_url = f"{settings.LEARNOVA_BACKEND_URL.rstrip('/')}{callback_path}"
    timestamp = get_current_timestamp()
    
    # Build the body exactly how the backend guy requested
    body_payload = {}
    if module_id is not None:
        body_payload["module_id"] = module_id
    if material_id is not None:
        body_payload["material_id"] = material_id
        
    if data:
        body_payload.update(data) 
        
    if status != "success":
        body_payload["message"] = message # Only send the message string if it failed

    # Build the envelope with STATUS AT THE ROOT
    payload_dict = {
        "request_id": request_id,
        "timestamp": timestamp,
        "operation_type": operation_type,
        "course_id": course_id,
        "status": "completed" if status == "success" else "failed", # Backend uses "completed"
        "body": body_payload
    }
    
    # Serialize and sign
    serialized_body = serialize_json_for_signing(payload_dict)
    
    signature = create_signature_from_bytes(
        secret=settings.AI_SHARED_SECRET,
        method="POST",
        path=callback_path,
        request_id=request_id,
        timestamp=timestamp,
        body=serialized_body
    )
    
    headers = build_ai_request_headers(
        request_id=request_id,
        timestamp=timestamp,
        signature=signature
    )
    
    # Send the Callback
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(callback_url, content=serialized_body, headers=headers)
            response.raise_for_status()
            logger.info(f"Callback sent successfully for request {request_id}")
    except Exception as e:
        logger.error(f"Failed to send callback to Learnova backend: {e}")