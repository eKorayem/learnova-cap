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
    message: str
):
    """
    The Mailman: Signs and sends the callback payload back to the Learnova backend.
    """
    settings = get_settings()
    
    # 1. Setup the destination (Ask your backend team for the exact path they want)
    callback_path = "/api/v1/ai/callback"
    callback_url = f"{settings.LEARNOVA_BACKEND_URL}{callback_path}"
    
    # 2. Generate a fresh timestamp
    timestamp = get_current_timestamp()
    
    # 3. Build the exact envelope they requested
    payload_dict = build_ai_request_envelope(
        request_id=request_id,
        timestamp=timestamp,
        operation_type=operation_type,
        course_id=course_id,
        body={
            "status": status,
            "message": message
        }
    )
    
    # 4. Serialize the JSON exactly how it will travel over the network
    serialized_body = serialize_json_for_signing(payload_dict)
    
    # 5. Calculate the HMAC SHA-256 Signature
    signature = create_signature_from_bytes(
        secret=settings.AI_SHARED_SECRET,
        method="POST",
        path=callback_path,
        request_id=request_id,
        timestamp=timestamp,
        body=serialized_body
    )
    
    # 6. Build the security headers
    headers = build_ai_request_headers(
        request_id=request_id,
        timestamp=timestamp,
        signature=signature
    )
    
    # 7. Send the Callback asynchronously
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(callback_url, content=serialized_body, headers=headers)
            response.raise_for_status()
            logger.info(f"Callback sent successfully for request {request_id}")
    except Exception as e:
        logger.error(f"Failed to send callback to Learnova backend: {e}")