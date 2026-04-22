from fastapi import Request, HTTPException, Header, Depends
from core.security.ai_signature import verify_signature_from_bytes, is_timestamp_valid

from helpers.config import get_settings, Settings 

from core.security.ai_signature import verify_signature_from_bytes, is_timestamp_valid
from helpers import get_settings, Settings

async def verify_backend_signature(
    request: Request,
    learnova_request_id: str = Header(..., alias="Learnova-Request-Id"),
    learnova_timestamp: str = Header(..., alias="Learnova-Timestamp"),
    learnova_signature: str = Header(..., alias="Learnova-Signature"),
    app_settings: Settings = Depends(get_settings)
):
    """
    FastAPI Dependency to verify the HMAC SHA256 signature from the Learnova backend.
    """
    # 1. Validate the timestamp to prevent replay attacks
    if not is_timestamp_valid(learnova_timestamp):
        raise HTTPException(
            status_code=401, 
            detail="Timestamp expired or invalid. Possible replay attack."
        )

    # 2. Extract the raw bytes of the request body
    # We MUST use the raw bytes exactly as they arrived over the network for the hash to match.
    raw_body = await request.body()

    # 3. Verify the signature
    is_valid = verify_signature_from_bytes(
        secret=app_settings.AI_SHARED_SECRET,
        method=request.method,
        path=request.url.path,  # Gets the route like /api/v1/courses/documents
        request_id=learnova_request_id,
        timestamp=learnova_timestamp,
        body=raw_body,
        received_signature=learnova_signature
    )

    if not is_valid:
        raise HTTPException(
            status_code=401, 
            detail="Invalid Learnova signature. Request rejected."
        )

    # If it passes, return the request_id so the endpoint can use it for tracing
    return learnova_request_id