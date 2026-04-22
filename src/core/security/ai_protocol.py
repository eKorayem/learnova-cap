import json
from typing import Any


# =========================================
# Header Names
# =========================================

AI_REQUEST_ID_HEADER = "Learnova-Request-Id"
AI_TIMESTAMP_HEADER = "Learnova-Timestamp"
AI_SIGNATURE_HEADER = "Learnova-Signature"


# =========================================
# Default Protocol Constants
# =========================================

DEFAULT_HTTP_METHOD = "POST"
DEFAULT_ALLOWED_TIMESTAMP_DRIFT_SECONDS = 300
DEFAULT_AI_REQUEST_TIMEOUT_SECONDS = 120


# =========================================
# JSON Serialization
# =========================================

def serialize_json_for_signing(data: dict[str, Any]) -> bytes:
    """
    Serialize JSON deterministically for signing and transport.

    Rules:
    - UTF-8 encoding
    - no unnecessary whitespace
    - stable key ordering
    - non-ASCII characters preserved as-is
    """
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


# =========================================
# Outbound Envelope Builder
# =========================================

def build_ai_request_envelope(
    *,
    request_id: str,
    timestamp: str,
    operation_type: str,
    course_id: int | None,
    body: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the standard outbound AI request envelope.

    Shared transport-level metadata stays at root level.
    Feature-specific payload always goes inside 'body'.
    """
    return {
        "request_id": request_id,
        "timestamp": timestamp,
        "operation_type": operation_type,
        "course_id": course_id,
        "body": body,
    }


# =========================================
# Header Builder
# =========================================

def build_ai_request_headers(
    *,
    request_id: str,
    timestamp: str,
    signature: str,
) -> dict[str, str]:
    """
    Build the standard headers used in backend <-> AI service communication.
    """
    return {
        AI_REQUEST_ID_HEADER: request_id,
        AI_TIMESTAMP_HEADER: timestamp,
        AI_SIGNATURE_HEADER: signature,
        "Content-Type": "application/json",
    }


# =========================================
# Header Extraction Helpers
# =========================================

def extract_protocol_headers(headers: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    """
    Extract standard AI integration headers from a generic headers mapping.
    """
    request_id = headers.get(AI_REQUEST_ID_HEADER)
    timestamp = headers.get(AI_TIMESTAMP_HEADER)
    signature = headers.get(AI_SIGNATURE_HEADER)

    return request_id, timestamp, signature