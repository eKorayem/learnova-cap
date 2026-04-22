import hashlib
import hmac
from datetime import datetime, timezone

from core.security.ai_protocol import serialize_json_for_signing


# =========================================
# Timestamp Helpers
# =========================================

def get_current_timestamp() -> str:
    """
    Return current UTC timestamp in ISO 8601 format.
    Example: 2026-04-14T18:30:00Z
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_timestamp_valid(
    timestamp: str,
    allowed_drift_seconds: int = 300,
) -> bool:
    """
    Validate timestamp against current UTC time.
    """
    try:
        request_time = datetime.strptime(
            timestamp,
            "%Y-%m-%dT%H:%M:%SZ",
        ).replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        diff_seconds = abs((now - request_time).total_seconds())

        return diff_seconds <= allowed_drift_seconds
    except Exception:
        return False


# =========================================
# Body Hashing
# =========================================

def hash_body_bytes(body: bytes) -> str:
    """
    Return SHA256 hash of raw request body bytes.
    """
    return hashlib.sha256(body).hexdigest()


# =========================================
# Canonical String Builder
# =========================================

def build_canonical_string(
    *,
    method: str,
    path: str,
    request_id: str,
    timestamp: str,
    body_hash: str,
) -> str:
    """
    Build canonical string used for HMAC signing.
    """
    return "\n".join(
        [
            method.upper(),
            path,
            request_id,
            timestamp,
            body_hash,
        ]
    )


# =========================================
# Signature Creation
# =========================================

def create_signature_from_bytes(
    *,
    secret: str,
    method: str,
    path: str,
    request_id: str,
    timestamp: str,
    body: bytes,
) -> str:
    """
    Create HMAC SHA256 signature from raw request body bytes.
    """
    body_hash = hash_body_bytes(body)

    canonical_string = build_canonical_string(
        method=method,
        path=path,
        request_id=request_id,
        timestamp=timestamp,
        body_hash=body_hash,
    )

    return hmac.new(
        key=secret.encode("utf-8"),
        msg=canonical_string.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()


def create_signature_from_json(
    *,
    secret: str,
    method: str,
    path: str,
    request_id: str,
    timestamp: str,
    body: dict,
) -> str:
    """
    Serialize JSON deterministically, then create signature.
    """
    serialized_body = serialize_json_for_signing(body)

    return create_signature_from_bytes(
        secret=secret,
        method=method,
        path=path,
        request_id=request_id,
        timestamp=timestamp,
        body=serialized_body,
    )


# =========================================
# Signature Verification
# =========================================

def verify_signature_from_bytes(
    *,
    secret: str,
    method: str,
    path: str,
    request_id: str,
    timestamp: str,
    body: bytes,
    received_signature: str,
) -> bool:
    """
    Verify signature using raw request body bytes.
    """
    expected_signature = create_signature_from_bytes(
        secret=secret,
        method=method,
        path=path,
        request_id=request_id,
        timestamp=timestamp,
        body=body,
    )

    return hmac.compare_digest(expected_signature, received_signature)