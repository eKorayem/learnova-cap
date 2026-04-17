from pydantic import BaseModel
from typing import Optional


class ProcessRequest(BaseModel):
    file_id: str = None
    chunk_size: Optional[int] = None
    overlap_size: Optional[int] = None
    do_reset: Optional[int] = 0