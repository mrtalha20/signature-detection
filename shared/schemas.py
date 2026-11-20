from typing import Literal, Optional

from pydantic import BaseModel


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class JobCreated(BaseModel):
    job_id: str
    status: Literal["queued"] = "queued"
    type: Literal["detect", "compare"]


class JobStatus(BaseModel):
    job_id: str
    type: str
    status: Literal["queued", "processing", "done", "failed"]
    attempts: int
    result: Optional[dict] = None
    error: Optional[str] = None
