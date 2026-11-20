"""FastAPI service: accepts images, stores them in S3, enqueues a job in SQS.

It never runs the models itself. The worker does, so the API stays fast and
can scale independently of the (heavier) inference workers.
"""
import json
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from shared import aws, config, db
from shared.schemas import JobCreated, JobStatus

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

ALLOWED = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    state["queue_url"] = aws.ensure_infra()
    state["s3"] = aws.s3()
    state["sqs"] = aws.sqs()
    log.info("api ready, queue=%s", state["queue_url"])
    yield


app = FastAPI(title="Signature Detection & Similarity API", version="1.0.0", lifespan=lifespan)


async def _store(job_id: str, name: str, f: UploadFile) -> str:
    ext = ALLOWED.get(f.content_type or "")
    if ext is None:
        raise HTTPException(415, f"{name}: only JPEG/PNG/WEBP images are accepted")
    data = await f.read()
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"{name}: larger than {config.MAX_UPLOAD_MB} MB")
    if not data:
        raise HTTPException(400, f"{name}: empty file")
    key = f"jobs/{job_id}/{name}.{ext}"
    state["s3"].put_object(Bucket=config.S3_BUCKET, Key=key, Body=data, ContentType=f.content_type)
    return key


@app.post("/v1/jobs", response_model=JobCreated, status_code=202)
async def create_job(
    image_a: UploadFile = File(..., description="Document or signature image"),
    image_b: UploadFile | None = File(None, description="Second image -> similarity comparison"),
):
    """image_a only  -> type 'detect'  (find signatures)
    image_a + image_b -> type 'compare' (detect in both, then similarity score)"""
    job_id = str(uuid.uuid4())
    keys = [await _store(job_id, "a", image_a)]
    if image_b is not None:
        keys.append(await _store(job_id, "b", image_b))
    job_type = "compare" if len(keys) == 2 else "detect"

    db.create_job(job_id, job_type, keys)
    state["sqs"].send_message(
        QueueUrl=state["queue_url"],
        MessageBody=json.dumps({"job_id": job_id, "type": job_type, "image_keys": keys}),
    )
    log.info("enqueued %s (%s)", job_id, job_type)
    return JobCreated(job_id=job_id, type=job_type)


@app.get("/v1/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(400, "job_id must be a UUID")
    row = db.get_job(job_id)
    if row is None:
        raise HTTPException(404, "job not found")
    return JobStatus(
        job_id=row["id"],
        type=row["type"],
        status=row["status"],
        attempts=row["attempts"],
        result=row["result"],
        error=row["error"],
    )


@app.get("/health")
def health():
    """Liveness: process is up (used by the load balancer)."""
    return {"status": "ok"}


@app.get("/health/ready")
def ready():
    """Readiness: dependencies reachable."""
    try:
        db.ping()
        state["sqs"].get_queue_attributes(QueueUrl=state["queue_url"], AttributeNames=["QueueArn"])
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"status": "unready", "error": str(e)}, status_code=503)
    return {"status": "ready"}
