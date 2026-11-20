# Signature Detection & Similarity System

```
Client -> FastAPI (api) -> S3 (images) + Postgres (job row) + SQS queue
                                                  |
                                    Worker(s): long-poll SQS
                                    YOLO ONNX detect -> crop -> embed -> cosine
                                                  |
                                       result -> Postgres (GET /v1/jobs/{id})
SQS redrive: 3 failed receives -> dead-letter queue
```

## Run locally

```bash
docker compose up --build            # api :8000, localstack :4566, postgres
python scripts/smoke_test.py sample.png other.png
docker compose up --scale worker=3   # show parallel workers
```

Interactive docs: http://localhost:8000/docs

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/jobs` | multipart `image_a` (+ optional `image_b`). Returns `202 {job_id}` |
| GET | `/v1/jobs/{id}` | `queued / processing / done / failed` + result |
| GET | `/health`, `/health/ready` | liveness / readiness |

`image_a` only -> **detect**. `image_a` + `image_b` -> **compare** (similarity score).

## Models

Put ONNX files in `./models/` (mounted into the worker):

- `best.onnx` : YOLO detector exported from Colab
- `embedder.onnx` : similarity embedding model (optional for now)

Without them the worker still runs: whole-image fallback box and a **placeholder**
pixel embedding, so you can test the queue/API plumbing before training finishes.
Do not report similarity numbers from the placeholder.

Pull weights from Hugging Face: `python scripts/fetch_models.py USER/signature-yolo best.onnx`

## Deploy to AWS (target architecture)

Push `Dockerfile.api` / `Dockerfile.worker` images to ECR, run both on ECS Fargate
(API behind an ALB, workers scaled on SQS queue depth), S3 bucket for uploads,
SQS + DLQ, RDS Postgres. Set `AUTO_CREATE_INFRA=false`, drop `AWS_ENDPOINT_URL` and
the dummy keys, and give the task roles s3/sqs permissions.

