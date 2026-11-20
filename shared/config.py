"""Central configuration. Everything comes from environment variables so the
same image runs locally (LocalStack) and on AWS (real SQS/S3) unchanged."""
import os

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
# Set to http://localstack:4566 locally. Leave empty on AWS.
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL") or None

S3_BUCKET = os.getenv("S3_BUCKET", "signature-uploads")
SQS_QUEUE_NAME = os.getenv("SQS_QUEUE_NAME", "signature-jobs")
SQS_DLQ_NAME = os.getenv("SQS_DLQ_NAME", "signature-jobs-dlq")
SQS_MAX_RECEIVE_COUNT = int(os.getenv("SQS_MAX_RECEIVE_COUNT", "3"))
SQS_VISIBILITY_TIMEOUT = int(os.getenv("SQS_VISIBILITY_TIMEOUT", "120"))

# On AWS, Terraform/CloudFormation creates the infra, so set this to "false".
AUTO_CREATE_INFRA = os.getenv("AUTO_CREATE_INFRA", "true").lower() == "true"

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://sig:sig@postgres:5432/signatures"
)

MODEL_DIR = os.getenv("MODEL_DIR", "/models")
DETECTOR_PATH = os.path.join(MODEL_DIR, os.getenv("DETECTOR_FILE", "best.onnx"))
EMBEDDER_PATH = os.path.join(MODEL_DIR, os.getenv("EMBEDDER_FILE", "embedder.onnx"))

DETECT_CONF = float(os.getenv("DETECT_CONF", "0.25"))
DETECT_IOU = float(os.getenv("DETECT_IOU", "0.5"))
# Cosine similarity at/above this is reported as "match". Calibrate on validation pairs.
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.80"))

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
