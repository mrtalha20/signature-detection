"""boto3 clients + idempotent creation of the bucket and queues."""
import json
import logging

import boto3
from botocore.exceptions import ClientError

from . import config

log = logging.getLogger(__name__)


def _client(service: str):
    return boto3.client(
        service,
        region_name=config.AWS_REGION,
        endpoint_url=config.AWS_ENDPOINT_URL,
    )


def s3():
    return _client("s3")


def sqs():
    return _client("sqs")


def ensure_infra() -> str:
    """Create bucket, DLQ and main queue if missing. Returns the main queue URL.

    Safe to call on every start. In AWS prefer provisioning with IaC and set
    AUTO_CREATE_INFRA=false (then this only looks the queue up).
    """
    sqs_client = sqs()

    if config.AUTO_CREATE_INFRA:
        s3_client = s3()
        try:
            s3_client.head_bucket(Bucket=config.S3_BUCKET)
        except ClientError:
            kwargs = {"Bucket": config.S3_BUCKET}
            # us-east-1 must NOT send a LocationConstraint
            if config.AWS_REGION != "us-east-1":
                kwargs["CreateBucketConfiguration"] = {
                    "LocationConstraint": config.AWS_REGION
                }
            s3_client.create_bucket(**kwargs)
            log.info("created bucket %s", config.S3_BUCKET)

        dlq_url = sqs_client.create_queue(QueueName=config.SQS_DLQ_NAME)["QueueUrl"]
        dlq_arn = sqs_client.get_queue_attributes(
            QueueUrl=dlq_url, AttributeNames=["QueueArn"]
        )["Attributes"]["QueueArn"]
        sqs_client.create_queue(
            QueueName=config.SQS_QUEUE_NAME,
            Attributes={
                "VisibilityTimeout": str(config.SQS_VISIBILITY_TIMEOUT),
                "RedrivePolicy": json.dumps(
                    {
                        "deadLetterTargetArn": dlq_arn,
                        "maxReceiveCount": config.SQS_MAX_RECEIVE_COUNT,
                    }
                ),
            },
        )

    return sqs_client.get_queue_url(QueueName=config.SQS_QUEUE_NAME)["QueueUrl"]
