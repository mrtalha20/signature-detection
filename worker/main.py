"""Worker: long-polls SQS, runs detection (+ similarity), writes results to Postgres.

Delivery semantics (at-least-once):
  - success            -> delete message
  - exception          -> leave message; it reappears after the visibility timeout
  - 3rd failed receive -> mark job failed; SQS redrive policy moves it to the DLQ
  - redelivery of a job that is already 'done' is acknowledged and skipped
"""
import json
import logging
import signal
import time

from shared import aws, config, db

from .inference import Detector, Embedder, cosine, crop_box, decode_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("worker")

_running = True


def _stop(*_):
    global _running
    log.info("shutdown requested, finishing current message")
    _running = False


def process(body: dict, s3, detector: Detector, embedder: Embedder) -> dict:
    images = []
    for key in body["image_keys"]:
        data = s3.get_object(Bucket=config.S3_BUCKET, Key=key)["Body"].read()
        images.append(decode_image(data))

    detections = [detector.detect(img) for img in images]
    result = {"detections": detections}

    if body["type"] == "compare":
        embs = []
        for img, dets in zip(images, detections):
            if not dets:
                result.update(similarity=None, match=None,
                              note="no signature detected in one of the images")
                return result
            embs.append(embedder.embed(crop_box(img, dets[0])))  # best box per image
        cos = cosine(embs[0], embs[1])
        result.update(
            similarity=round(cos, 4),
            similarity_percent=round(max(0.0, cos) * 100, 1),
            match=cos >= config.MATCH_THRESHOLD,
            threshold=config.MATCH_THRESHOLD,
        )
    return result


def main() -> None:
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    db.init_db()
    queue_url = aws.ensure_infra()
    s3, sqs = aws.s3(), aws.sqs()
    detector, embedder = Detector(), Embedder()
    log.info("worker started, polling %s", queue_url)

    while _running:
        resp = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,  # long polling
            AttributeNames=["ApproximateReceiveCount"],
        )
        for msg in resp.get("Messages", []):
            receipt = msg["ReceiptHandle"]
            receive_count = int(msg["Attributes"]["ApproximateReceiveCount"])
            try:
                body = json.loads(msg["Body"])
                job_id = body["job_id"]
            except (ValueError, KeyError):
                log.error("malformed message, dropping: %r", msg["Body"][:200])
                sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
                continue

            try:
                job = db.get_job(job_id)
                if job and job["status"] == "done":
                    log.info("job %s already done, acking duplicate", job_id)
                    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
                    continue

                db.mark_processing(job_id, receive_count)
                t0 = time.time()
                result = process(body, s3, detector, embedder)
                result["processing_ms"] = int((time.time() - t0) * 1000)
                db.mark_done(job_id, result)
                sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
                log.info("job %s done in %d ms", job_id, result["processing_ms"])
            except Exception as e:  # noqa: BLE001
                final = receive_count >= config.SQS_MAX_RECEIVE_COUNT
                log.exception("job %s failed (attempt %d/%d)", job_id, receive_count,
                              config.SQS_MAX_RECEIVE_COUNT)
                try:
                    db.mark_failed(job_id, repr(e), final=final)
                except Exception:  # noqa: BLE001
                    log.exception("could not record failure for %s", job_id)
                # message is NOT deleted: SQS redelivers, then moves it to the DLQ


if __name__ == "__main__":
    main()
