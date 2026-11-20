"""Tiny Postgres layer (psycopg2, no ORM) for job state."""
import json
import logging
import time
from contextlib import contextmanager

import psycopg2
import psycopg2.extras

from . import config

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          UUID PRIMARY KEY,
    type        TEXT NOT NULL,
    status      TEXT NOT NULL,            -- queued | processing | done | failed
    image_keys  JSONB NOT NULL,
    result      JSONB,
    error       TEXT,
    attempts    INT NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs (status);
"""


@contextmanager
def conn():
    c = psycopg2.connect(config.DATABASE_URL)
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


def init_db(retries: int = 30, delay: float = 2.0) -> None:
    """Create tables, waiting for Postgres to come up (docker start order)."""
    for attempt in range(1, retries + 1):
        try:
            with conn() as c, c.cursor() as cur:
                cur.execute(SCHEMA)
            return
        except psycopg2.OperationalError as e:
            log.warning("db not ready (%s/%s): %s", attempt, retries, e)
            time.sleep(delay)
    raise RuntimeError("database never became ready")


def create_job(job_id: str, job_type: str, image_keys: list[str]) -> None:
    with conn() as c, c.cursor() as cur:
        cur.execute(
            "INSERT INTO jobs (id, type, status, image_keys) VALUES (%s,%s,'queued',%s)",
            (job_id, job_type, json.dumps(image_keys)),
        )


def get_job(job_id: str) -> dict | None:
    with conn() as c, c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM jobs WHERE id = %s", (job_id,))
        row = cur.fetchone()
    if row:
        row["id"] = str(row["id"])
    return row


def mark_processing(job_id: str, attempts: int) -> None:
    with conn() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET status='processing', attempts=%s, updated_at=now() WHERE id=%s",
            (attempts, job_id),
        )


def mark_done(job_id: str, result: dict) -> None:
    with conn() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET status='done', result=%s, error=NULL, updated_at=now() WHERE id=%s",
            (json.dumps(result), job_id),
        )


def mark_failed(job_id: str, error: str, final: bool) -> None:
    """final=False keeps status 'queued' because SQS will redeliver the message."""
    status = "failed" if final else "queued"
    with conn() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET status=%s, error=%s, updated_at=now() WHERE id=%s",
            (status, error[:2000], job_id),
        )


def ping() -> bool:
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT 1")
        return cur.fetchone() == (1,)
