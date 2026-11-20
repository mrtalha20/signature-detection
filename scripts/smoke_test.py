"""End-to-end check against a running stack.

    python scripts/smoke_test.py a.png [b.png]
"""
import sys
import time

import requests

API = "http://localhost:8000"
paths = sys.argv[1:]
if not paths:
    sys.exit(__doc__)

files = {"image_a": open(paths[0], "rb")}
if len(paths) > 1:
    files["image_b"] = open(paths[1], "rb")

r = requests.post(f"{API}/v1/jobs", files=files)
r.raise_for_status()
job = r.json()
print("submitted:", job)

for _ in range(60):
    s = requests.get(f"{API}/v1/jobs/{job['job_id']}").json()
    print(" status:", s["status"])
    if s["status"] in ("done", "failed"):
        print(s)
        break
    time.sleep(1)
