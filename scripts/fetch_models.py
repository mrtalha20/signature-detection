"""Download trained ONNX weights from your Hugging Face model repo into ./models.

    python scripts/fetch_models.py YOUR_USERNAME/signature-yolo best.onnx
"""
import sys

from huggingface_hub import hf_hub_download

repo, *files = sys.argv[1:] or ["", ""]
if not repo or not files:
    sys.exit(__doc__)
for f in files:
    print(hf_hub_download(repo, f, repo_type="model", local_dir="models"))
