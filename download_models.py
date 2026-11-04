"""Build-time public model download; pinned sizes and SHA-256 checks."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request


def download(model):
    path = Path("/comfyui/models") / model["folder"] / model["name"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(model["url"], timeout=120) as response, path.open("wb") as stream:
        shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if path.stat().st_size != model["bytes"] or digest.hexdigest() != model["sha256"]:
        raise RuntimeError("Model integrity mismatch: " + model["name"])
    print("Verified:", model["name"], flush=True)


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, json.loads(Path("models.json").read_text())))
