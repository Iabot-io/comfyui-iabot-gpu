"""ComfyUI GPU jobs for RunPod Serverless. Customer persistence stays outside workers."""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import threading
import time
import uuid

import requests
import runpod
import torch

from validation import ALLOWED, validate

BASE = "http://127.0.0.1:8188"
MAX_OUTPUT = 6 * 1024 * 1024
LOCK = threading.Lock()


def request(path, body=None):
    response = requests.post(BASE + path, json=body, timeout=30) if body is not None else requests.get(BASE + path, timeout=30)
    if response.status_code != 200:
        raise ValueError("ComfyUI rejected the graph: " + response.text[:500])
    return response.json()


def gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("A real NVIDIA GPU is required")
    return {"cuda": True, "device": torch.cuda.get_device_name(0)}


def execute(job):
    info = gpu()
    payload = job.get("input", {})
    if payload.get("operation") == "catalog":
        catalog = request("/object_info")
        return {"catalog": {key: value for key, value in catalog.items() if key in ALLOWED}, "gpu": info}
    workflow = copy.deepcopy(validate(payload.get("workflow")))
    unique = str(uuid.uuid4())
    input_dir = Path("/comfyui/input/iabot") / unique
    output_dir = Path("/comfyui/output/iabot") / unique
    input_dir.mkdir(parents=True, exist_ok=True)
    files = payload.get("images", [])
    if not isinstance(files, list) or len(files) > 4:
        raise ValueError("At most four input images")
    mapping, total = {}, 0
    try:
        from PIL import Image
        for index, item in enumerate(files):
            name = item.get("name", "")
            if not isinstance(name, str) or Path(name).name != name or len(name) > 120:
                raise ValueError("Invalid image filename")
            raw = base64.b64decode(item.get("data", ""), validate=True)
            total += len(raw)
            if total > 6 * 1024 * 1024:
                raise ValueError("Input images exceed 6 MiB")
            image = Image.open(io.BytesIO(raw))
            if image.width * image.height > 16_000_000:
                raise ValueError("Input image dimensions exceed the limit")
            target = input_dir / (str(index) + ".png")
            image.convert("RGB").save(target)
            mapping[name] = "iabot/" + unique + "/" + target.name
        for node in workflow.values():
            if node["class_type"] == "LoadImage":
                wanted = node["inputs"].get("image")
                if wanted not in mapping:
                    raise ValueError("An input image is missing")
                node["inputs"]["image"] = mapping[wanted]
            if node["class_type"] == "SaveVideo":
                node["inputs"]["filename_prefix"] = "iabot/" + unique + "/video"
                node["inputs"]["format"], node["inputs"]["codec"] = "mp4", "h264"
        accepted = request("/prompt", {"prompt": workflow, "client_id": unique})
        prompt_id = accepted.get("prompt_id")
        if not prompt_id:
            raise RuntimeError("ComfyUI did not accept the graph")
        deadline = time.monotonic() + 1500
        while time.monotonic() < deadline:
            history = request("/history/" + prompt_id).get(prompt_id)
            if history and (history.get("status", {}).get("completed") or history.get("status", {}).get("status_str") == "error"):
                break
            time.sleep(0.5)
        else:
            requests.post(BASE + "/interrupt", timeout=10)
            raise RuntimeError("GPU execution exceeded its deadline")
        if history["status"].get("status_str") != "success":
            raise RuntimeError("ComfyUI could not finish the graph: " + json.dumps(history["status"].get("messages", []))[-1200:])
        videos = list(output_dir.rglob("*.mp4")) if output_dir.exists() else []
        if len(videos) != 1 or videos[0].stat().st_size > MAX_OUTPUT:
            raise RuntimeError("Expected one MP4 no larger than 6 MiB")
        result = videos[0].read_bytes()
        if result[4:8] != b"ftyp":
            raise RuntimeError("The video is not an MP4")
        devices = request("/system_stats").get("devices", [])
        if not any(device.get("type") == "cuda" for device in devices):
            raise RuntimeError("ComfyUI did not report its CUDA device")
        return {"video": base64.b64encode(result).decode(), "bytes": len(result),
                "sha256": hashlib.sha256(result).hexdigest(), "gpu": info,
                "format": "mp4", "promptId": prompt_id, "comfyDevices": devices}
    finally:
        shutil.rmtree(input_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)


def handler(job):
    with LOCK:
        try:
            return execute(job)
        except Exception as error:
            return {"error": str(error)[:1500]}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
