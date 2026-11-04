# ComfyUI GPU worker for iabot architecture tools

A standalone RunPod Serverless worker for bounded Wan 2.2 5B video workflows. It runs the original ComfyUI server on NVIDIA CUDA and accepts API-format workflow JSON. The companion templates cover text to video, image to video and a short GPU smoke test.

Workers are disposable. The calling platform must retain editable workflows, inputs, job identities and final outputs in durable storage. This repository contains no platform engine, account credentials or customer projects.

## Build and run

Build the Dockerfile on an amd64 builder with enough space for the public model weights. Source and model downloads are verified against pinned SHA-256 hashes. The image runs the graph-boundary tests and ComfyUI CPU import smoke test during its build. Run it on a queue-based RunPod Serverless endpoint with active workers set to zero.

Send `input.workflow` in ComfyUI API format. Optional `input.images` entries contain `name` and base64 `data`. The response contains the base64 MP4, its byte count, SHA-256 and CUDA device evidence. `input.operation=catalog` returns the supported native node catalogue.

The initial profile accepts a single video, up to 81 frames, dimensions divisible by 16 up to 1280×720, 30 sampling steps and a 6 MiB MP4. Unsupported models, arbitrary execution nodes and remote loaders are rejected. The calling platform must validate identity, quota, billing and permissions before submitting.

## Upstream sources

- [ComfyUI v0.34.0](https://github.com/Comfy-Org/ComfyUI/tree/v0.34.0) (GPL-3.0)
- [Official Wan 2.2 workflows](https://docs.comfy.org/tutorials/video/wan/wan2_2)
- [Comfy-Org model distribution](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged)
- [RunPod Serverless](https://docs.runpod.io/serverless/overview)

Retain the upstream licences for ComfyUI, the runtime libraries and the included model distributions.
