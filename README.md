# ComfyUI GPU tools for architecture

A standalone RunPod Serverless worker with closed graph profiles for video,
architectural images and reference editing, background music and synthetic
narration. The editor and project storage can run separately from the GPU.

| Profile | Model | Output |
| --- | --- | --- |
| video | Wan 2.2 TI2V 5B | MP4, up to 720p and 81 frames |
| video_quality | Wan 2.2 A14B FP8 with LightX2V distilled adapters | MP4 image-to-video |
| image | FLUX.2 Klein 4B FP8, Qwen3 4B encoder, FLUX.2 VAE | PNG, up to one megapixel |
| music | ACE-Step 1.5 Turbo AIO | MP3, up to 90 seconds |
| voice | Qwen3 TTS 1.7B VoiceDesign | MP3, up to 600 input characters |

Set the image's `IABOT_PROFILE` at build time and supply its pinned `models.json`.
Workers accept a native ComfyUI API graph under `input.workflow` and up to four
base64 input images under `input.images`. They never install nodes or download
models from customer workflows. This is a GPU worker: it refuses CPU inference.
VoiceDesign creates synthetic narration and does not accept reference recordings.

Deploy with zero minimum workers, a bounded maximum, a short idle timeout and a
container disk large enough for the selected profile. GPU execution and startup
still have provider costs. No persistent GPU computer is required for each user.
Keep API credentials in your server and persist projects, inputs and results
outside workers. A worker can be replaced without deleting those projects.

Architectural recipes are starting points for concept design and presentation.
Generative image editing and camera motion can change geometry. Check them
against the actual CAD/model before treating them as a technical deliverable.

Model releases and model checksums are pinned in each release branch. Follow the
upstream model licenses and attribution. Code in this repository is a generic
utility: it contains no platform engine, application credentials or customer data.

Run `python -m unittest -q test_profiles` for the graph boundary checks.
