"""Bounded, native ComfyUI video graphs. No user code, remote loaders or installers."""
MODELS = {
    "UNETLoader": ("unet_name", "wan2.2_ti2v_5B_fp16.safetensors"),
    "CLIPLoader": ("clip_name", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
    "VAELoader": ("vae_name", "wan2.2_vae.safetensors"),
}
ALLOWED = set(MODELS) | {
    "CLIPTextEncode", "ModelSamplingSD3", "Wan22ImageToVideoLatent", "KSampler",
    "VAEDecode", "VAEDecodeTiled", "CreateVideo", "SaveVideo", "LoadImage",
    "ImageScale", "ImageScaleBy", "ImageCrop", "ImageBatch", "PreviewImage",
    "ConditioningCombine", "ConditioningConcat", "ConditioningSetArea",
}


def validate(workflow):
    if not isinstance(workflow, dict) or not 1 <= len(workflow) <= 64:
        raise ValueError("The workflow must contain 1 to 64 nodes")
    classes = set()
    for key, node in workflow.items():
        if not isinstance(key, str) or not key.isdigit() or not isinstance(node, dict):
            raise ValueError("Invalid node identity")
        name, values = node.get("class_type"), node.get("inputs")
        if name not in ALLOWED or not isinstance(values, dict):
            raise ValueError("Unsupported node: " + str(name))
        classes.add(name)
        if name in MODELS:
            field, model = MODELS[name]
            if values.get(field) != model:
                raise ValueError("This worker does not contain that model")
        if name == "CLIPTextEncode" and (not isinstance(values.get("text"), str) or len(values["text"]) > 16000):
            raise ValueError("Invalid text input")
        if name == "KSampler":
            if not 1 <= int(values.get("steps", 0)) <= 30 or not 1 <= float(values.get("cfg", 0)) <= 10:
                raise ValueError("Sampling limits exceeded")
        if name in {"Wan22ImageToVideoLatent", "ImageScale"}:
            width, height = int(values.get("width", 0)), int(values.get("height", 0))
            if min(width, height) < 64 or width * height > 1280 * 720 or width % 16 or height % 16:
                raise ValueError("Use dimensions divisible by 16, up to 720p")
        if name == "Wan22ImageToVideoLatent":
            length = int(values.get("length", 0))
            if not 5 <= length <= 81 or (length - 1) % 4 or values.get("batch_size", 1) != 1:
                raise ValueError("Use 5 to 81 frames (4n+1), one video per job")
        if name == "CreateVideo" and not 1 <= float(values.get("fps", 0)) <= 30:
            raise ValueError("Unsupported FPS")
        if name == "SaveVideo" and (values.get("format") not in {"auto", "mp4"} or values.get("codec") not in {"auto", "h264"}):
            raise ValueError("This endpoint exports MP4/H.264")
        for value in values.values():
            if isinstance(value, list) and (len(value) != 2 or str(value[0]) not in workflow or not isinstance(value[1], int)):
                raise ValueError("Invalid node connection")
    if not {"UNETLoader", "KSampler", "Wan22ImageToVideoLatent", "CreateVideo", "SaveVideo"} <= classes:
        raise ValueError("The graph must generate and save a video with the installed Wan model")
    return workflow
