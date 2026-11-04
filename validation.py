"""Closed graph profiles for open models. No remote loaders or user code."""
import os

PROFILE = os.environ.get('IABOT_PROFILE', 'video')
COMMON = {'CLIPTextEncode', 'LoadImage', 'VAEDecode', 'VAEDecodeTiled', 'ImageScale'}
PROFILES = {
    'video': {'nodes': COMMON | {'UNETLoader', 'CLIPLoader', 'VAELoader', 'KSampler', 'ModelSamplingSD3', 'Wan22ImageToVideoLatent', 'CreateVideo', 'SaveVideo'},
              'models': {'UNETLoader': ('unet_name', 'wan2.2_ti2v_5B_fp16.safetensors'), 'CLIPLoader': ('clip_name', 'umt5_xxl_fp8_e4m3fn_scaled.safetensors'), 'VAELoader': ('vae_name', 'wan2.2_vae.safetensors')}, 'output': 'SaveVideo'},
    'image': {'nodes': COMMON | {'UNETLoader', 'CLIPLoader', 'VAELoader', 'SaveImage', 'EmptyFlux2LatentImage', 'Flux2Scheduler', 'CFGGuider', 'ConditioningZeroOut', 'SamplerCustomAdvanced', 'KSamplerSelect', 'RandomNoise', 'ReferenceLatent', 'VAEEncode', 'GetImageSize', 'ImageScaleToTotalPixels'},
              'models': {'UNETLoader': ('unet_name', 'flux-2-klein-4b-fp8.safetensors'), 'CLIPLoader': ('clip_name', 'qwen_3_4b.safetensors'), 'VAELoader': ('vae_name', 'flux2-vae.safetensors')}, 'output': 'SaveImage'},
    'music': {'nodes': {'CheckpointLoaderSimple', 'TextEncodeAceStepAudio1.5', 'ModelSamplingAuraFlow', 'EmptyAceStep1.5LatentAudio', 'KSampler', 'ConditioningZeroOut', 'VAEDecodeAudio', 'SaveAudioAdvanced'},
              'models': {'CheckpointLoaderSimple': ('ckpt_name', 'ace_step_1.5_turbo_aio.safetensors')}, 'output': 'SaveAudioAdvanced'},
    'voice': {'nodes': {'IabotQwenVoice', 'SaveAudioAdvanced'}, 'models': {}, 'output': 'SaveAudioAdvanced'},
}
ALLOWED = PROFILES[PROFILE]['nodes']


def bounded(value, low, high):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and low <= value <= high


def validate(workflow, profile=PROFILE):
    if not isinstance(workflow, dict) or not 1 <= len(workflow) <= 64:
        raise ValueError('Use 1 to 64 nodes')
    spec, counts = PROFILES[profile], {}
    for key, node in workflow.items():
        if not isinstance(key, str) or not key.isdigit() or not isinstance(node, dict):
            raise ValueError('Invalid node identity')
        kind, values = node.get('class_type'), node.get('inputs')
        if kind not in spec['nodes'] or not isinstance(values, dict):
            raise ValueError('Unsupported node: ' + str(kind))
        counts[kind] = counts.get(kind, 0) + 1
        if kind in spec['models']:
            field, model = spec['models'][kind]
            if values.get(field) != model:
                raise ValueError('This profile does not contain that model')
        for name, value in values.items():
            if isinstance(value, list):
                if len(value) != 2 or str(value[0]) not in workflow or not isinstance(value[1], int) or not 0 <= value[1] <= 8:
                    raise ValueError('Invalid node connection')
            elif value is not None and not isinstance(value, (str, int, float, bool)):
                raise ValueError('Invalid node input')
            if isinstance(value, str) and len(value) > 16000:
                raise ValueError('Text input exceeds its limit')
        if kind == 'KSampler' and (not bounded(values.get('steps'), 1, 30) or not bounded(values.get('cfg'), 1, 10)):
            raise ValueError('Sampling limits exceeded')
        if kind in {'EmptyFlux2LatentImage', 'Flux2Scheduler', 'Wan22ImageToVideoLatent', 'ImageScale'}:
            width, height = values.get('width'), values.get('height')
            pixels = 1280 * 720 if profile == 'video' else 1024 * 1024
            if not bounded(width, 64, 2048) or not bounded(height, 64, 2048) or width % 16 or height % 16 or width * height > pixels:
                raise ValueError('Image dimensions exceed the profile limit')
        if kind in {'EmptyFlux2LatentImage', 'Wan22ImageToVideoLatent', 'EmptyAceStep1.5LatentAudio'} and values.get('batch_size') != 1:
            raise ValueError('One output per job')
        if kind == 'Wan22ImageToVideoLatent' and (not bounded(values.get('length'), 5, 81) or (values['length'] - 1) % 4):
            raise ValueError('Use 5 to 81 frames (4n+1)')
        if kind == 'Flux2Scheduler' and not bounded(values.get('steps'), 1, 20):
            raise ValueError('Use 1 to 20 image steps')
        if kind == 'ImageScaleToTotalPixels' and not bounded(values.get('megapixels'), .05, 1):
            raise ValueError('Reference scaling exceeds one megapixel')
        if kind == 'EmptyAceStep1.5LatentAudio' and not bounded(values.get('seconds'), 1, 90):
            raise ValueError('Music is limited to 90 seconds')
        if kind == 'TextEncodeAceStepAudio1.5' and not bounded(values.get('duration'), 1, 90):
            raise ValueError('Music conditioning is limited to 90 seconds')
        if kind == 'IabotQwenVoice':
            if not isinstance(values.get('text'), str) or not 1 <= len(values['text']) <= 600 or not isinstance(values.get('instruct'), str) or len(values['instruct']) > 600:
                raise ValueError('Use at most 600 characters for speech and voice description')
        if kind == 'CreateVideo' and not bounded(values.get('fps'), 1, 30):
            raise ValueError('Unsupported FPS')
        if kind == 'SaveVideo' and (values.get('format') not in {'auto', 'mp4'} or values.get('codec') not in {'auto', 'h264'}):
            raise ValueError('This video profile exports MP4/H.264')
    if counts.get(spec['output']) != 1 or any(counts.get(node, 0) != 1 for node in spec['models']):
        raise ValueError('Use exactly one output and the installed profile models')
    required = {'video': {'KSampler', 'Wan22ImageToVideoLatent', 'CreateVideo'}, 'image': {'SamplerCustomAdvanced', 'EmptyFlux2LatentImage', 'Flux2Scheduler'}, 'music': {'KSampler', 'EmptyAceStep1.5LatentAudio', 'TextEncodeAceStepAudio1.5'}, 'voice': {'IabotQwenVoice'}}[profile]
    if any(counts.get(node, 0) != 1 for node in required):
        raise ValueError('The graph does not match its bounded profile')
    return workflow
