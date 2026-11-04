"""A synthetic voice node. No reference recordings or identity cloning."""
import numpy as np
import torch

_model = None


class IabotQwenVoice:
    CATEGORY = 'iabot/Arquitectura'
    RETURN_TYPES = ('AUDIO',)
    FUNCTION = 'generate'

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'text': ('STRING', {'multiline': True, 'default': 'Diseñamos espacios para vivir mejor.'}),
            'language': (['Spanish', 'English', 'French', 'German', 'Italian', 'Portuguese'],),
            'instruct': ('STRING', {'multiline': True, 'default': 'Voz cálida y natural en español, dicción clara y ritmo tranquilo para presentar un proyecto de arquitectura.'}),
        }}

    def generate(self, text, language, instruct):
        global _model
        if not torch.cuda.is_available() or not 1 <= len(text) <= 600 or len(instruct) > 600:
            raise ValueError('GPU speech requires bounded text and a synthetic voice description')
        if _model is None:
            from qwen_tts import Qwen3TTSModel
            _model = Qwen3TTSModel.from_pretrained('/comfyui/models/iabot_voice', device_map='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa')
        with torch.inference_mode():
            waves, rate = _model.generate_voice_design(text=text, language=language, instruct=instruct, max_new_tokens=2048)
        wave = np.asarray(waves[0], dtype=np.float32)
        if wave.ndim != 1 or len(wave) > 90 * rate:
            raise ValueError('Speech output exceeds 90 seconds')
        return ({'waveform': torch.from_numpy(wave)[None, None, :], 'sample_rate': rate},)


NODE_CLASS_MAPPINGS = {'IabotQwenVoice': IabotQwenVoice}
NODE_DISPLAY_NAME_MAPPINGS = {'IabotQwenVoice': 'Locución · Qwen3 TTS · GPU'}
