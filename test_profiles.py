import copy
import json
from pathlib import Path
import unittest
from validation import validate, PROFILES

class Profiles(unittest.TestCase):
    def test_valid_native_profiles(self):
        for profile in PROFILES:
            with self.subTest(profile=profile):validate(json.loads(Path(profile+'-test.json').read_text()),profile)

    def test_profiles_are_isolated(self):
        for profile in PROFILES:
            for other in PROFILES:
                if profile==other:continue
                with self.subTest(profile=profile,other=other):
                    with self.assertRaises(ValueError):validate(json.loads(Path(other+'-test.json').read_text()),profile)

    def test_model_substitution_and_expensive_jobs_are_rejected(self):
        image=json.loads(Path('image-test.json').read_text());image['1']['inputs']['unet_name']='remote.safetensors'
        with self.assertRaises(ValueError):validate(image,'image')
        image=json.loads(Path('image-test.json').read_text());image['10']['inputs']['width']=4096
        with self.assertRaises(ValueError):validate(image,'image')
        music=json.loads(Path('music-test.json').read_text());music['5']['inputs']['seconds']=901
        with self.assertRaises(ValueError):validate(music,'music')
        voice=json.loads(Path('voice-test.json').read_text());voice['1']['inputs']['text']='A'*601
        with self.assertRaises(ValueError):validate(voice,'voice')
        voice=json.loads(Path('voice-test.json').read_text());voice['1']['class_type']='VoiceClone'
        with self.assertRaises(ValueError):validate(voice,'voice')

    def test_second_output_or_unknown_inputs_are_not_executable(self):
        voice=json.loads(Path('voice-test.json').read_text());voice['3']=copy.deepcopy(voice['2'])
        with self.assertRaises(ValueError):validate(voice,'voice')
        image=json.loads(Path('image-test.json').read_text());image['14']={'class_type':'RunShell','inputs':{}}
        with self.assertRaises(ValueError):validate(image,'image')

if __name__=='__main__':unittest.main()
