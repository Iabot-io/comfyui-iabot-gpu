import copy
import json
from pathlib import Path
import unittest

from validation import validate


class GraphBoundaries(unittest.TestCase):
    def setUp(self):
        local = Path(__file__).parent
        fixture = local / 'test-workflow.json'
        if not fixture.exists():
            fixture = local / 'templates/03_Prueba_GPU_corta_api.json'
        self.graph = json.loads(fixture.read_text())

    def test_real_template_is_accepted(self):
        self.assertIs(validate(self.graph), self.graph)

    def test_unknown_execution_nodes_are_rejected(self):
        self.graph['37']['class_type'] = 'ExecutePython'
        with self.assertRaises(ValueError):
            validate(self.graph)

    def test_uninstalled_models_are_rejected(self):
        self.graph['37']['inputs']['unet_name'] = '../../other-company/model.safetensors'
        with self.assertRaises(ValueError):
            validate(self.graph)

    def test_resource_expansion_is_rejected(self):
        for field, value in [('length', 1001), ('width', 4096), ('batch_size', 8)]:
            with self.subTest(field=field):
                graph = copy.deepcopy(self.graph)
                graph['55']['inputs'][field] = value
                with self.assertRaises(ValueError):
                    validate(graph)

    def test_broken_connections_are_rejected(self):
        self.graph['3']['inputs']['model'] = ['not-a-node', 0]
        with self.assertRaises(ValueError):
            validate(self.graph)

    def test_non_video_graph_cannot_claim_video_generation(self):
        del self.graph['58']
        with self.assertRaises(ValueError):
            validate(self.graph)


if __name__ == '__main__':
    unittest.main()
