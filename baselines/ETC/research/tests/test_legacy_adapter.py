import unittest
from types import SimpleNamespace

from baselines.ETC.research.legacy_adapter import install_last_layer_attention_capture, resolve_max_memory


class FakeLayer:
    def forward(self, hidden, output_attentions=False, **kwargs):
        return (hidden, "last-attention") if output_attentions else (hidden,)


class FakeInnerModel:
    def __init__(self, layer):
        self.layers = [layer]


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(_attn_implementation="sdpa")
        self.layer = FakeLayer()
        self.model = FakeInnerModel(self.layer)

    def forward(self, hidden, output_attentions=False, **kwargs):
        layer_output = self.layer.forward(hidden, output_attentions=output_attentions)
        attentions = (layer_output[1],) if output_attentions else None
        return SimpleNamespace(attentions=attentions, hidden=layer_output[0])


class LegacyAdapterTests(unittest.TestCase):
    def test_only_last_attention_is_attached_and_config_is_restored(self):
        model = FakeModel()
        install_last_layer_attention_capture(model)
        outputs = model.forward("hidden", output_attentions=True)
        self.assertEqual(outputs.attentions, ("last-attention",))
        self.assertEqual(model.config._attn_implementation, "sdpa")
        plain = model.forward("hidden", output_attentions=False)
        self.assertIsNone(plain.attentions)

    def test_memory_limit_covers_each_visible_gpu(self):
        self.assertEqual(resolve_max_memory(18, 3), {0: "18GiB", 1: "18GiB", 2: "18GiB"})

    def test_memory_limit_requires_gpu_and_positive_value(self):
        with self.assertRaises(ValueError):
            resolve_max_memory(0)
        with self.assertRaises(RuntimeError):
            resolve_max_memory(18, 0)


if __name__ == "__main__":
    unittest.main()
