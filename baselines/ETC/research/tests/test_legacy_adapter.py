import unittest

from baselines.ETC.research.legacy_adapter import resolve_max_memory


class LegacyAdapterTests(unittest.TestCase):
    def test_memory_limit_covers_each_visible_gpu(self):
        self.assertEqual(resolve_max_memory(18, 3), {0: "18GiB", 1: "18GiB", 2: "18GiB"})

    def test_memory_limit_requires_gpu_and_positive_value(self):
        with self.assertRaises(ValueError):
            resolve_max_memory(0)
        with self.assertRaises(RuntimeError):
            resolve_max_memory(18, 0)


if __name__ == "__main__":
    unittest.main()
