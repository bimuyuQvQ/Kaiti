import unittest

from baselines.ETC.research.collect_rollouts import ConfigNamespace


class CollectRolloutTests(unittest.TestCase):
    def test_config_namespace_supports_both_legacy_access_styles(self):
        config = ConfigNamespace({"es_index_name": "wiki", "sample": 1})
        self.assertEqual(config.es_index_name, "wiki")
        self.assertIn("es_index_name", config)
        self.assertNotIn("missing", config)


if __name__ == "__main__":
    unittest.main()
