import tempfile
import unittest
from pathlib import Path

from baselines.ETC.research.manifest import build_manifest, sha256_file
from baselines.ETC.research.schema import stable_id
from baselines.ETC.research.splits import assign_splits, split_for_qid, validate_disjoint


class SchemaSplitManifestTests(unittest.TestCase):
    def test_stable_id_ignores_mapping_order(self):
        self.assertEqual(stable_id("x", {"a": 1, "b": 2}), stable_id("x", {"b": 2, "a": 1}))

    def test_split_is_order_independent(self):
        qids = [f"q{i}" for i in range(100)]
        forward = assign_splits(qids)
        backward = assign_splits(reversed(qids))
        self.assertEqual(forward, backward)
        self.assertEqual(forward["q7"], split_for_qid("q7"))

    def test_duplicate_and_overlap_rejected(self):
        with self.assertRaises(ValueError):
            assign_splits(["q1", "q1"])
        with self.assertRaises(ValueError):
            validate_disjoint({"train": ["q1"], "test": ["q1"]})

    def test_manifest_run_id_is_content_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.txt"
            path.write_text("固定输入", encoding="utf-8")
            first = build_manifest({"b": 2, "a": 1}, [path], directory, "2026-01-01T00:00:00Z")
            second = build_manifest({"a": 1, "b": 2}, [path], directory, "2026-02-01T00:00:00Z")
            self.assertEqual(first.run_id, second.run_id)
            self.assertEqual(first.config_sha256, second.config_sha256)
            self.assertEqual(next(iter(first.input_files.values())), sha256_file(path))


if __name__ == "__main__":
    unittest.main()

