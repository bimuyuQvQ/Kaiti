import json
import tempfile
import unittest
from pathlib import Path

from baselines.ETC.research.collect_rollouts import (
    ConfigNamespace,
    build_audit,
    resolve_sample_indices,
)


class CollectRolloutTests(unittest.TestCase):
    def test_config_namespace_supports_both_legacy_access_styles(self):
        config = ConfigNamespace({"es_index_name": "wiki", "sample": 1})
        self.assertEqual(config.es_index_name, "wiki")
        self.assertIn("es_index_name", config)
        self.assertNotIn("missing", config)

    def test_sample_shard_indices_are_global_and_bounded(self):
        self.assertEqual(resolve_sample_indices(20, 5, 10), [10, 11, 12, 13, 14])
        self.assertEqual(resolve_sample_indices(12, -1, 10), [10, 11])
        self.assertEqual(resolve_sample_indices(12, 10, 10), [10, 11])
        with self.assertRaises(ValueError):
            resolve_sample_indices(12, 1, 12)

    @staticmethod
    def _bundle(skip_prediction="The answer is x."):
        scores = {"em": 1.0, "accuracy": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0}
        alternative_scores = {"first_answer_sentence_v2": scores}
        return {
            "sample_index": 0,
            "qid": "q0",
            "no_retrieval_prediction": "The answer is x.",
            "no_retrieval_extracted_answer": "x",
            "no_retrieval_scores": scores,
            "no_retrieval_alternative_extractions": {"first_answer_sentence_v2": "x"},
            "no_retrieval_alternative_scores": alternative_scores,
            "states": [{"state_id": "s1", "prefix_token_ids": [1, 2]}],
            "queries": [],
            "actions": [
                {
                    "qid": "q0",
                    "state_id": "s1",
                    "action_id": "a1",
                    "action_type": "skip",
                    "prediction": skip_prediction,
                    "extracted_answer": "x",
                    "scores": scores,
                    "alternative_extractions": {"first_answer_sentence_v2": "x"},
                    "alternative_scores": alternative_scores,
                    "status": "complete",
                    "query_candidate_id": None,
                    "retrieved_documents": [],
                    "generation_metadata": {"canonical_skip_reused": True},
                    "error": None,
                }
            ],
        }

    def test_protocol_audit_accepts_exact_canonical_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.json"
            path.write_text(json.dumps(self._bundle()), encoding="utf-8")
            report = build_audit([path])
        self.assertTrue(report["complete"], report["errors"])
        self.assertTrue(report["protocol_consistency_complete"])

    def test_protocol_audit_rejects_prediction_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.json"
            path.write_text(json.dumps(self._bundle("The answer is y.")), encoding="utf-8")
            report = build_audit([path])
        self.assertFalse(report["complete"])
        self.assertTrue(any("prediction 漂移" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
