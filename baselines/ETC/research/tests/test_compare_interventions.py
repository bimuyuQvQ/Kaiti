import unittest

from baselines.ETC.research.compare_interventions import compare_bundle_sets


VERSION = "first_answer_sentence_v2"


def action(action_id, action_type, score, accuracy, state_id="s1", query_id=None, metadata=None):
    return {
        "action_id": action_id,
        "action_type": action_type,
        "state_id": state_id,
        "query_candidate_id": query_id,
        "alternative_scores": {VERSION: {"f1": score, "accuracy": accuracy}},
        "scores": {"f1": score, "accuracy": accuracy},
        "retrieved_documents": (
            []
            if action_type == "skip"
            else [{"document_id": "d1", "rank": 1, "text": "evidence"}]
        ),
        "generation_metadata": metadata or {},
    }


class CompareInterventionsTests(unittest.TestCase):
    def test_matches_state_query_documents_and_reports_harm(self):
        source = {
            "sample_index": 0,
            "qid": "q0",
            "queries": [{"candidate_id": "q1", "source": "question", "text": "query"}],
            "actions": [
                action("keep", "skip", 1.0, 1.0),
                action("append", "retrieve", 0.5, 0.0, query_id="q1"),
            ],
        }
        restart = {
            "sample_index": 0,
            "qid": "q0",
            "queries": [],
            "actions": [
                action("keep", "skip", 1.0, 1.0),
                action(
                    "restart",
                    "retrieve",
                    1.0,
                    1.0,
                    metadata={"original_action_id": "append"},
                ),
            ],
        }
        report = compare_bundle_sets([source], [restart])
        self.assertEqual(report["matched_actions"], 1)
        self.assertEqual(report["restart_vs_append"]["restart_better"], 1)
        self.assertEqual(report["accuracy_flips"]["append"]["correct_to_wrong"], 1)
        self.assertEqual(report["accuracy_flips"]["restart"]["unchanged"], 1)
        self.assertTrue(report["protocol_checks"]["same_retrieved_documents"])

    def test_rejects_document_mismatch(self):
        source = {
            "sample_index": 0,
            "qid": "q0",
            "queries": [{"candidate_id": "q1", "source": "question", "text": "query"}],
            "actions": [
                action("keep", "skip", 0.0, 0.0),
                action("append", "retrieve", 0.0, 0.0, query_id="q1"),
            ],
        }
        mismatch = action(
            "restart", "retrieve", 1.0, 1.0, metadata={"original_action_id": "append"}
        )
        mismatch["retrieved_documents"][0]["document_id"] = "other"
        restart = {
            "sample_index": 0,
            "qid": "q0",
            "queries": [],
            "actions": [action("keep", "skip", 0.0, 0.0), mismatch],
        }
        with self.assertRaisesRegex(ValueError, "检索文档不匹配"):
            compare_bundle_sets([source], [restart])


if __name__ == "__main__":
    unittest.main()
