import unittest

from baselines.ETC.research.summarize_rollouts import summarize_bundles


class SummarizeTests(unittest.TestCase):
    def test_benefits_flips_oracle_and_groups(self):
        bundle = {
            "sample_index": 0,
            "qid": "q0",
            "no_retrieval_extracted_answer": "",
            "no_retrieval_scores": {"f1": 0.0},
            "states": [
                {"state_id": "s1", "checkpoint_type": "first_sentence_boundary"},
                {"state_id": "s2", "checkpoint_type": "first_etc_trigger"},
            ],
            "queries": [
                {"candidate_id": "q1", "source": "question", "text": "question text"},
                {"candidate_id": "q2", "source": "etc_qfs", "text": "etc query"},
            ],
            "actions": [
                {"state_id": "s1", "action_type": "skip", "extracted_answer": "", "scores": {"f1": 0.0, "accuracy": 0}},
                {"state_id": "s1", "action_type": "retrieve", "query_candidate_id": "q1", "extracted_answer": "x", "scores": {"f1": 1.0, "accuracy": 1}},
                {"state_id": "s2", "action_type": "skip", "extracted_answer": "", "scores": {"f1": 0.0, "accuracy": 0}},
                {"state_id": "s2", "action_type": "retrieve", "query_candidate_id": "q2", "extracted_answer": "", "scores": {"f1": 0.0, "accuracy": 0}},
            ],
        }
        summary = summarize_bundles([bundle])
        self.assertEqual(summary["benefit_counts"], {"positive": 1, "zero": 1, "negative": 0})
        self.assertEqual(summary["flips"]["wrong_to_correct"], 1)
        self.assertEqual(summary["mean_sample_timing_query_oracle_gain"], 1.0)
        self.assertEqual(summary["by_query_source"]["question"]["positive_rate"], 1.0)
        self.assertEqual(summary["skip_inconsistency_count"], 0)

    def test_inconsistent_skip_refuses_oracle(self):
        bundle = {
            "sample_index": 0,
            "qid": "q0",
            "no_retrieval_extracted_answer": "x",
            "no_retrieval_scores": {"f1": 1.0},
            "states": [{"state_id": "s1", "checkpoint_type": "first_etc_trigger"}],
            "queries": [],
            "actions": [
                {
                    "state_id": "s1",
                    "action_type": "skip",
                    "extracted_answer": "y",
                    "scores": {"f1": 0.0, "accuracy": 0},
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "拒绝计算 oracle"):
            summarize_bundles([bundle])

    def test_sensitivity_extractor_uses_parallel_scores(self):
        version = "first_answer_sentence_v2"
        bundle = {
            "sample_index": 0,
            "qid": "q0",
            "no_retrieval_extracted_answer": "yes. explanation",
            "no_retrieval_scores": {"f1": 0.0},
            "no_retrieval_alternative_extractions": {version: "yes"},
            "no_retrieval_alternative_scores": {version: {"f1": 1.0}},
            "states": [{"state_id": "s1", "checkpoint_type": "before_first_answer_marker"}],
            "queries": [],
            "actions": [
                {
                    "state_id": "s1",
                    "action_type": "skip",
                    "extracted_answer": "yes. explanation",
                    "scores": {"f1": 0.0, "accuracy": 0},
                    "alternative_extractions": {version: "yes"},
                    "alternative_scores": {version: {"f1": 1.0, "accuracy": 1}},
                }
            ],
        }
        summary = summarize_bundles([bundle], extractor_version=version)
        self.assertEqual(summary["extractor_version"], version)
        self.assertEqual(summary["skip_inconsistency_count"], 0)


if __name__ == "__main__":
    unittest.main()
