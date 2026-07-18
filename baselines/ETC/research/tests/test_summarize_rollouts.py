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
                {"candidate_id": "q1", "source": "question"},
                {"candidate_id": "q2", "source": "etc_qfs"},
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


if __name__ == "__main__":
    unittest.main()
