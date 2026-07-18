import unittest

from baselines.ETC.research.evidence_attribution import (
    attribute_bundles,
    build_gold_index,
    normalize_text,
    normalize_title,
)


VERSION = "first_answer_sentence_v2"


def action(state, kind, score, query_id=None, documents=None):
    return {
        "state_id": state,
        "action_type": kind,
        "query_candidate_id": query_id,
        "alternative_scores": {VERSION: {"f1": score}},
        "retrieved_documents": documents or [],
    }


class EvidenceAttributionTests(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "rows": [
                {
                    "row": {
                        "question": "Who connected Alpha and Beta?",
                        "answer": "Ada",
                        "supporting_facts": {
                            "title": ["Alpha_Page", "Beta"],
                            "sent_id": [0, 1],
                        },
                        "context": {
                            "title": ["Alpha Page", "Beta"],
                            "sentences": [
                                ["Ada created Alpha."],
                                ["Intro.", "Beta was connected by Ada!"],
                            ],
                        },
                    }
                }
            ]
        }

    def test_normalization_handles_case_underscore_and_punctuation(self):
        self.assertEqual(normalize_text("Alpha_Page!"), normalize_text("alpha page"))
        self.assertEqual(normalize_title("Alpha_Page"), normalize_title("alpha page"))
        self.assertNotEqual(normalize_title("Romeo + Juliet"), normalize_title("Romeo ～ Juliet"))

    def test_gold_loader_resolves_supporting_sentences(self):
        gold = build_gold_index(self.payload)
        row = gold[normalize_text("Who connected Alpha and Beta?")]
        self.assertEqual(row["normalized_gold_titles"], ["alpha page", "beta"])
        self.assertEqual(len(row["gold_support_sentences"]), 2)

    def test_attributes_retrieval_and_utilization_failures(self):
        bundle = {
            "sample_index": 0,
            "qid": "dev_0",
            "question": "Who connected Alpha and Beta?",
            "ground_truth": "Ada",
            "states": [{"state_id": "s0", "checkpoint_type": "grid"}],
            "queries": [
                {"candidate_id": "q_good", "source": "question", "text": "Alpha Beta"},
                {"candidate_id": "q_bad", "source": "gap", "text": "unrelated"},
                {"candidate_id": "q_unused", "source": "gap", "text": "Beta Ada"},
            ],
            "actions": [
                action("s0", "skip", 0.0),
                action(
                    "s0",
                    "retrieve",
                    1.0,
                    "q_good",
                    [{"title": "Alpha_Page", "text": "Ada created Alpha.", "rank": 2}],
                ),
                action(
                    "s0",
                    "retrieve",
                    0.0,
                    "q_bad",
                    [{"title": "Gamma", "text": "No evidence here.", "rank": 1}],
                ),
                action(
                    "s0",
                    "retrieve",
                    0.0,
                    "q_unused",
                    [{"title": "Beta", "text": "Beta was connected by Ada!", "rank": 1}],
                ),
            ],
        }
        result = attribute_bundles([bundle], build_gold_index(self.payload))
        self.assertEqual(result["by_benefit_bucket"]["positive"]["count"], 1)
        self.assertEqual(result["attribution_counts"]["gold_title_miss"]["actions"], 1)
        self.assertEqual(
            result["attribution_counts"]["support_sentence_hit_but_no_gain"]["actions"], 1
        )
        positive = next(row for row in result["diagnostic_cases"] if row["benefit"] > 0)
        self.assertEqual(positive["first_gold_title_rank"], 2)
        self.assertTrue(positive["answer_hit"])

    def test_rejects_missing_gold_question(self):
        bundle = {
            "question": "missing",
            "qid": "x",
            "sample_index": 0,
            "states": [],
            "queries": [],
            "actions": [],
        }
        with self.assertRaisesRegex(ValueError, "找不到"):
            attribute_bundles([bundle], build_gold_index(self.payload))


if __name__ == "__main__":
    unittest.main()
