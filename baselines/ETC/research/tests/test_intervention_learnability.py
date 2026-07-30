import importlib.util
import unittest

from baselines.ETC.research.build_intervention_supervision import build_supervision_rows
from baselines.ETC.research.probe_intervention_learnability import run_probe


VERSION = "first_answer_sentence_v2"


def action(action_id, action_type, f1, accuracy, query_id=None, metadata=None):
    return {
        "action_id": action_id,
        "action_type": action_type,
        "state_id": "state",
        "query_candidate_id": query_id,
        "alternative_scores": {VERSION: {"f1": f1, "accuracy": accuracy}},
        "scores": {"f1": f1, "accuracy": accuracy},
        "retrieved_documents": (
            []
            if action_type == "skip"
            else [
                {
                    "document_id": "d1",
                    "rank": 1,
                    "title": "title",
                    "text": "visible evidence",
                    "score": 3.0,
                }
            ]
        ),
        "generation_metadata": metadata or {},
    }


def bundles(index, keep_f1=0.0, restart_f1=1.0):
    source = {
        "sample_index": index,
        "qid": f"q{index}",
        "question": f"question {index}",
        "ground_truth": ["forbidden answer"],
        "states": [
            {
                "state_id": "state",
                "prefix_text": f"prefix {index}",
                "features": {"entropy_last": 0.1 + index},
            }
        ],
        "queries": [
            {
                "candidate_id": "query",
                "source": "question",
                "text": f"query {index}",
            }
        ],
        "actions": [
            action("append", "retrieve", 0.5, 0.0, query_id="query"),
        ],
    }
    restart = {
        "sample_index": index,
        "qid": f"q{index}",
        "actions": [
            action("keep", "skip", keep_f1, float(keep_f1 >= 1.0)),
            action(
                "restart",
                "retrieve",
                restart_f1,
                float(restart_f1 >= 1.0),
                metadata={"original_action_id": "append"},
            ),
        ],
    }
    return source, restart


class BuildInterventionSupervisionTests(unittest.TestCase):
    def test_builds_visible_state_without_answer_fields(self):
        source, restart = bundles(0, keep_f1=1.0, restart_f1=0.0)
        rows = build_supervision_rows([source], [restart])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["labels"]["restart_harm"], 1)
        self.assertEqual(rows[0]["labels"]["restart_rescue"], 0)
        self.assertIn("[PREFIX]", rows[0]["text_full"])
        self.assertNotIn("forbidden answer", rows[0]["text_full"])
        self.assertNotIn("ground_truth", rows[0])

    def test_rejects_document_mismatch(self):
        source, restart = bundles(0)
        restart["actions"][1]["retrieved_documents"][0]["document_id"] = "other"
        with self.assertRaisesRegex(ValueError, "文档不一致"):
            build_supervision_rows([source], [restart])

    @unittest.skipUnless(
        importlib.util.find_spec("sklearn") is not None, "需要 scikit-learn"
    )
    def test_nested_probe_emits_one_oof_record_per_variant_and_row(self):
        sources, restarts = [], []
        for index in range(12):
            keep = 1.0 if index % 4 == 0 else 0.0
            restart = 0.0 if index % 4 == 0 else 1.0
            source, restart_bundle = bundles(index, keep_f1=keep, restart_f1=restart)
            sources.append(source)
            restarts.append(restart_bundle)
        rows = build_supervision_rows(sources, restarts)
        report, records = run_probe(
            rows,
            variants=("numeric", "full_tfidf"),
            outer_folds=3,
            inner_folds=2,
            bootstrap_samples=50,
        )
        self.assertEqual(report["rows"], 12)
        self.assertEqual(len(records), 24)
        self.assertEqual(set(report["variants"]), {"numeric", "full_tfidf"})
        self.assertEqual(
            {record["qid"] for record in records if record["variant"] == "numeric"},
            {f"q{index}" for index in range(12)},
        )


if __name__ == "__main__":
    unittest.main()
