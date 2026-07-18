import tempfile
import unittest
from pathlib import Path

from baselines.ETC.research.jsonl_io import append_jsonl, read_jsonl
from baselines.ETC.research.rollout import audit_rollouts, counterfactual_benefit, make_action_id
from baselines.ETC.research.schema import ActionRollout


def rollout(state, action_type, score, query_id=None, status="complete"):
    return ActionRollout(
        qid="q1",
        state_id=state,
        action_id=make_action_id(state, action_type, query_id),
        action_type=action_type,
        query_candidate_id=query_id,
        prediction="The answer is x.",
        extracted_answer="x",
        scores={"f1": score, "accuracy": float(score == 1)},
        status=status,
    )


class RolloutJsonlTests(unittest.TestCase):
    def test_benefit_and_complete_audit(self):
        skip = rollout("s1", "skip", 0.25)
        positive = rollout("s1", "retrieve", 0.75, "q-positive")
        negative = rollout("s1", "retrieve", 0.0, "q-negative")
        self.assertEqual(counterfactual_benefit(positive, skip), 0.5)
        report = audit_rollouts(
            [skip, positive, negative],
            {"s1": ["q-positive", "q-negative"]},
        )
        self.assertTrue(report.complete, report.errors)
        self.assertEqual(report.benefit_counts, {"positive": 1, "zero": 0, "negative": 1})

    def test_missing_action_fails_audit(self):
        report = audit_rollouts([rollout("s1", "skip", 0.5)], {"s1": ["missing"]})
        self.assertFalse(report.complete)
        self.assertTrue(any("缺少查询动作" in error for error in report.errors))

    def test_jsonl_roundtrip_and_truncation_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            append_jsonl(path, [{"qid": "一"}, {"qid": "二"}])
            self.assertEqual([row["qid"] for row in read_jsonl(path)], ["一", "二"])
            path.write_text('{"qid": "一"}\n{"qid":', encoding="utf-8")
            with self.assertRaises(ValueError):
                list(read_jsonl(path))


if __name__ == "__main__":
    unittest.main()

