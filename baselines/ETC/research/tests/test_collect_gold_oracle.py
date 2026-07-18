import unittest

from baselines.ETC.research.collect_gold_oracle import (
    make_gold_candidate,
    make_gold_documents,
    select_source_bundles,
)
from baselines.ETC.research.evidence_attribution import build_gold_index, normalize_text
from baselines.ETC.research.schema import CheckpointState


class GoldOracleTests(unittest.TestCase):
    @staticmethod
    def _payload():
        return [
            {
                "question": "Who linked A and B?",
                "answer": "Ada",
                "supporting_facts": [["A", 0], ["A", 1], ["B", 0]],
                "context": [
                    ["A", ["Ada created A.", "A is linked to B."]],
                    ["B", ["B was completed by Ada."]],
                ],
            }
        ]

    def test_gold_documents_group_sentences_by_title(self):
        gold = build_gold_index(self._payload())[normalize_text("Who linked A and B?")]
        documents = make_gold_documents(gold)
        self.assertEqual([document.title for document in documents], ["A", "B"])
        self.assertEqual(documents[0].text, "Ada created A. A is linked to B.")
        self.assertEqual(documents[0].raw_metadata["sentence_ids"], [0, 1])

    def test_candidate_is_explicitly_non_deployable_oracle(self):
        gold = build_gold_index(self._payload())[normalize_text("Who linked A and B?")]
        state = CheckpointState(
            qid="dev_0",
            sample_index=0,
            checkpoint_index=0,
            checkpoint_type="grid",
            prefix_text="prefix",
            prefix_token_ids=[1],
            state_id="state_0",
        )
        candidate = make_gold_candidate(state, gold)
        self.assertTrue(candidate.metadata["oracle"])
        self.assertFalse(candidate.metadata["deployment_available"])
        self.assertEqual(candidate.source, "gold_supporting_facts_oracle_v1")

    def test_source_selection_uses_contiguous_global_indices(self):
        bundles = [
            {"sample_index": index, "qid": f"q{index}"}
            for index in range(4)
        ]
        selected = select_source_bundles(bundles, start_index=1, sample=2)
        self.assertEqual([row["sample_index"] for row in selected], [1, 2])
        with self.assertRaisesRegex(ValueError, "不足"):
            select_source_bundles(bundles, start_index=3, sample=2)


if __name__ == "__main__":
    unittest.main()
