import unittest

from baselines.ETC.research.collect_retrieved_restart import (
    make_restart_candidate,
    select_first_etc_state,
)
from baselines.ETC.research.schema import CheckpointState


class RetrievedRestartTests(unittest.TestCase):
    def test_selects_single_etc_state_or_none(self):
        states = [
            {"state_id": "s0", "checkpoint_type": "sentence_boundary"},
            {"state_id": "s1", "checkpoint_type": "first_etc_trigger"},
        ]
        self.assertEqual(select_first_etc_state(states)["state_id"], "s1")
        self.assertIsNone(select_first_etc_state(states[:1]))
        with self.assertRaisesRegex(ValueError, "多个"):
            select_first_etc_state([states[1], {**states[1], "state_id": "s2"}])

    def test_restart_candidate_preserves_query_and_changes_identity(self):
        state = CheckpointState(
            qid="dev_0",
            sample_index=0,
            checkpoint_index=0,
            checkpoint_type="first_etc_trigger",
            prefix_text="prefix",
            prefix_token_ids=[1],
            state_id="state_0",
        )
        original = {
            "candidate_id": "original_q",
            "source": "question",
            "text": "Who wrote Hamlet?",
            "normalized_text": "who wrote hamlet?",
        }
        candidate = make_restart_candidate(state, original)
        self.assertEqual(candidate.text, original["text"])
        self.assertNotEqual(candidate.candidate_id, original["candidate_id"])
        self.assertEqual(candidate.metadata["original_query_source"], "question")
        self.assertTrue(candidate.metadata["deployment_available"])


if __name__ == "__main__":
    unittest.main()
