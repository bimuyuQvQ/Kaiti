import unittest

from baselines.ETC.research.collect_retrieved_revision import (
    make_revision_candidate,
    select_prior_sentence_state,
)
from baselines.ETC.research.schema import CheckpointState


class RetrievedRevisionTests(unittest.TestCase):
    def test_selects_nearest_strictly_prior_sentence(self):
        states = [
            {"state_id": "s0", "checkpoint_type": "sentence_boundary", "token_index": 8},
            {"state_id": "s1", "checkpoint_type": "token_grid", "token_index": 12},
            {"state_id": "s2", "checkpoint_type": "sentence_boundary", "token_index": 19},
            {"state_id": "etc", "checkpoint_type": "first_etc_trigger", "token_index": 24},
            {"state_id": "s3", "checkpoint_type": "sentence_boundary", "token_index": 38},
        ]
        self.assertEqual(select_prior_sentence_state(states, states[3])["state_id"], "s2")
        self.assertIsNone(select_prior_sentence_state(states, states[0]))

    def test_revision_candidate_preserves_query_and_marks_operator(self):
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
        candidate = make_revision_candidate(state, original)
        self.assertEqual(candidate.text, original["text"])
        self.assertEqual(candidate.metadata["intervention"], "local_revision_from_prior_sentence")
        self.assertTrue(candidate.metadata["deployment_available"])


if __name__ == "__main__":
    unittest.main()
