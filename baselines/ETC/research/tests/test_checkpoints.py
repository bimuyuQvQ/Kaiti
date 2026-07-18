import unittest

from baselines.ETC.research.checkpoints import CheckpointCollector, TraceObservation


class CheckpointTests(unittest.TestCase):
    def test_temporal_order_and_answer_prefix(self):
        collector = CheckpointCollector("q1", 7)
        collector.observe(
            TraceObservation("First fact.", 3, {"entropy": 1.2}, prefix_token_ids=[1, 2, 3])
        )
        collector.observe(
            TraceObservation(
                "First fact. Second uncertain fact",
                7,
                {"etc_signal": 1.5},
                etc_triggered=True,
                etc_query="second fact",
                prefix_token_ids=[1, 2, 3, 4, 5, 6, 7],
            )
        )
        collector.observe(
            TraceObservation(
                "First fact. So",
                9,
                prefix_token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9],
            )
        )
        collector.observe(
            TraceObservation(
                "First fact. So the answer is x",
                12,
                prefix_token_ids=list(range(1, 13)),
            )
        )
        states = collector.finalize()
        self.assertEqual(
            [state.checkpoint_type for state in states],
            ["first_sentence_boundary", "first_etc_trigger", "before_first_answer_marker"],
        )
        self.assertEqual(states[-1].prefix_text, "First fact. So")
        self.assertEqual(states[-1].prefix_token_ids[-1], 9)
        self.assertEqual(states[1].trace_metadata["etc_query"], "second fact")
        self.assertEqual([state.checkpoint_index for state in states], [0, 1, 2])

    def test_equivalent_prefixes_are_deduplicated(self):
        collector = CheckpointCollector("q1", 0)
        collector.observe(
            TraceObservation(
                "A fact.",
                4,
                etc_triggered=True,
                etc_query="fact",
                prefix_token_ids=[1, 2, 3, 4],
            )
        )
        states = collector.finalize()
        self.assertEqual(len(states), 1)


if __name__ == "__main__":
    unittest.main()
