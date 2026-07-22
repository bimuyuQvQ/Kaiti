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

    @staticmethod
    def _dense_observations():
        return [
            TraceObservation("Alpha beta", 4, prefix_token_ids=list(range(1, 5))),
            TraceObservation("Alpha beta.", 6, prefix_token_ids=list(range(1, 7))),
            TraceObservation("Alpha beta. Gamma", 8, prefix_token_ids=list(range(1, 9))),
            TraceObservation(
                "Alpha beta. Gamma uncertain",
                10,
                etc_triggered=True,
                etc_query="gamma",
                prefix_token_ids=list(range(1, 11)),
            ),
            TraceObservation(
                "Alpha beta. Gamma uncertain delta",
                12,
                prefix_token_ids=list(range(1, 13)),
            ),
            TraceObservation(
                "Alpha beta. Gamma uncertain delta.",
                14,
                prefix_token_ids=list(range(1, 15)),
            ),
            TraceObservation(
                "Alpha beta. Gamma uncertain delta. So",
                16,
                prefix_token_ids=list(range(1, 17)),
            ),
            TraceObservation(
                "Alpha beta. Gamma uncertain delta. So the answer is x",
                20,
                prefix_token_ids=list(range(1, 21)),
            ),
            TraceObservation(
                "Alpha beta. Gamma uncertain delta. So the answer is x. Extra text.",
                24,
                prefix_token_ids=list(range(1, 25)),
            ),
        ]

    def test_dense_timing_is_not_gated_by_etc(self):
        collector = CheckpointCollector(
            "q1",
            0,
            max_checkpoints=8,
            timing_config={
                "mode": "dense_timing_v1",
                "token_stride": 4,
                "max_token_grid_checkpoints": 3,
            },
        )
        for observation in self._dense_observations():
            collector.observe(observation)
        states = collector.finalize()
        types = [state.checkpoint_type for state in states]
        self.assertIn("token_grid", types)
        self.assertIn("sentence_boundary", types)
        self.assertIn("first_etc_trigger", types)
        self.assertIn("before_first_answer_marker", types)
        self.assertLess(types.index("token_grid"), types.index("first_etc_trigger"))
        self.assertTrue(all((state.token_index or 0) <= 16 for state in states))
        etc_state = next(state for state in states if state.checkpoint_type == "first_etc_trigger")
        self.assertEqual(etc_state.trace_metadata["etc_query"], "gamma")

    def test_dense_budget_preserves_etc_and_preanswer_anchors(self):
        collector = CheckpointCollector(
            "q1",
            0,
            max_checkpoints=3,
            timing_config={
                "mode": "dense_timing_v1",
                "token_stride": 4,
                "max_token_grid_checkpoints": 3,
            },
        )
        for observation in self._dense_observations():
            collector.observe(observation)
        states = collector.finalize()
        self.assertEqual(len(states), 3)
        self.assertIn("first_etc_trigger", [state.checkpoint_type for state in states])
        self.assertIn("before_first_answer_marker", [state.checkpoint_type for state in states])
        self.assertEqual([state.checkpoint_index for state in states], [0, 1, 2])

    def test_first_etc_only_policy_emits_no_other_state(self):
        collector = CheckpointCollector(
            "q1",
            0,
            max_checkpoints=1,
            timing_config={
                "mode": "dense_timing_v1",
                "include_etc_trigger": True,
                "include_sentence_boundaries": False,
                "include_before_answer": False,
                "token_stride": 1000000,
                "max_token_grid_checkpoints": 0,
                "max_candidates_per_sample": 1,
            },
        )
        for observation in self._dense_observations():
            collector.observe(observation)
        states = collector.finalize()
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].checkpoint_type, "first_etc_trigger")
        self.assertEqual(states[0].token_index, 10)


if __name__ == "__main__":
    unittest.main()
