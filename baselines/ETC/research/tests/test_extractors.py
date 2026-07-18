import unittest

from baselines.ETC.research.extractors import extract_answer


class ExtractorTests(unittest.TestCase):
    def test_first_answer_before_repetition(self):
        text = "Reasoning. The answer is Paris. The answer is London."
        self.assertEqual(extract_answer(text), "Paris")

    def test_so_the_answer_is_is_a_repeat(self):
        text = "The answer is Ada Lovelace. So the answer is Grace Hopper."
        self.assertEqual(extract_answer(text), "Ada Lovelace")

    def test_question_tail_and_special_suffix(self):
        text = "The answer is blue.</s> Question: unrelated"
        self.assertEqual(extract_answer(text), "blue")

    def test_missing_marker_is_empty(self):
        self.assertEqual(extract_answer("Only reasoning, no marker."), "")

    def test_legacy_requires_explicit_function(self):
        with self.assertRaises(ValueError):
            extract_answer("The answer is x", mode="legacy_original")


if __name__ == "__main__":
    unittest.main()

