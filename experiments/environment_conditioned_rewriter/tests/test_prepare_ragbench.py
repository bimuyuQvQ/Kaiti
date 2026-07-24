from env_rewriter.prepare_ragbench import (
    document_id,
    normalize_text,
    relevant_document_indices,
)


def test_normalization_makes_stable_id() -> None:
    assert normalize_text("a\n  b") == "a b"
    assert document_id("a\n b") == document_id("a  b")


def test_relevant_document_indices() -> None:
    row = {
        "all_relevant_sentence_keys": ["d1-s0"],
        "documents_sentences": [
            [["d0-s0", "irrelevant"]],
            [["d1-s0", "relevant"], ["d1-s1", "other"]],
        ],
    }
    assert relevant_document_indices(row) == {1}
