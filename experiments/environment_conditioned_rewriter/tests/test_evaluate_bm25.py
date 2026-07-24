from env_rewriter.evaluate_bm25 import metrics, tokenize


def test_metrics() -> None:
    result = metrics(["a", "b", "c"], {"b"}, cutoff=3)
    assert 0.0 < result["ndcg@3"] < 1.0
    assert result["recall@3"] == 1.0
    assert result["mrr@3"] == 0.5


def test_tokenize() -> None:
    assert tokenize("GPU-aware RAG 2.0") == ["gpu", "aware", "rag", "2", "0"]
