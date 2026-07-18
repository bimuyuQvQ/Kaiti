import unittest

from baselines.ETC.research.query_candidates import QueryContext, build_query_candidates
from baselines.ETC.research.retrieval_adapter import MetadataBM25


class FakeBackend:
    text_key = "txt"
    title_key = "title"

    def __init__(self):
        self.last_request = None

    def search(self, *, index, body, size):
        self.last_request = {"index": index, "body": body, "size": size}
        return {
            "hits": {
                "hits": [
                    {
                        "_id": "doc-1",
                        "_score": 12.5,
                        "_source": {"title": "标题", "txt": "正文", "year": 2020},
                    },
                    {
                        "_id": "doc-2",
                        "_score": 9,
                        "_source": {"title": "标题二", "txt": "正文二"},
                    },
                ]
            }
        }


class QueryRetrievalTests(unittest.TestCase):
    def test_query_candidates_are_deduplicated(self):
        context = QueryContext(
            qid="q1",
            state_id="s1",
            question="Who wrote Hamlet?",
            prefix_text="We need the author.",
            etc_query="  who   wrote HAMLET? ",
        )
        candidates = build_query_candidates(context, "Hamlet author")
        self.assertEqual([item.source for item in candidates], ["question", "prefix_gap_v1"])
        self.assertEqual(len({item.candidate_id for item in candidates}), 2)

    def test_metadata_retriever_preserves_fields_and_legacy_query_shape(self):
        backend = FakeBackend()
        retriever = MetadataBM25(index_name="wiki", backend=backend)
        documents = retriever("hamlet author", topk=1)
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].document_id, "doc-1")
        self.assertEqual(documents[0].score, 12.5)
        self.assertEqual(documents[0].rank, 1)
        self.assertEqual(documents[0].raw_metadata, {"year": 2020})
        multi_match = backend.last_request["body"]["query"]["multi_match"]
        self.assertEqual(multi_match["type"], "best_fields")
        self.assertEqual(multi_match["tie_breaker"], 0.5)
        self.assertEqual(backend.last_request["size"], 1)


if __name__ == "__main__":
    unittest.main()

