from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

from kb_landscape.bm25 import BM25Index
from kb_landscape.io import load_beir_dataset
from kb_landscape.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from kb_landscape.prepare_hotpotqa import convert
from kb_landscape.prepare_mtrag_candidates import build_candidates
from kb_landscape.run_diagnostic import run


def _write_fixture(root: Path, *, suffix: str = "") -> None:
    corpus = [
        {"_id": f"d1{suffix}", "title": "Alpha", "text": "rarealpha fruit vitamin nutrition"},
        {"_id": f"d2{suffix}", "title": "Beta", "text": "common fruit recipe kitchen"},
        {"_id": f"d3{suffix}", "title": "Gamma", "text": "cloud storage object bucket technical"},
        {"_id": f"d4{suffix}", "title": "Delta", "text": "cloud compute virtual machine technical"},
    ]
    queries = [
        {"_id": f"q1{suffix}", "text": "which rarealpha fruit has vitamin nutrition"},
        {"_id": f"q2{suffix}", "text": "cloud object bucket storage"},
    ]
    qrels = [
        ["query-id", "corpus-id", "score"],
        [f"q1{suffix}", f"d1{suffix}", "1"],
        [f"q2{suffix}", f"d3{suffix}", "1"],
    ]
    root.mkdir(parents=True)
    (root / "qrels").mkdir()
    with (root / "corpus.jsonl").open("w", encoding="utf-8") as handle:
        for row in corpus:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (root / "queries.jsonl").open("w", encoding="utf-8") as handle:
        for row in queries:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (root / "qrels" / "test.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerows(qrels)


class MetricsTest(unittest.TestCase):
    def test_binary_metrics(self) -> None:
        qrels = {"d1": 1.0, "d2": 1.0}
        retrieved = ["d3", "d1", "d4", "d2"]
        self.assertGreater(ndcg_at_k(retrieved, qrels, 4), 0.0)
        self.assertAlmostEqual(recall_at_k(retrieved, qrels, 3), 0.5)
        self.assertAlmostEqual(mrr_at_k(retrieved, qrels, 4), 0.5)


class DiagnosticTest(unittest.TestCase):
    def test_bm25_returns_numeric_ranked_scores(self) -> None:
        index = BM25Index().fit(["alpha beta", "beta gamma", "delta"])
        result = index.search("alpha", top_k=2)
        self.assertEqual(result.indices.tolist()[0], 0)
        self.assertEqual(result.scores.dtype.kind, "f")
        self.assertGreater(float(result.scores[0]), float(result.scores[1]))

    def test_beir_loading_and_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "toy"
            _write_fixture(root)
            dataset = load_beir_dataset(root)
            self.assertEqual(len(dataset.documents), 4)
            self.assertEqual(len(dataset.queries), 2)

            args = argparse.Namespace(
                dataset=str(root),
                corpus_name="toy",
                output_dir=str(Path(temporary) / "output"),
                split="test",
                top_k=3,
                max_queries=None,
                seed=7,
                external_candidates=None,
                actions=["keep", "keywords", "prf_expand", "prf_reduce"],
            )
            frame, summary = run(args)
            self.assertEqual(len(frame), 2)
            self.assertIn("feat_score_margin12", frame.columns)
            self.assertIn("ndcg__prf_expand", frame.columns)
            self.assertEqual(summary["queries"], 2)

    def test_mtrag_candidate_alignment(self) -> None:
        rows = [
            {"_id": "q1", "text": "first"},
            {"_id": "q2", "text": "second"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = root / "left.jsonl"
            right = root / "right.jsonl"
            left.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            right.write_text(
                "\n".join(json.dumps({"_id": row["_id"], "text": row["text"] + " alt"}) for row in rows)
                + "\n",
                encoding="utf-8",
            )
            output = root / "candidates.jsonl"
            summary = build_candidates([f"left={left}", f"right={right}"], output, reference=left)
            self.assertEqual(summary["queries"], 2)
            self.assertEqual(summary["rows"], 4)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 4)

    def test_hotpotqa_conversion(self) -> None:
        row = {
            "id": "q-hotpot",
            "question": "Which document is relevant?",
            "metadata": {
                "supporting_facts": {"title": ["Relevant"], "sent_id": [0]},
                "context": {
                    "title": ["Relevant", "Distractor"],
                    "sentences": [["relevant evidence"], ["unrelated text"]],
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "dev.jsonl"
            input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            output = root / "beir"
            summary = convert(input_path, output, max_queries=1)
            dataset = load_beir_dataset(output)
            self.assertEqual(summary["queries"], 1)
            self.assertEqual(summary["documents"], 2)
            self.assertEqual(len(dataset.qrels["q-hotpot"]), 1)

    def test_hotpotqa_huggingface_rows_conversion(self) -> None:
        row = {
            "id": "q-hf",
            "question": "Which document is relevant?",
            "supporting_facts": {"title": ["Relevant"], "sent_id": [0]},
            "context": {
                "title": ["Relevant", "Distractor"],
                "sentences": [["relevant evidence"], ["unrelated text"]],
            },
        }
        payload = {"rows": [{"row_idx": 0, "row": row, "truncated_cells": []}], "partial": False}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "hf.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            output = root / "beir"
            summary = convert(input_path, output, max_queries=1)
            self.assertEqual(summary["queries"], 1)
            self.assertEqual(len(load_beir_dataset(output).qrels["q-hf"]), 1)


if __name__ == "__main__":
    unittest.main()
