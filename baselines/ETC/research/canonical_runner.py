"""Canonical single-retrieval counterfactual runner built beside legacy ETC.

The runner reuses the released model, detector, QFS, BM25 query shape, and
one-sentence evidence injection.  It never mutates the legacy ETC classes.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .checkpoints import CheckpointCollector, TraceObservation
from .extractors import EXTRACTOR_VERSION, SENSITIVITY_EXTRACTOR_VERSION, extract_answer
from .query_candidates import QueryContext, build_query_candidates, prefix_gap_prompt
from .retrieval_adapter import MetadataBM25
from .rollout import make_action_id
from .schema import ActionRollout, CheckpointState, QueryCandidate, RetrievedDocument, to_dict


class CanonicalTrajectoryRunner:
    def __init__(self, etc_model: Any, dataset: Any, research_config: Dict[str, Any]) -> None:
        # Imports stay local so CPU-only schema/tests do not load torch/spaCy/BEIR.
        from generate import CheckerOutput, get_top_sentence

        self.etc = etc_model
        self.dataset = dataset
        self.config = research_config
        self.CheckerOutput = CheckerOutput
        self.get_top_sentence = get_top_sentence
        self.generator = etc_model.generator
        self.tokenizer = etc_model.tokenizer
        self.max_tokens = int(etc_model.generate_max_length)
        self.timing_config = dict(research_config.get("timing_candidates", {}))
        self.max_checkpoints = int(
            self.timing_config.get(
                "max_candidates_per_sample",
                research_config.get("max_checkpoints_per_sample", 3),
            )
        )
        self.extractor = research_config.get("answer_extractor", EXTRACTOR_VERSION)
        self.sensitivity_extractors = list(
            research_config.get(
                "sensitivity_answer_extractors",
                [SENSITIVITY_EXTRACTOR_VERSION],
            )
        )
        self.retriever = MetadataBM25(
            index_name=research_config.get("es_index_name", getattr(etc_model, "es_index_name", "wiki"))
        )

    @staticmethod
    def _join_prefix(left: str, right: str) -> str:
        return " ".join(part for part in (left, right) if part)

    def _answer_token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @staticmethod
    def _tensor_values(value: Any) -> List[float]:
        if value is None:
            return []
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().reshape(-1).tolist()
        return [float(item) for item in value]

    def _features(self, outputs: Any, check_info: Any) -> Dict[str, Any]:
        entropy = self._tensor_values(outputs.entropies)
        attention = self._tensor_values(outputs.max_atten)
        mt_s2 = [float(item["val"].detach().cpu().item()) for item in (outputs.mt_s2 or [])]
        features: Dict[str, Any] = {
            "feature_version": "etc_online_features_v1",
            "entropy_mean": sum(entropy) / len(entropy) if entropy else None,
            "entropy_last": entropy[-1] if entropy else None,
            "max_attention_mean": sum(attention) / len(attention) if attention else None,
            "max_attention_last": attention[-1] if attention else None,
            "mt_s2_last": mt_s2[-1] if mt_s2 else None,
            "etc_triggered": bool(check_info and check_info.hallucination),
            "detector_threshold": float(self.etc.hallucination_threshold),
        }
        if check_info and check_info.hallucination:
            features["trigger_word_index"] = int(check_info.curr_st)
            features["trigger_sentence_end_word_index"] = int(check_info.curr_en)
            if 0 <= check_info.curr_st < len(attention) and check_info.curr_st < len(mt_s2):
                # Diagnostic raw product only; legacy thresholding additionally normalizes
                # attention within the current sentence, so this is not named ETC signal.
                features["trigger_raw_attention_x_mt_s2"] = attention[check_info.curr_st] * mt_s2[check_info.curr_st]
        return features

    def collect_no_retrieval_trace(
        self,
        qid: str,
        sample_index: int,
        question: str,
        demo_text: str,
    ) -> Tuple[str, List[int], List[CheckpointState]]:
        generated = ""
        final_generated_token_ids: List[int] = []
        collector = CheckpointCollector(
            qid,
            sample_index,
            self.max_checkpoints,
            timing_config=self.timing_config,
        )

        while self._answer_token_count(generated) < self.max_tokens:
            remaining = max(1, self.max_tokens - self._answer_token_count(generated))
            existing_tokens = self.generator.tokenize(generated, is_start=False)
            existing_token_ids = self.tokenizer.convert_tokens_to_ids(existing_tokens)

            def observe_without_retrieving(outputs: Any) -> Any:
                with contextlib.redirect_stdout(io.StringIO()):
                    check_info = self.etc.hallucination_check(outputs)
                etc_query = None
                if check_info.hallucination:
                    try:
                        with contextlib.redirect_stdout(io.StringIO()):
                            etc_query = self.etc.generate_retrieve_qry(outputs, check_info).strip()
                    except (IndexError, RuntimeError, ValueError) as exc:
                        etc_query = None
                prefix = self._join_prefix(generated, outputs.new_text.strip())
                new_token_ids = self.tokenizer.convert_tokens_to_ids(outputs.blocks[-1].tokens)
                prefix_token_ids = list(existing_token_ids) + list(new_token_ids)
                collector.observe(
                    TraceObservation(
                        generated_prefix=prefix,
                        token_index=len(prefix_token_ids),
                        prefix_token_ids=prefix_token_ids,
                        features=self._features(outputs, check_info),
                        etc_triggered=bool(check_info.hallucination),
                        etc_query=etc_query,
                    )
                )
                # Continue the no-retrieval trajectory even when legacy ETC fires.
                return self.CheckerOutput(hallucination=False)

            with contextlib.redirect_stdout(io.StringIO()):
                outputs, _ = self.generator.generate_online(
                    input_texts=[demo_text, "\nQuestion:", question, "\nAnswer:", generated],
                    max_length=remaining,
                    should_retrieve=observe_without_retrieving,
                )
            if outputs.empty:
                break
            final_generated_token_ids = list(existing_token_ids) + list(
                self.tokenizer.convert_tokens_to_ids(outputs.blocks[-1].tokens)
            )
            updated = self._join_prefix(generated, outputs.new_text.strip())
            if len(updated) <= len(generated):
                break
            generated = updated
            if outputs.ended:
                break
        return generated, final_generated_token_ids, collector.finalize()

    def _generate_from_ids(
        self,
        input_token_ids: Sequence[int],
        max_new_tokens: int,
        stop_at_newline: bool = False,
    ) -> Tuple[bool, str, List[int]]:
        if max_new_tokens <= 0:
            return True, "", []
        import torch

        input_ids = torch.tensor([list(input_token_ids)], device=self.generator.model.device)
        kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if stop_at_newline:
            kwargs.update({"stop_strings": "\n", "tokenizer": self.tokenizer})
        output_ids = self.generator.model.generate(**kwargs)[0, input_ids.shape[1] :].tolist()
        if output_ids and output_ids[0] == self.tokenizer.bos_token_id:
            output_ids = output_ids[1:]
        ended = bool(output_ids and output_ids[-1] == self.tokenizer.eos_token_id)
        if ended:
            output_ids = output_ids[:-1]
        return ended, self.tokenizer.decode(output_ids), output_ids

    def _greedy_completion(self, prompt: str, max_new_tokens: int) -> Tuple[bool, str]:
        ended, text, _ = self._generate_from_ids(
            self.tokenizer.encode(prompt),
            max_new_tokens,
        )
        return ended, text

    def _prefix_gap_query(self, state: CheckpointState, question: str) -> str:
        config = self.config.get("prefix_gap", {})
        context = QueryContext(state.qid, state.state_id, question, state.prefix_text)
        _, generated = self._greedy_completion(
            prefix_gap_prompt(context),
            int(config.get("max_new_tokens", 24)),
        )
        query = generated.splitlines()[0].strip().strip('"')
        return query

    def build_candidates(self, state: CheckpointState, question: str) -> List[QueryCandidate]:
        enabled = set(self.config.get("query_candidates", ["question", "etc_qfs", "prefix_gap_v1"]))
        etc_query = state.trace_metadata.get("etc_query") if "etc_qfs" in enabled else None
        gap_query = self._prefix_gap_query(state, question) if "prefix_gap_v1" in enabled else None
        context = QueryContext(
            qid=state.qid,
            state_id=state.state_id,
            question=question,
            prefix_text=state.prefix_text,
            etc_query=etc_query,
        )
        candidates = build_query_candidates(context, gap_query)
        return [candidate for candidate in candidates if candidate.source in enabled]

    def _score(self, answer: str, ground_truth: Any, ground_truth_id: Any) -> Dict[str, float]:
        em = self.dataset.exact_match_score(answer, ground_truth, ground_truth_id)
        f1 = self.dataset.f1_score(answer, ground_truth, ground_truth_id)
        return {
            "em": float(em["correct"]),
            "accuracy": float(em["correct"]),
            "f1": float(f1["f1"]),
            "precision": float(f1["precision"]),
            "recall": float(f1["recall"]),
        }

    def _make_rollout(
        self,
        state: CheckpointState,
        action_type: str,
        prediction: str,
        ground_truth: Any,
        ground_truth_id: Any,
        candidate: Optional[QueryCandidate] = None,
        documents: Optional[Sequence[RetrievedDocument]] = None,
        extra_generation_metadata: Optional[Dict[str, Any]] = None,
    ) -> ActionRollout:
        answer = extract_answer(prediction, self.extractor)
        alternative_extractions = {
            version: extract_answer(prediction, version)
            for version in self.sensitivity_extractors
            if version != self.extractor
        }
        alternative_scores = {
            version: self._score(value, ground_truth, ground_truth_id)
            for version, value in alternative_extractions.items()
        }
        candidate_id = candidate.candidate_id if candidate else None
        return ActionRollout(
            qid=state.qid,
            state_id=state.state_id,
            action_id=make_action_id(state.state_id, action_type, candidate_id),
            action_type=action_type,
            query_candidate_id=candidate_id,
            prediction=prediction,
            extracted_answer=answer,
            scores=self._score(answer, ground_truth, ground_truth_id),
            alternative_extractions=alternative_extractions,
            alternative_scores=alternative_scores,
            retrieved_documents=list(documents or []),
            generation_metadata={
                "rollout_version": "single_retrieval_rollout_v2",
                "max_answer_tokens": self.max_tokens,
                "post_action_retrieval": "disabled",
                **(extra_generation_metadata or {}),
            },
        )

    def rollout_state(
        self,
        state: CheckpointState,
        candidates: Sequence[QueryCandidate],
        question: str,
        demo_text: str,
        ground_truth: Any,
        ground_truth_id: Any,
        canonical_skip_prediction: str,
    ) -> List[ActionRollout]:
        actions = [
            self._make_rollout(
                state,
                "skip",
                canonical_skip_prediction,
                ground_truth,
                ground_truth_id,
                extra_generation_metadata={
                    "canonical_skip_reused": True,
                    "continuation_source": "no_retrieval_trajectory",
                },
            )
        ]
        topk = int(self.config.get("retrieve_topk", 3))
        for candidate in candidates:
            documents = self.retriever(candidate.text, topk=topk)
            prompt = demo_text + "\nContext:\n"
            for index, document in enumerate(documents, start=1):
                prompt += f"[{index}] {document.text}\n"
            prompt += "Answer in the same format as before.\n"
            prompt += "\nQuestion:" + question + "\nAnswer:"
            retrieval_input_ids = self.tokenizer.encode(prompt) + list(state.prefix_token_ids)
            _, regenerated, _ = self._generate_from_ids(
                retrieval_input_ids,
                self.max_tokens,
                stop_at_newline=True,
            )
            injected_sentence = self.get_top_sentence(regenerated).strip()
            injected_token_ids = self.tokenizer.encode(
                (" " if state.prefix_token_ids else "") + injected_sentence,
                add_special_tokens=False,
            )
            answer_token_ids = list(state.prefix_token_ids) + list(injected_token_ids)
            continuation_input_ids = self.tokenizer.encode(
                demo_text + "\nQuestion:" + question + "\nAnswer:"
            ) + answer_token_ids
            remaining = max(0, self.max_tokens - len(answer_token_ids))
            _, _, continuation_token_ids = self._generate_from_ids(
                continuation_input_ids,
                remaining,
            )
            prediction = self.tokenizer.decode(answer_token_ids + continuation_token_ids).strip()
            actions.append(
                self._make_rollout(
                    state,
                    "retrieve",
                    prediction,
                    ground_truth,
                    ground_truth_id,
                    candidate,
                    documents,
                    {
                        "retrieval_query_text": candidate.text,
                        "injected_sentence": injected_sentence,
                    },
                )
            )
        return actions

    def run_sample(self, entry: Dict[str, Any], sample_index: int) -> Dict[str, Any]:
        demo_text = "\n".join(item["case"] for item in entry["demo"])
        ground_truth_id = entry.get("answer_id")
        no_retrieval_prediction, no_retrieval_token_ids, states = self.collect_no_retrieval_trace(
            entry["qid"], sample_index, entry["question"], demo_text
        )
        candidate_rows: List[QueryCandidate] = []
        action_rows: List[ActionRollout] = []
        for state in states:
            candidates = self.build_candidates(state, entry["question"])
            candidate_rows.extend(candidates)
            action_rows.extend(
                self.rollout_state(
                    state,
                    candidates,
                    entry["question"],
                    demo_text,
                    entry["answer"],
                    ground_truth_id,
                    no_retrieval_prediction,
                )
            )
        no_retrieval_answer = extract_answer(no_retrieval_prediction, self.extractor)
        no_retrieval_alternative_extractions = {
            version: extract_answer(no_retrieval_prediction, version)
            for version in self.sensitivity_extractors
            if version != self.extractor
        }
        return {
            "bundle_version": "cura_sample_bundle_v2",
            "qid": entry["qid"],
            "sample_index": sample_index,
            "question": entry["question"],
            "ground_truth": entry["answer"],
            "ground_truth_id": ground_truth_id,
            "no_retrieval_prediction": no_retrieval_prediction,
            "no_retrieval_token_ids": no_retrieval_token_ids,
            "no_retrieval_extracted_answer": no_retrieval_answer,
            "no_retrieval_scores": self._score(no_retrieval_answer, entry["answer"], ground_truth_id),
            "no_retrieval_alternative_extractions": no_retrieval_alternative_extractions,
            "no_retrieval_alternative_scores": {
                version: self._score(value, entry["answer"], ground_truth_id)
                for version, value in no_retrieval_alternative_extractions.items()
            },
            "states": [to_dict(item) for item in states],
            "queries": [to_dict(item) for item in candidate_rows],
            "actions": [to_dict(item) for item in action_rows],
        }
