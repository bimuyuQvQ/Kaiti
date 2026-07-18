"""Versioned, JSON-safe data contracts for counterfactual ETC experiments."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


SCHEMA_VERSION = "cura_schema_v1"


def canonical_json(value: Any) -> str:
    """Serialize a JSON-compatible value deterministically."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(namespace: str, payload: Mapping[str, Any], length: int = 20) -> str:
    """Return a readable, stable identifier for an immutable payload."""

    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{namespace}_{digest[:length]}"


def to_dict(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {item.name: to_dict(getattr(value, item.name)) for item in dataclasses.fields(value)}
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class RetrievedDocument:
    document_id: str
    text: str
    score: float
    rank: int
    title: Optional[str] = None
    index_name: Optional[str] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryCandidate:
    qid: str
    state_id: str
    source: str
    text: str
    normalized_text: str
    candidate_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckpointState:
    qid: str
    sample_index: int
    checkpoint_index: int
    checkpoint_type: str
    prefix_text: str
    state_id: str
    token_index: Optional[int] = None
    etc_signal: Optional[float] = None
    features: Dict[str, Any] = field(default_factory=dict)
    trace_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionRollout:
    qid: str
    state_id: str
    action_id: str
    action_type: str
    prediction: str
    extracted_answer: str
    scores: Dict[str, float]
    status: str = "complete"
    query_candidate_id: Optional[str] = None
    retrieved_documents: List[RetrievedDocument] = field(default_factory=list)
    generation_metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    created_at_utc: str
    schema_version: str
    extractor_version: str
    git_commit: Optional[str]
    config_sha256: str
    config: Dict[str, Any]
    input_files: Dict[str, str]
    python_version: str
    platform: str
    package_version: str = "cura_research_v1"

