"""ETC/CURA research layer.

This package is intentionally separate from the released ETC baseline.  Importing
it must not load a language model, CUDA, BEIR, or Elasticsearch.
"""

from .extractors import EXTRACTOR_VERSION, SENSITIVITY_EXTRACTOR_VERSION, extract_answer
from .schema import (
    ActionRollout,
    CheckpointState,
    QueryCandidate,
    RetrievedDocument,
    RunManifest,
)

__all__ = [
    "EXTRACTOR_VERSION",
    "SENSITIVITY_EXTRACTOR_VERSION",
    "ActionRollout",
    "CheckpointState",
    "QueryCandidate",
    "RetrievedDocument",
    "RunManifest",
    "extract_answer",
]
