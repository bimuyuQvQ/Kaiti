"""Reproducibility manifests without environment-variable or secret capture."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .extractors import EXTRACTOR_VERSION
from .schema import RunManifest, SCHEMA_VERSION, canonical_json, stable_id


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit(repo_root: str | Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def build_manifest(
    config: Dict[str, Any],
    input_paths: Iterable[str | Path],
    repo_root: str | Path,
    created_at_utc: Optional[str] = None,
) -> RunManifest:
    config_hash = hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()
    input_files = {str(Path(path).resolve()): sha256_file(path) for path in input_paths}
    immutable = {
        "schema_version": SCHEMA_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "config_sha256": config_hash,
        "input_files": input_files,
    }
    return RunManifest(
        run_id=stable_id("run", immutable),
        created_at_utc=created_at_utc or datetime.now(timezone.utc).isoformat(),
        schema_version=SCHEMA_VERSION,
        extractor_version=EXTRACTOR_VERSION,
        git_commit=current_git_commit(repo_root),
        config_sha256=config_hash,
        config=config,
        input_files=input_files,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
    )

