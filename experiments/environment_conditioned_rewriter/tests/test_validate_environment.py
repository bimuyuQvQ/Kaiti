from pathlib import Path

from env_rewriter.validate_environment import inspect_model_cache, package_version


def test_known_package_is_available() -> None:
    assert package_version("pip")["available"] is True


def test_missing_model_cache(tmp_path: Path) -> None:
    report = inspect_model_cache(tmp_path, "owner/model")
    assert report["exists"] is False
    assert report["snapshot_count"] == 0
