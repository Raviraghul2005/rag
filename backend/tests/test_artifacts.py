from __future__ import annotations

from unittest.mock import patch

from app.artifacts import ensure_artifacts


def test_skips_download_when_a_strategy_is_already_built(tmp_path):
    strategy_dir = tmp_path / "index" / "recursive_512"
    strategy_dir.mkdir(parents=True)
    (strategy_dir / "BUILD_COMPLETE").write_text("{}")

    with patch("app.artifacts.snapshot_download") as mock_download:
        ensure_artifacts(tmp_path, strategies=["recursive_512", "fixed_256_overlap_64"])

    mock_download.assert_not_called()


def test_downloads_when_no_strategy_is_built(tmp_path):
    with patch("app.artifacts.snapshot_download") as mock_download:
        ensure_artifacts(tmp_path, strategies=["recursive_512"], repo_id="user/repo")

    mock_download.assert_called_once_with(repo_id="user/repo", repo_type="dataset", local_dir=str(tmp_path))


def test_downloads_when_strategy_dir_exists_but_incomplete(tmp_path):
    strategy_dir = tmp_path / "index" / "recursive_512"
    strategy_dir.mkdir(parents=True)
    # No BUILD_COMPLETE marker — a partial/interrupted local build shouldn't count as "ready".

    with patch("app.artifacts.snapshot_download") as mock_download:
        ensure_artifacts(tmp_path, strategies=["recursive_512"])

    mock_download.assert_called_once()
