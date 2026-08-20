from __future__ import annotations

import logging
import os
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

# The corpus + built indexes are multi-GB and can't live in git (backend/.gitignore
# excludes data/) or be baked into the Docker image (no repo file to COPY). Deployment
# platforms with ephemeral container disks (Render, a fresh Railway build without a
# volume) need to pull them from somewhere durable at startup — spec §16.2's own
# reasoning for HF Spaces ("Space storage is wiped on rebuild: never treat container
# disk as durable") applies equally here, platform-agnostic. A Railway volume, if
# attached, works as a *cache* in front of this — ensure_artifacts() is idempotent and
# skips the download entirely once local_dir already has a built index.
DEFAULT_ARTIFACTS_REPO = os.environ.get("RAINGOA_ARTIFACTS_REPO", "RaviR2005/raingoa-artifacts")


def ensure_artifacts(
    local_dir: Path, strategies: list[str], repo_id: str = DEFAULT_ARTIFACTS_REPO
) -> None:
    """Pulls corpus + built indexes from the HF dataset repo into local_dir, unless at
    least one of `strategies` already has a completed local build (BUILD_COMPLETE
    marker — see scripts/build_index.py) — cheap idempotency check so a restart with a
    warm local disk or Railway volume doesn't re-download several GB for nothing.
    """
    index_dir = local_dir / "index"
    if any((index_dir / name / "BUILD_COMPLETE").exists() for name in strategies):
        logger.info("artifacts already present in %s, skipping download", local_dir)
        return

    logger.info("pulling artifacts from %s into %s", repo_id, local_dir)
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(local_dir))
    logger.info("artifact pull complete")
