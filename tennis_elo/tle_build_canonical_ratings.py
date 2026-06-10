from __future__ import annotations

import os
from pathlib import Path

from tennis_elo.config import ROOT_DIR
from tennis_elo import tle_build_ratings


CANONICAL_MANIFEST = (
    ROOT_DIR
    / "data"
    / "tle"
    / "processed"
    / "canonical"
    / "tle_matches_manifest.json"
)


def main() -> None:
    manifest_override = os.getenv("TLE_RATINGS_MANIFEST")

    manifest_path = (
        Path(manifest_override)
        if manifest_override
        else CANONICAL_MANIFEST
    )

    if not manifest_path.is_absolute():
        manifest_path = ROOT_DIR / manifest_path

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing canonical TLE manifest: {manifest_path}"
        )

    # Reuse existing, tested rating builder, only switch its input manifest.
    tle_build_ratings.MANIFEST_FILE = manifest_path
    tle_build_ratings.main()


if __name__ == "__main__":
    main()
