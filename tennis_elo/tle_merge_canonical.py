from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tennis_elo.config import ROOT_DIR


DEFAULT_SACKMANN_MANIFEST = (
    ROOT_DIR
    / "data"
    / "tle"
    / "processed"
    / "sackmann"
    / "tle_sackmann_manifest.json"
)

DEFAULT_API_MANIFEST = (
    ROOT_DIR
    / "data"
    / "tle"
    / "source"
    / "api"
    / "tle_api_matches_manifest.json"
)

DEFAULT_OUTPUT_DIR = (
    ROOT_DIR
    / "data"
    / "tle"
    / "processed"
    / "canonical"
)

DEFAULT_CANONICAL_MANIFEST = (
    ROOT_DIR
    / "data"
    / "tle"
    / "processed"
    / "canonical"
    / "tle_matches_manifest.json"
)

DEFAULT_REPORT = (
    ROOT_DIR
    / "data"
    / "tle"
    / "reports"
    / "tle_merge_canonical_report.json"
)

SOURCE_PRIORITY = {
    "sackmann": 1,
    "api_tennis": 2,
}

VALID_SURFACES = {"hard", "clay", "grass", "carpet"}
UNKNOWN_SURFACE_ALLOWED_LEVELS = {"itf", "qualifying"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", clean_text(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON object: {path}")

    return payload


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    temporary.replace(path)


def iter_manifest_matches(
    manifest_path: Path,
    default_source: str,
):
    manifest = load_json(manifest_path)
    year_files = manifest.get("year_files") or []

    if not isinstance(year_files, list):
        raise ValueError(f"Invalid year_files in manifest: {manifest_path}")

    for item in sorted(year_files, key=lambda row: int(row.get("year", 0))):
        relative = item.get("path")
        if not relative:
            continue

        path = Path(relative)
        if not path.is_absolute():
            path = ROOT_DIR / path

        if not path.exists():
            raise FileNotFoundError(f"Missing TLE source file: {path}")

        for match in read_jsonl_gz(path):
            if not isinstance(match, dict):
                continue

            if not clean_text(match.get("source")):
                match["source"] = default_source

            yield match


def player_name(match: dict[str, Any], side: str) -> str:
    player = match.get(side) or {}
    if not isinstance(player, dict):
        return ""
    return clean_text(player.get("name"))


def source_priority(match: dict[str, Any]) -> int:
    return SOURCE_PRIORITY.get(clean_text(match.get("source")), 99)


def match_fingerprint(match: dict[str, Any]) -> str:
    winner = normalize_key(player_name(match, "winner"))
    loser = normalize_key(player_name(match, "loser"))
    players = sorted([winner, loser])

    tournament = match.get("tournament") or {}
    if not isinstance(tournament, dict):
        tournament = {}

    components = [
        clean_text(match.get("date")),
        clean_text(match.get("gender")).lower(),
        clean_text(match.get("tour_level")).lower(),
        normalize_key(tournament.get("name")),
        normalize_key(match.get("round")),
        *players,
    ]

    raw = "|".join(components)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"fp_{digest}"


def is_better_match(
    challenger: dict[str, Any],
    incumbent: dict[str, Any],
) -> bool:
    challenger_priority = source_priority(challenger)
    incumbent_priority = source_priority(incumbent)

    if challenger_priority != incumbent_priority:
        return challenger_priority < incumbent_priority

    # Äe je isti source, obdrÅ¾i zapis z znano podlago.
    challenger_surface = clean_text(
        (challenger.get("tournament") or {}).get("surface")
    ).lower()
    incumbent_surface = clean_text(
        (incumbent.get("tournament") or {}).get("surface")
    ).lower()

    if (
        challenger_surface in VALID_SURFACES
        and incumbent_surface not in VALID_SURFACES
    ):
        return True

    return False


def valid_for_canonical(match: dict[str, Any]) -> tuple[bool, str | None]:
    if not match.get("ready_for_tle"):
        return False, "not_ready_for_tle"

    gender = clean_text(match.get("gender")).lower()
    if gender not in {"men", "women"}:
        return False, "invalid_gender"

    level = clean_text(match.get("tour_level")).lower()
    if level not in {
        "grand_slam",
        "atp_wta",
        "challenger",
        "qualifying",
        "itf",
    }:
        return False, "invalid_level"

    tournament = match.get("tournament") or {}
    if not isinstance(tournament, dict):
        return False, "invalid_tournament"

    surface = clean_text(tournament.get("surface")).lower()

    if surface not in VALID_SURFACES:
        if level not in UNKNOWN_SURFACE_ALLOWED_LEVELS:
            return False, "unknown_surface_not_allowed_for_level"

    if not clean_text(match.get("date")):
        return False, "missing_date"

    if not player_name(match, "winner") or not player_name(match, "loser"):
        return False, "missing_player"

    if player_name(match, "winner") == player_name(match, "loser"):
        return False, "winner_equals_loser"

    return True, None


def add_match(
    match: dict[str, Any],
    selected_by_fingerprint: dict[str, dict[str, Any]],
    selected_id_by_fingerprint: dict[str, str],
    counters: Counter,
    replaced: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> None:
    counters[f"input_source_{clean_text(match.get('source')) or 'unknown'}"] += 1

    ok, reason = valid_for_canonical(match)
    if not ok:
        counters[f"skipped_{reason}"] += 1
        skipped.append(
            {
                "tle_match_id": match.get("tle_match_id"),
                "source": match.get("source"),
                "date": match.get("date"),
                "reason": reason,
            }
        )
        return

    fingerprint = match_fingerprint(match)
    tle_match_id = clean_text(match.get("tle_match_id")) or fingerprint

    incumbent = selected_by_fingerprint.get(fingerprint)

    if incumbent is None:
        selected_by_fingerprint[fingerprint] = match
        selected_id_by_fingerprint[fingerprint] = tle_match_id
        counters[f"kept_source_{clean_text(match.get('source')) or 'unknown'}"] += 1
        return

    if is_better_match(match, incumbent):
        replaced.append(
            {
                "fingerprint": fingerprint,
                "kept_source": match.get("source"),
                "replaced_source": incumbent.get("source"),
                "date": match.get("date"),
                "kept_id": match.get("tle_match_id"),
                "replaced_id": incumbent.get("tle_match_id"),
            }
        )

        counters[
            f"replaced_{clean_text(incumbent.get('source'))}_with_{clean_text(match.get('source'))}"
        ] += 1
        counters[f"kept_source_{clean_text(match.get('source')) or 'unknown'}"] += 1
        counters[f"duplicate_source_{clean_text(incumbent.get('source')) or 'unknown'}"] += 1
        selected_by_fingerprint[fingerprint] = match
        selected_id_by_fingerprint[fingerprint] = tle_match_id
    else:
        counters[f"duplicate_source_{clean_text(match.get('source')) or 'unknown'}"] += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "ZdruÅ¾i Sackmann source in API overlay v canonical TLE bazo. "
            "Pri prekrivanju ima Sackmann prednost pred API."
        )
    )

    parser.add_argument(
        "--sackmann-manifest",
        default=str(DEFAULT_SACKMANN_MANIFEST),
    )

    parser.add_argument(
        "--api-manifest",
        default=str(DEFAULT_API_MANIFEST),
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )

    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_CANONICAL_MANIFEST),
    )

    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
    )

    args = parser.parse_args()

    sackmann_manifest = Path(args.sackmann_manifest)
    api_manifest = Path(args.api_manifest)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)
    report_path = Path(args.report)

    selected_by_fingerprint: dict[str, dict[str, Any]] = {}
    selected_id_by_fingerprint: dict[str, str] = {}
    counters: Counter = Counter()
    replaced: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for match in iter_manifest_matches(sackmann_manifest, "sackmann"):
        add_match(
            match,
            selected_by_fingerprint,
            selected_id_by_fingerprint,
            counters,
            replaced,
            skipped,
        )

    if api_manifest.exists():
        for match in iter_manifest_matches(api_manifest, "api_tennis"):
            add_match(
                match,
                selected_by_fingerprint,
                selected_id_by_fingerprint,
                counters,
                replaced,
                skipped,
            )
    else:
        counters["api_manifest_missing"] += 1

    matches = list(selected_by_fingerprint.values())

    matches.sort(
        key=lambda row: (
            clean_text(row.get("date")),
            clean_text(row.get("gender")),
            clean_text(row.get("tour_level")),
            clean_text((row.get("tournament") or {}).get("name")),
            clean_text(row.get("round")),
            clean_text(row.get("tle_match_id")),
        )
    )

    by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
    levels = Counter()
    surfaces = Counter()
    genders = Counter()
    sources = Counter()

    for match in matches:
        year = int(clean_text(match.get("date"))[:4])
        by_year[year].append(match)

        level = clean_text(match.get("tour_level")).lower()
        surface = clean_text((match.get("tournament") or {}).get("surface")).lower()
        gender = clean_text(match.get("gender")).lower()
        source = clean_text(match.get("source")) or "unknown"

        levels[level] += 1
        surfaces[surface or "unknown"] += 1
        genders[gender or "unknown"] += 1
        sources[source] += 1

    year_files = []

    for year, rows in sorted(by_year.items()):
        relative_path = (
            Path("data")
            / "tle"
            / "processed"
            / "canonical"
            / f"tle_matches_{year}.jsonl.gz"
        )
        output_path = ROOT_DIR / relative_path
        write_jsonl_gz(output_path, rows)

        year_files.append(
            {
                "year": year,
                "path": str(relative_path),
                "matches": len(rows),
                "created_at": now_iso(),
            }
        )

    generated_at = now_iso()

    canonical_manifest = {
        "schema_version": 1,
        "source": "canonical",
        "generated_at": generated_at,
        "source_priority": SOURCE_PRIORITY,
        "inputs": {
            "sackmann_manifest": str(sackmann_manifest.relative_to(ROOT_DIR))
            if sackmann_manifest.is_absolute()
            and sackmann_manifest.is_relative_to(ROOT_DIR)
            else str(sackmann_manifest),
            "api_manifest": str(api_manifest.relative_to(ROOT_DIR))
            if api_manifest.is_absolute()
            and api_manifest.is_relative_to(ROOT_DIR)
            else str(api_manifest),
        },
        "year_files": year_files,
        "matches_total": sum(item["matches"] for item in year_files),
        "surface_policy": {
            "unknown_surface_allowed_for_levels": sorted(
                UNKNOWN_SURFACE_ALLOWED_LEVELS
            ),
            "rating_behavior": (
                "unknown surface matches update overall layers only; "
                "surface layers are skipped by tle_build_ratings.py"
            ),
        },
    }

    report = {
        "generated_at": generated_at,
        "summary": {
            "canonical_matches": canonical_manifest["matches_total"],
            "year_files": year_files,
            "levels": dict(sorted(levels.items())),
            "surfaces": dict(sorted(surfaces.items())),
            "genders": dict(sorted(genders.items())),
            "sources": dict(sorted(sources.items())),
            **dict(counters),
        },
        "replaced_sample": replaced[:300],
        "replaced_count": len(replaced),
        "skipped_sample": skipped[:300],
        "skipped_count": len(skipped),
    }

    save_json(manifest_path, canonical_manifest)
    save_json(report_path, report)

    print("TLE CANONICAL MERGE DONE")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"Manifest: {manifest_path}")
    print(f"Report:   {report_path}")


if __name__ == "__main__":
    main()
