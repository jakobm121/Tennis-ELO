from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from tennis_elo.config import ROOT_DIR


DEFAULT_API_BACKFILL = (
    "https://raw.githubusercontent.com/jakobm121/Ai/refs/heads/main/"
    "data/tle/tle_api_results_backfill.json"
)

DEFAULT_METADATA = (
    ROOT_DIR
    / "data"
    / "tle"
    / "mappings"
    / "api_tournament_metadata.json"
)

DEFAULT_OUTPUT = (
    ROOT_DIR
    / "data"
    / "tle"
    / "source"
    / "api"
    / "tle_api_results_backfill_enriched.json"
)

DEFAULT_REPORT = (
    ROOT_DIR
    / "data"
    / "tle"
    / "reports"
    / "tle_enrich_api_results_report.json"
)

VALID_GENDERS = {"men", "women"}
VALID_LEVELS = {"main_tour", "challenger", "itf", "qualifying"}
VALID_SURFACES = {"hard", "clay", "grass", "carpet"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def lower_text(value: Any) -> str:
    return clean_text(value).lower()


def load_json(source: str | Path) -> Any:
    source_text = str(source)

    if source_text.startswith(("http://", "https://")):
        request = Request(
            source_text,
            headers={"User-Agent": "Tennis-ELO API result enricher/1.0"},
        )
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))

    return json.loads(Path(source_text).read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def get_tournament_mapping(
    metadata: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    tournaments = metadata.get("tournaments") or {}

    if not isinstance(tournaments, dict):
        raise RuntimeError(
            "Metadata nima pravilnega polja 'tournaments'."
        )

    result = {}

    for key, value in tournaments.items():
        if isinstance(value, dict):
            result[str(key)] = value

    return result


def use_mapping_value(
    original: str,
    mapped: str,
    allowed: set[str],
) -> tuple[str, str]:
    original = lower_text(original)
    mapped = lower_text(mapped)

    if mapped in allowed:
        return mapped, "metadata"

    if original in allowed:
        return original, "api_backfill"

    return "unknown", "unknown"


def enrich_match(
    match: dict[str, Any],
    mapping: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    enriched = dict(match)

    tournament_key = str(match.get("tournament_key") or "")
    entry = mapping.get(tournament_key, {})

    surface, surface_source = use_mapping_value(
        match.get("surface"),
        entry.get("surface"),
        VALID_SURFACES,
    )

    gender, gender_source = use_mapping_value(
        match.get("gender"),
        entry.get("gender"),
        VALID_GENDERS,
    )

    # Qualifying iz backfilla ima prednost, ker je to loÄen TLE pool
    # in ga ne smemo po pomoti prestaviti v main/challenger/itf.
    if bool(match.get("qualification")):
        level = "qualifying"
        level_source = "qualification"
    else:
        level, level_source = use_mapping_value(
            match.get("tour_level"),
            entry.get("tour_level"),
            VALID_LEVELS,
        )

    enriched["surface"] = surface
    enriched["surface_source"] = surface_source
    enriched["gender"] = gender
    enriched["gender_source"] = gender_source
    enriched["tour_level"] = level
    enriched["tour_level_source"] = level_source

    ready_reasons = []

    if gender not in VALID_GENDERS:
        ready_reasons.append("gender_unknown")

    if surface not in VALID_SURFACES:
        ready_reasons.append("surface_unknown")

    if level not in VALID_LEVELS:
        ready_reasons.append("tour_level_unknown")

    for field in ("date", "winner", "loser", "final_result"):
        if not clean_text(enriched.get(field)):
            ready_reasons.append(f"{field}_missing")

    if enriched.get("winner") == enriched.get("loser"):
        ready_reasons.append("winner_equals_loser")

    enriched["ready_for_tle"] = not ready_reasons
    enriched["not_ready_reasons"] = ready_reasons

    enriched["metadata_match"] = {
        "tournament_key": tournament_key,
        "metadata_status": entry.get("status"),
        "metadata_method": entry.get("match_method"),
        "metadata_confidence": entry.get("confidence"),
        "metadata_needs_review": entry.get("needs_review"),
        "metadata_name": entry.get("name"),
    }

    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Uporabi api_tournament_metadata.json na API backfill "
            "in izdela enriched source datoteko za TLE import."
        )
    )

    parser.add_argument(
        "--api-backfill",
        default=DEFAULT_API_BACKFILL,
        help="Lokalna pot ali URL do tle_api_results_backfill.json.",
    )

    parser.add_argument(
        "--metadata",
        default=str(DEFAULT_METADATA),
        help="Pot do data/tle/mappings/api_tournament_metadata.json.",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
    )

    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
    )

    args = parser.parse_args()

    api_payload = load_json(args.api_backfill)
    metadata_payload = load_json(args.metadata)

    matches = api_payload.get("matches") or []

    if not isinstance(matches, list):
        raise RuntimeError(
            "API backfill nima pravilnega polja 'matches'."
        )

    mapping = get_tournament_mapping(metadata_payload)

    enriched_matches = [
        enrich_match(match, mapping)
        for match in matches
    ]

    ready = [
        match
        for match in enriched_matches
        if match.get("ready_for_tle")
    ]

    not_ready = [
        match
        for match in enriched_matches
        if not match.get("ready_for_tle")
    ]

    levels = Counter(
        match.get("tour_level") or "unknown"
        for match in enriched_matches
    )
    surfaces = Counter(
        match.get("surface") or "unknown"
        for match in enriched_matches
    )
    genders = Counter(
        match.get("gender") or "unknown"
        for match in enriched_matches
    )

    reasons = Counter(
        reason
        for match in not_ready
        for reason in match.get("not_ready_reasons", [])
    )

    not_ready_tournaments = Counter(
        (
            match.get("tournament_key"),
            match.get("tournament"),
            match.get("event_type"),
            tuple(match.get("not_ready_reasons", [])),
        )
        for match in not_ready
    )

    summary = {
        "matches_total": len(enriched_matches),
        "ready_for_tle": len(ready),
        "not_ready_for_tle": len(not_ready),
        "levels": dict(sorted(levels.items())),
        "surfaces": dict(sorted(surfaces.items())),
        "genders": dict(sorted(genders.items())),
        "not_ready_reasons": dict(sorted(reasons.items())),
        "metadata_tournaments": len(mapping),
        "source_api_backfill": str(args.api_backfill),
        "source_metadata": str(args.metadata),
    }

    output_payload = {
        "schema_version": 1,
        "generated_at": now_iso(),
        "summary": summary,
        "matches": enriched_matches,
    }

    report_payload = {
        "generated_at": output_payload["generated_at"],
        "summary": summary,
        "not_ready_tournaments": [
            {
                "matches": count,
                "tournament_key": key[0],
                "tournament": key[1],
                "event_type": key[2],
                "reasons": list(key[3]),
            }
            for key, count in not_ready_tournaments.most_common()
        ],
        "not_ready_matches_sample": not_ready[:100],
    }

    save_json(Path(args.output), output_payload)
    save_json(Path(args.report), report_payload)

    print("TLE ENRICH API RESULTS DONE")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
