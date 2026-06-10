from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from tennis_elo.config import ROOT_DIR


DEFAULT_ENRICHED = (
    ROOT_DIR
    / "data"
    / "tle"
    / "source"
    / "api"
    / "tle_api_results_backfill_enriched.json"
)

DEFAULT_SACKMANN_MANIFEST = (
    ROOT_DIR
    / "data"
    / "tle"
    / "processed"
    / "sackmann"
    / "tle_sackmann_manifest.json"
)

DEFAULT_OUTPUT_DIR = (
    ROOT_DIR
    / "data"
    / "tle"
    / "source"
    / "api"
)

DEFAULT_MANIFEST = (
    ROOT_DIR
    / "data"
    / "tle"
    / "source"
    / "api"
    / "tle_api_matches_manifest.json"
)

DEFAULT_REPORT = (
    ROOT_DIR
    / "data"
    / "tle"
    / "reports"
    / "tle_import_api_results_report.json"
)

VALID_GENDERS = {"men", "women"}
VALID_LEVELS = {"main_tour", "challenger", "itf", "qualifying"}
VALID_SURFACES = {"hard", "clay", "grass", "carpet"}
LEVELS_ALLOW_UNKNOWN_SURFACE = {"itf", "qualifying"}

LEVEL_FOR_TLE = {
    "main_tour": "atp_wta",
    "challenger": "challenger",
    "itf": "itf",
    "qualifying": "qualifying",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def lower_text(value: Any) -> str:
    return clean_text(value).lower()


def normalize_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", clean_text(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def parse_date(value: Any) -> date | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def load_json(source: str | Path) -> Any:
    source_text = str(source)

    if source_text.startswith(("http://", "https://")):
        request = Request(
            source_text,
            headers={"User-Agent": "Tennis-ELO TLE API importer/2.0"},
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


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    temporary.replace(path)


def read_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def latest_sackmann_date(manifest_path: Path) -> date | None:
    manifest = load_json(manifest_path)
    year_files = manifest.get("year_files") or []

    latest: date | None = None

    for file_info in year_files:
        relative_path = file_info.get("path")
        if not relative_path:
            continue

        path = ROOT_DIR / relative_path
        if not path.exists():
            continue

        for match in read_jsonl_gz(path):
            match_date = parse_date(match.get("date"))
            if match_date is None:
                continue
            if latest is None or match_date > latest:
                latest = match_date

    return latest


def safe_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def api_player_identity(
    api_player_key: Any,
    name: str,
) -> dict[str, Any]:
    key = safe_int(api_player_key)

    return {
        "sackmann_player_id": None,
        "api_player_key": key,
        "name": name,
        "player_key": (
            f"api:{key}"
            if key is not None
            else f"name:{normalize_key(name)}"
        ),
    }


def make_tle_match_id(match: dict[str, Any]) -> str:
    event_key = safe_int(match.get("event_key"))
    if event_key is not None:
        return f"api_tennis_{event_key}"

    raw = "|".join(
        [
            clean_text(match.get("date")),
            clean_text(match.get("tournament_key")),
            normalize_key(match.get("tournament")),
            normalize_key(match.get("winner")),
            normalize_key(match.get("loser")),
            normalize_key(match.get("final_result")),
        ]
    )

    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    return f"api_tennis_{digest}"


def normalize_round(value: Any) -> str:
    text = clean_text(value)

    mapping = {
        "final": "F",
        "semi-finals": "SF",
        "semifinals": "SF",
        "semi final": "SF",
        "quarter-finals": "QF",
        "quarterfinals": "QF",
        "quarter final": "QF",
        "1/2-finals": "SF",
        "1/4-finals": "QF",
        "1/8-finals": "R16",
        "1/16-finals": "R32",
        "1/32-finals": "R64",
        "1/64-finals": "R128",
    }

    lower = text.lower()
    return mapping.get(lower, text or "unknown")


def score_from_api(match: dict[str, Any]) -> str:
    final_result = clean_text(match.get("final_result"))
    scores = match.get("scores") or []

    if isinstance(scores, list) and scores:
        parts = []

        for item in scores:
            if not isinstance(item, dict):
                continue

            first = (
                item.get("score_first")
                or item.get("first_score")
                or item.get("score_home")
                or item.get("home_score")
            )

            second = (
                item.get("score_second")
                or item.get("second_score")
                or item.get("score_away")
                or item.get("away_score")
            )

            if first not in {None, ""} and second not in {None, ""}:
                parts.append(f"{first}-{second}")

        if parts:
            return " ".join(parts)

    return final_result


def validate_surface_for_level(level: str, surface: str) -> str:
    if surface in VALID_SURFACES:
        return surface

    if level in LEVELS_ALLOW_UNKNOWN_SURFACE:
        return "unknown"

    raise ValueError(
        f"surface ni znan za level={level}; za ta level je obvezen"
    )


def convert_to_tle_match(match: dict[str, Any]) -> dict[str, Any]:
    gender = lower_text(match.get("gender"))
    level = lower_text(match.get("tour_level"))
    surface = lower_text(match.get("surface"))

    if gender not in VALID_GENDERS:
        raise ValueError("gender ni znan")
    if level not in VALID_LEVELS:
        raise ValueError("tour_level ni znan")

    surface = validate_surface_for_level(level, surface)

    winner_name = clean_text(match.get("winner"))
    loser_name = clean_text(match.get("loser"))

    winner_key = (
        match.get("first_player_key")
        if match.get("winner_side") == "player_1"
        else match.get("second_player_key")
    )

    loser_key = (
        match.get("second_player_key")
        if match.get("winner_side") == "player_1"
        else match.get("first_player_key")
    )

    api_date = parse_date(match.get("date"))
    if api_date is None:
        raise ValueError("date ni veljaven")

    tournament_key = safe_int(match.get("tournament_key"))
    tournament_name = clean_text(match.get("tournament"))
    event_key = safe_int(match.get("event_key"))

    tle_match = {
        "tle_match_id": make_tle_match_id(match),
        "date": api_date.isoformat(),
        "gender": gender,
        "tour_level": LEVEL_FOR_TLE[level],
        "source": "api_tennis",
        "source_event_key": event_key,
        "tournament": {
            "id": (
                f"api_tournament:{tournament_key}"
                if tournament_key is not None
                else f"api_tournament:{normalize_key(tournament_name)}"
            ),
            "api_tournament_key": tournament_key,
            "name": tournament_name,
            "surface": surface,
            "indoor": bool(match.get("indoor")),
        },
        "round": normalize_round(match.get("round")),
        "best_of": int(match.get("best_of") or 3),
        "score": score_from_api(match),
        "retired": False,
        "winner": api_player_identity(winner_key, winner_name),
        "loser": api_player_identity(loser_key, loser_name),
        "ready_for_tle": True,
        "api": {
            "event_id": clean_text(match.get("event_id")),
            "event_type": clean_text(match.get("event_type")),
            "status": clean_text(match.get("status")),
            "qualification": bool(match.get("qualification")),
            "is_grand_slam": bool(match.get("is_grand_slam")),
            "surface_source": clean_text(match.get("surface_source")),
            "surface_policy": clean_text(match.get("surface_policy")),
            "gender_source": clean_text(match.get("gender_source")),
            "tour_level_source": clean_text(match.get("tour_level_source")),
            "metadata_match": match.get("metadata_match") or {},
        },
    }

    return tle_match


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pretvori enriched API rezultate v TLE JSONL.GZ source "
            "datoteke. ITF/qualifying lahko imata surface=unknown; "
            "rating builder bo pri unknown surface posodobil samo overall Elo."
        )
    )

    parser.add_argument(
        "--enriched",
        default=str(DEFAULT_ENRICHED),
        help="Pot ali URL do tle_api_results_backfill_enriched.json.",
    )

    parser.add_argument(
        "--sackmann-manifest",
        default=str(DEFAULT_SACKMANN_MANIFEST),
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )

    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
    )

    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
    )

    parser.add_argument(
        "--include-before-latest-sackmann",
        action="store_true",
        help=(
            "Privzeto importamo samo tekme po latest_sackmann_date. "
            "Ta opcija dovoli tudi starejÅ¡e tekme."
        ),
    )

    args = parser.parse_args()

    enriched_payload = load_json(args.enriched)
    enriched_matches = enriched_payload.get("matches") or []

    if not isinstance(enriched_matches, list):
        raise RuntimeError(
            "Enriched API datoteka nima pravilnega polja 'matches'."
        )

    latest = latest_sackmann_date(Path(args.sackmann_manifest))

    converted_by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
    skipped: list[dict[str, Any]] = []
    counters = Counter()
    seen_ids: set[str] = set()

    for match in enriched_matches:
        counters["input_matches"] += 1

        match_date = parse_date(match.get("date"))
        if match_date is None:
            counters["skipped_bad_date"] += 1
            skipped.append(
                {
                    "event_key": match.get("event_key"),
                    "reason": "bad_date",
                    "date": match.get("date"),
                }
            )
            continue

        if (
            latest is not None
            and match_date <= latest
            and not args.include_before_latest_sackmann
        ):
            counters["skipped_not_after_latest_sackmann"] += 1
            continue

        if not match.get("ready_for_tle"):
            counters["skipped_not_ready"] += 1
            skipped.append(
                {
                    "event_key": match.get("event_key"),
                    "date": match.get("date"),
                    "tournament": match.get("tournament"),
                    "reason": "not_ready_for_tle",
                    "not_ready_reasons": match.get("not_ready_reasons"),
                }
            )
            continue

        try:
            tle_match = convert_to_tle_match(match)
        except ValueError as exc:
            counters["skipped_conversion_error"] += 1
            skipped.append(
                {
                    "event_key": match.get("event_key"),
                    "date": match.get("date"),
                    "tournament": match.get("tournament"),
                    "reason": str(exc),
                }
            )
            continue

        tle_id = tle_match["tle_match_id"]
        if tle_id in seen_ids:
            counters["skipped_duplicate_in_input"] += 1
            continue

        seen_ids.add(tle_id)
        converted_by_year[match_date.year].append(tle_match)
        counters["imported"] += 1

        surface = tle_match["tournament"]["surface"]
        if surface == "unknown":
            counters["imported_unknown_surface_overall_only"] += 1

    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)
    report_path = Path(args.report)

    year_files = []

    for year, rows in sorted(converted_by_year.items()):
        rows.sort(
            key=lambda row: (
                row["date"],
                row["tournament"]["name"],
                row["round"],
                row["tle_match_id"],
            )
        )

        relative_path = Path("data") / "tle" / "source" / "api" / (
            f"tle_api_matches_{year}.jsonl.gz"
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

    levels = Counter()
    surfaces = Counter()
    genders = Counter()

    for rows in converted_by_year.values():
        for row in rows:
            levels[row["tour_level"]] += 1
            surfaces[row["tournament"]["surface"]] += 1
            genders[row["gender"]] += 1

    output_manifest = {
        "schema_version": 1,
        "source": "api_tennis",
        "generated_at": now_iso(),
        "input": str(args.enriched),
        "latest_sackmann_date": (
            latest.isoformat()
            if latest is not None
            else None
        ),
        "include_before_latest_sackmann": (
            args.include_before_latest_sackmann
        ),
        "year_files": year_files,
        "matches_total": sum(item["matches"] for item in year_files),
        "surface_policy": {
            "unknown_surface_allowed_for_levels": sorted(
                LEVELS_ALLOW_UNKNOWN_SURFACE
            ),
            "rating_behavior": (
                "unknown surface matches update global overall and "
                "level overall only; they do not update surface layers"
            ),
        },
    }

    report = {
        "generated_at": output_manifest["generated_at"],
        "summary": {
            **dict(counters),
            "latest_sackmann_date": output_manifest[
                "latest_sackmann_date"
            ],
            "year_files": year_files,
            "levels": dict(sorted(levels.items())),
            "surfaces": dict(sorted(surfaces.items())),
            "genders": dict(sorted(genders.items())),
        },
        "skipped_sample": skipped[:300],
        "skipped_count": len(skipped),
    }

    save_json(manifest_path, output_manifest)
    save_json(report_path, report)

    print("TLE IMPORT API RESULTS DONE")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"Manifest: {manifest_path}")
    print(f"Report:   {report_path}")


if __name__ == "__main__":
    main()
