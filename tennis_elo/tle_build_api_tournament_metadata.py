from __future__ import annotations

import argparse
import gzip
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from tennis_elo.config import ROOT_DIR


DEFAULT_API_REPORT = (
    "https://raw.githubusercontent.com/jakobm121/Ai/refs/heads/main/"
    "data/tle/tle_api_results_backfill_report.json"
)

DEFAULT_SACKMANN_MANIFEST = (
    ROOT_DIR
    / "data"
    / "tle"
    / "processed"
    / "sackmann"
    / "tle_sackmann_manifest.json"
)

DEFAULT_OUTPUT = (
    ROOT_DIR
    / "data"
    / "tle"
    / "mappings"
    / "api_tournament_metadata.json"
)

DEFAULT_REPORT = (
    ROOT_DIR
    / "data"
    / "tle"
    / "reports"
    / "tle_build_api_tournament_metadata_report.json"
)

VALID_SURFACES = {"hard", "clay", "grass", "carpet"}
VALID_GENDERS = {"men", "women"}
VALID_LEVELS = {
    "main_tour",
    "challenger",
    "itf",
    "qualifying",
}

GENERIC_TOKENS = {
    "atp",
    "wta",
    "itf",
    "challenger",
    "challenge",
    "tennis",
    "open",
    "singles",
    "men",
    "women",
    "male",
    "female",
    "qualification",
    "qualifying",
    "qualifier",
    "main",
    "tour",
}

LEVEL_EQUIVALENTS = {
    "atp_wta": "main_tour",
    "grand_slam": "main_tour",
    "main_tour": "main_tour",
    "challenger": "challenger",
    "itf": "itf",
    "qualifying": "qualifying",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def ascii_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", clean_text(value))
    return "".join(char for char in text if not unicodedata.combining(char))


def normalize_name(value: Any) -> str:
    text = ascii_text(value).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"\b(?:mens|womens)\b", " ", text)
    text = re.sub(r"\b(?:men|women)\b", " ", text)
    text = re.sub(r"\b(?:singles|doubles)\b", " ", text)
    text = re.sub(r"\b(?:qualification|qualifying|qualifier)\b", " ", text)
    text = re.sub(r"\b(?:atp|wta|itf|challenger)\b", " ", text)
    text = re.sub(r"\b[cmw]\s?(\d{2,3})\b", r" \1 ", text)
    text = re.sub(r"\b(?:20\d{2})\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def significant_tokens(value: Any) -> tuple[str, ...]:
    tokens = [
        token
        for token in normalize_name(value).split()
        if token not in GENERIC_TOKENS and len(token) > 1
    ]
    return tuple(sorted(set(tokens)))


def normalized_level(value: Any) -> str:
    return LEVEL_EQUIVALENTS.get(clean_text(value).lower(), "unknown")


def load_json(source: str | Path) -> Any:
    source_text = str(source)

    if source_text.startswith(("http://", "https://")):
        request = Request(
            source_text,
            headers={"User-Agent": "Tennis-ELO TLE metadata matcher/1.0"},
        )
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))

    path = Path(source_text)
    return json.loads(path.read_text(encoding="utf-8"))


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


def build_sackmann_catalog(
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_json(manifest_path)
    year_files = manifest.get("year_files") or []

    grouped: dict[
        tuple[str, str, str],
        dict[str, Any],
    ] = {}

    rows_seen = 0

    for file_info in year_files:
        relative = file_info.get("path")
        if not relative:
            continue

        path = ROOT_DIR / relative

        for match in read_jsonl_gz(path):
            rows_seen += 1

            tournament = match.get("tournament") or {}
            name = clean_text(tournament.get("name"))
            surface = clean_text(tournament.get("surface")).lower()
            gender = clean_text(match.get("gender")).lower()
            level = normalized_level(match.get("tour_level"))
            date_text = clean_text(match.get("date"))

            if not name:
                continue
            if surface not in VALID_SURFACES:
                continue
            if gender not in VALID_GENDERS:
                continue
            if level not in VALID_LEVELS:
                continue

            normalized = normalize_name(name)
            tokens = significant_tokens(name)

            if not normalized:
                continue

            key = (normalized, gender, level)

            if key not in grouped:
                grouped[key] = {
                    "normalized_name": normalized,
                    "tokens": tokens,
                    "gender": gender,
                    "tour_level": level,
                    "names": Counter(),
                    "surfaces": Counter(),
                    "years": Counter(),
                    "matches": 0,
                    "latest_date": "",
                }

            entry = grouped[key]
            entry["names"][name] += 1
            entry["surfaces"][surface] += 1
            entry["matches"] += 1

            if len(date_text) >= 4:
                entry["years"][date_text[:4]] += 1

            if date_text > entry["latest_date"]:
                entry["latest_date"] = date_text

    catalog: list[dict[str, Any]] = []

    for entry in grouped.values():
        names = entry["names"]
        surfaces = entry["surfaces"]
        years = entry["years"]

        total = sum(surfaces.values())
        top_surface, top_count = surfaces.most_common(1)[0]
        surface_share = top_count / total if total else 0.0

        catalog.append(
            {
                "name": names.most_common(1)[0][0],
                "normalized_name": entry["normalized_name"],
                "tokens": list(entry["tokens"]),
                "gender": entry["gender"],
                "tour_level": entry["tour_level"],
                "surface": top_surface,
                "surface_share": round(surface_share, 6),
                "surface_counts": dict(surfaces),
                "matches": entry["matches"],
                "year_counts": dict(years),
                "latest_date": entry["latest_date"],
            }
        )

    stats = {
        "rows_seen": rows_seen,
        "catalog_entries": len(catalog),
    }

    return catalog, stats


def extract_api_tournaments(
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    tournaments = report.get("tournaments") or []

    if not isinstance(tournaments, list):
        raise RuntimeError(
            "API report nima priÄakovanega polja 'tournaments'."
        )

    result = []

    for row in tournaments:
        key = row.get("tournament_key")
        name = clean_text(row.get("tournament"))
        event_type = clean_text(row.get("event_type"))
        levels = row.get("tour_levels") or {}
        genders = row.get("genders") or {}
        surfaces = row.get("surfaces") or {}

        level = (
            max(levels, key=levels.get)
            if levels
            else "unknown"
        )
        gender = (
            max(genders, key=genders.get)
            if genders
            else "unknown"
        )
        existing_surface = (
            max(surfaces, key=surfaces.get)
            if surfaces
            else "unknown"
        )

        result.append(
            {
                "tournament_key": key,
                "name": name,
                "event_type": event_type,
                "matches": int(row.get("matches") or 0),
                "gender": gender,
                "tour_level": level,
                "existing_surface": existing_surface,
                "normalized_name": normalize_name(name),
                "tokens": list(significant_tokens(name)),
            }
        )

    return result


def token_jaccard(
    left: list[str] | tuple[str, ...],
    right: list[str] | tuple[str, ...],
) -> float:
    left_set = set(left)
    right_set = set(right)

    if not left_set or not right_set:
        return 0.0

    return len(left_set & right_set) / len(left_set | right_set)


def candidate_score(
    api_row: dict[str, Any],
    sackmann_row: dict[str, Any],
) -> float:
    name_score = SequenceMatcher(
        None,
        api_row["normalized_name"],
        sackmann_row["normalized_name"],
    ).ratio()

    token_score = token_jaccard(
        api_row["tokens"],
        sackmann_row["tokens"],
    )

    score = 0.65 * name_score + 0.35 * token_score

    if api_row["normalized_name"] == sackmann_row["normalized_name"]:
        score += 0.20

    if (
        api_row["tokens"]
        and set(api_row["tokens"]) == set(sackmann_row["tokens"])
    ):
        score += 0.10

    if sackmann_row["surface_share"] < 0.80:
        score -= 0.10

    return min(score, 1.0)


def choose_match(
    api_row: dict[str, Any],
    catalog: list[dict[str, Any]],
) -> dict[str, Any]:
    existing_surface = api_row["existing_surface"]

    if existing_surface in VALID_SURFACES:
        return {
            "surface": existing_surface,
            "status": "already_known",
            "confidence": 1.0,
            "method": "api_existing_surface",
            "candidate": None,
            "alternatives": [],
        }

    gender = api_row["gender"]
    level = api_row["tour_level"]

    candidates = [
        row
        for row in catalog
        if row["gender"] == gender
        and (
            row["tour_level"] == level
            or level == "qualifying"
        )
    ]

    scored = sorted(
        (
            (candidate_score(api_row, candidate), candidate)
            for candidate in candidates
        ),
        key=lambda item: (
            item[0],
            item[1]["matches"],
            item[1]["latest_date"],
        ),
        reverse=True,
    )

    if not scored:
        return {
            "surface": "unknown",
            "status": "unmatched",
            "confidence": 0.0,
            "method": "no_candidates",
            "candidate": None,
            "alternatives": [],
        }

    best_score, best = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    margin = best_score - second_score

    exact = (
        api_row["normalized_name"]
        == best["normalized_name"]
    )

    same_tokens = bool(
        api_row["tokens"]
        and set(api_row["tokens"]) == set(best["tokens"])
    )

    stable_surface = best["surface_share"] >= 0.90

    auto_accept = (
        stable_surface
        and (
            exact
            or same_tokens
            or (
                best_score >= 0.88
                and margin >= 0.08
            )
        )
    )

    method = (
        "exact_name_gender_level"
        if exact
        else (
            "exact_tokens_gender_level"
            if same_tokens
            else "fuzzy_name_gender_level"
        )
    )

    alternatives = [
        {
            "score": round(score, 6),
            "name": row["name"],
            "gender": row["gender"],
            "tour_level": row["tour_level"],
            "surface": row["surface"],
            "surface_share": row["surface_share"],
            "matches": row["matches"],
            "latest_date": row["latest_date"],
        }
        for score, row in scored[:5]
    ]

    return {
        "surface": best["surface"] if auto_accept else "unknown",
        "status": "matched" if auto_accept else "review",
        "confidence": round(best_score, 6),
        "margin": round(margin, 6),
        "method": method,
        "candidate": {
            "name": best["name"],
            "normalized_name": best["normalized_name"],
            "gender": best["gender"],
            "tour_level": best["tour_level"],
            "surface": best["surface"],
            "surface_share": best["surface_share"],
            "surface_counts": best["surface_counts"],
            "matches": best["matches"],
            "year_counts": best["year_counts"],
            "latest_date": best["latest_date"],
        },
        "alternatives": alternatives,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Zgradi API tournament_key -> surface mapping iz "
            "Sackmann TLE zgodovine."
        )
    )

    parser.add_argument(
        "--api-report",
        default=DEFAULT_API_REPORT,
        help="Lokalna pot ali URL do API backfill reporta.",
    )

    parser.add_argument(
        "--sackmann-manifest",
        default=str(DEFAULT_SACKMANN_MANIFEST),
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

    manifest_path = Path(args.sackmann_manifest)
    output_path = Path(args.output)
    report_path = Path(args.report)

    api_report = load_json(args.api_report)
    api_tournaments = extract_api_tournaments(api_report)

    catalog, catalog_stats = build_sackmann_catalog(
        manifest_path
    )

    mappings: dict[str, Any] = {}
    decisions: list[dict[str, Any]] = []

    counters = Counter()
    matches_by_status = Counter()

    for api_row in api_tournaments:
        decision = choose_match(api_row, catalog)
        status = decision["status"]

        counters[status] += 1
        matches_by_status[status] += api_row["matches"]

        key = str(api_row["tournament_key"])

        mappings[key] = {
            "name": api_row["name"],
            "event_type": api_row["event_type"],
            "gender": api_row["gender"],
            "tour_level": api_row["tour_level"],
            "surface": decision["surface"],
            "matches_seen": api_row["matches"],
            "status": status,
            "match_method": decision["method"],
            "confidence": decision["confidence"],
            "matched_sackmann": decision["candidate"],
            "needs_review": decision["surface"] == "unknown",
        }

        decisions.append(
            {
                "tournament_key": api_row["tournament_key"],
                "api_name": api_row["name"],
                "event_type": api_row["event_type"],
                "gender": api_row["gender"],
                "tour_level": api_row["tour_level"],
                "matches": api_row["matches"],
                **decision,
            }
        )

    decisions.sort(
        key=lambda row: (
            row["surface"] != "unknown",
            -row["matches"],
            row["api_name"],
        )
    )

    unresolved = [
        row
        for row in decisions
        if row["surface"] == "unknown"
    ]

    output_payload = {
        "schema_version": 1,
        "generated_at": now_iso(),
        "source": {
            "api_report": str(args.api_report),
            "sackmann_manifest": str(manifest_path),
        },
        "summary": {
            "api_tournaments": len(api_tournaments),
            "mapping_statuses": dict(counters),
            "matches_by_status": dict(matches_by_status),
            "resolved_tournaments": sum(
                row["surface"] != "unknown"
                for row in decisions
            ),
            "unresolved_tournaments": len(unresolved),
            "resolved_matches": sum(
                row["matches"]
                for row in decisions
                if row["surface"] != "unknown"
            ),
            "unresolved_matches": sum(
                row["matches"]
                for row in unresolved
            ),
            **catalog_stats,
        },
        "tournaments": dict(
            sorted(
                mappings.items(),
                key=lambda item: (
                    item[1]["surface"] == "unknown",
                    -item[1]["matches_seen"],
                    item[1]["name"],
                ),
            )
        ),
    }

    report_payload = {
        "generated_at": output_payload["generated_at"],
        "summary": output_payload["summary"],
        "unresolved": unresolved,
        "all_decisions": decisions,
    }

    save_json(output_path, output_payload)
    save_json(report_path, report_payload)

    print("TLE API TOURNAMENT METADATA DONE")
    print(
        json.dumps(
            output_payload["summary"],
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Output: {output_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
