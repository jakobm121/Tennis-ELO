import json
from collections import Counter, defaultdict
from typing import Any
from urllib.parse import urlparse

from tennis_elo.config import CANONICAL_MATCHES_FILE, ROOT_DIR
from tennis_elo.utils import clean_str, load_json, now_iso, save_json


OUTPUT_FILE = ROOT_DIR / "data" / "reports" / "tournament_code_audit.json"
TOP_PLAYERS_PER_CODE = 12
TOP_SOURCE_HOSTS_PER_CODE = 8
TOP_SOURCE_URLS_PER_CODE = 8


def safe_year(value: Any) -> str:
    text = clean_str(value)
    return text[:4] if len(text) >= 4 and text[:4].isdigit() else "unknown"


def source_host(value: Any) -> str:
    text = clean_str(value)
    if not text:
        return "unknown"
    try:
        return urlparse(text).netloc or "unknown"
    except ValueError:
        return "unknown"


def normalize_surface(value: Any) -> str:
    surface = clean_str(value).lower()
    aliases = {
        "hardcourt": "hard",
        "hard court": "hard",
        "cement": "hard",
        "red clay": "clay",
        "green clay": "clay",
        "lawn": "grass",
    }
    return aliases.get(surface, surface) or "unknown"


def tournament_code(row: dict[str, Any]) -> str:
    return (
        clean_str(row.get("tournament_code"))
        or clean_str(row.get("tournament"))
        or "unknown"
    )


def player_names(row: dict[str, Any]) -> list[str]:
    values = []
    for key in ("player_1", "player_2", "winner", "loser"):
        value = clean_str(row.get(key))
        if value:
            values.append(value)
    return values


def main() -> None:
    payload = load_json(CANONICAL_MATCHES_FILE, {})
    matches = payload.get("matches", []) if isinstance(payload, dict) else []
    if not isinstance(matches, list):
        matches = []

    grouped: dict[str, dict[str, Any]] = {}

    for row in matches:
        if not isinstance(row, dict):
            continue

        code = tournament_code(row)
        grouped.setdefault(
            code,
            {
                "matches": 0,
                "years": Counter(),
                "surfaces": Counter(),
                "players": Counter(),
                "source_hosts": Counter(),
                "source_urls": Counter(),
                "sample_matches": [],
            },
        )

        bucket = grouped[code]
        bucket["matches"] += 1
        bucket["years"][safe_year(row.get("date"))] += 1
        bucket["surfaces"][normalize_surface(row.get("surface"))] += 1

        for player in player_names(row):
            bucket["players"][player] += 1

        source_url = clean_str(row.get("source_url"))
        if source_url:
            bucket["source_urls"][source_url] += 1
            bucket["source_hosts"][source_host(source_url)] += 1

        if len(bucket["sample_matches"]) < 8:
            bucket["sample_matches"].append(
                {
                    "date": row.get("date"),
                    "player_1": row.get("player_1"),
                    "player_2": row.get("player_2"),
                    "winner": row.get("winner"),
                    "surface": row.get("surface"),
                    "source_url": row.get("source_url"),
                    "canonical_match_id": row.get("canonical_match_id"),
                }
            )

    codes = []
    for code, bucket in grouped.items():
        years = dict(sorted(bucket["years"].items(), key=lambda item: item[0]))
        surfaces = dict(bucket["surfaces"].most_common())
        known_years = [int(year) for year in years if year.isdigit()]

        codes.append(
            {
                "tournament_code": code,
                "matches": bucket["matches"],
                "first_year": min(known_years) if known_years else None,
                "last_year": max(known_years) if known_years else None,
                "years": years,
                "surfaces": surfaces,
                "top_players": [
                    {"player": player, "matches_seen": count}
                    for player, count in bucket["players"].most_common(TOP_PLAYERS_PER_CODE)
                ],
                "top_source_hosts": [
                    {"host": host, "count": count}
                    for host, count in bucket["source_hosts"].most_common(TOP_SOURCE_HOSTS_PER_CODE)
                ],
                "top_source_urls": [
                    {"url": url, "count": count}
                    for url, count in bucket["source_urls"].most_common(TOP_SOURCE_URLS_PER_CODE)
                ],
                "sample_matches": bucket["sample_matches"],
            }
        )

    codes.sort(key=lambda row: (-int(row["matches"]), row["tournament_code"]))

    year_code_counts: dict[str, set[str]] = defaultdict(set)
    for row in codes:
        for year in row["years"]:
            if year != "unknown":
                year_code_counts[year].add(row["tournament_code"])

    summary = {
        "input_matches": len(matches),
        "distinct_tournament_codes": len(codes),
        "unknown_code_matches": next(
            (row["matches"] for row in codes if row["tournament_code"] == "unknown"),
            0,
        ),
        "codes_with_one_surface": sum(
            len([key for key, value in row["surfaces"].items() if key != "unknown" and value > 0]) == 1
            for row in codes
        ),
        "codes_across_multiple_years": sum(
            len([year for year in row["years"] if year != "unknown"]) > 1
            for row in codes
        ),
    }

    output = {
        "generated_at": now_iso(),
        "source_file": str(CANONICAL_MATCHES_FILE),
        "summary": summary,
        "codes": codes,
        "codes_per_year": {
            year: len(code_set)
            for year, code_set in sorted(year_code_counts.items())
        },
    }

    save_json(OUTPUT_FILE, output)

    print("")
    print("TOURNAMENT CODE AUDIT DONE")
    print("SUMMARY:", summary)
    print("\nTOP 30 TOURNAMENT CODES:")

    for row in codes[:30]:
        print(
            {
                "tournament_code": row["tournament_code"],
                "matches": row["matches"],
                "first_year": row["first_year"],
                "last_year": row["last_year"],
                "surfaces": row["surfaces"],
                "top_players": [item["player"] for item in row["top_players"][:5]],
            }
        )

    print(f"\nOutput: {OUTPUT_FILE}")
    print("")


if __name__ == "__main__":
    main()
