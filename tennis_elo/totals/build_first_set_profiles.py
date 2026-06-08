import math
import os
import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any

from tennis_elo.config import ROOT_DIR
from tennis_elo.utils import clean_str, load_json, now_iso, save_json


ARCHIVE_FILE = ROOT_DIR / "data" / "totals" / "scorelines_archive.json"
OUTPUT_FILE = ROOT_DIR / "data" / "totals" / "player_first_set_profiles.json"
REPORT_FILE = ROOT_DIR / "data" / "reports" / "build_first_set_profiles_report.json"

RECENT_DAYS = int(os.getenv("FIRST_SET_RECENT_DAYS", "90"))
MIN_VALID_FIRST_SET_GAMES = int(os.getenv("MIN_VALID_FIRST_SET_GAMES", "6"))
MAX_VALID_FIRST_SET_GAMES = int(os.getenv("MAX_VALID_FIRST_SET_GAMES", "20"))


def parse_date(raw: Any) -> date | None:
    value = clean_str(raw)

    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%d.%m.%Y",
        "%Y%m%d",
    ):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def player_key(name: Any) -> str:
    value = clean_str(name).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def rate(count: int, sample: int) -> float | None:
    if sample <= 0:
        return None
    return round(count / sample, 4)


def confidence(sample: int) -> float:
    # Reaches 1.0 at 50 valid first-set samples.
    return round(min(1.0, sample / 50.0), 4)


def maturity(sample: int) -> str:
    if sample >= 30:
        return "mature"
    if sample >= 15:
        return "usable"
    if sample >= 8:
        return "limited"
    return "provisional"


def normalize_surface(raw: Any) -> str:
    value = clean_str(raw).lower()
    return value if value in {"hard", "clay", "grass", "carpet"} else "unknown"


def valid_match(row: dict[str, Any]) -> tuple[bool, str]:
    if row.get("retired") or row.get("walkover"):
        return False, "retired_or_walkover"

    first_games = row.get("first_set_games")
    try:
        first_games = int(first_games)
    except (TypeError, ValueError):
        return False, "missing_first_set_games"

    if first_games < MIN_VALID_FIRST_SET_GAMES or first_games > MAX_VALID_FIRST_SET_GAMES:
        return False, "invalid_first_set_games"

    if not clean_str(row.get("player_1")) or not clean_str(row.get("player_2")):
        return False, "missing_players"

    if not parse_date(row.get("date")):
        return False, "invalid_date"

    return True, "ok"


def make_match_view(
    row: dict[str, Any],
    player_name: str,
    opponent_name: str,
    side: int,
) -> dict[str, Any]:
    match_date = parse_date(row.get("date"))
    first_set = row.get("set_scores", [{}])[0] if row.get("set_scores") else {}

    try:
        p1_games = int(first_set.get("p1_games"))
        p2_games = int(first_set.get("p2_games"))
    except (TypeError, ValueError):
        raw = clean_str(row.get("first_set_score"))
        parts = raw.split("-")
        p1_games = int(parts[0]) if len(parts) == 2 and parts[0].isdigit() else 0
        p2_games = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0

    player_games = p1_games if side == 1 else p2_games
    opponent_games = p2_games if side == 1 else p1_games

    return {
        "date": match_date.isoformat() if match_date else "",
        "surface": normalize_surface(row.get("surface")),
        "tour_level": clean_str(row.get("tour_level")),
        "tournament": clean_str(row.get("tournament")),
        "gender": clean_str(row.get("gender")),
        "best_of": row.get("best_of"),
        "is_grand_slam": bool(row.get("is_grand_slam")),
        "opponent": opponent_name,
        "first_set_games": int(row.get("first_set_games")),
        "first_set_score_for_player": f"{player_games}-{opponent_games}",
        "first_set_won": player_games > opponent_games,
        "first_set_lost": player_games < opponent_games,
        "first_set_tiebreak": bool(row.get("first_set_tiebreak")),
        "over_8_5": int(row.get("first_set_games")) > 8.5,
        "over_9_5": int(row.get("first_set_games")) > 9.5,
        "over_10_5": int(row.get("first_set_games")) > 10.5,
        "over_11_5": int(row.get("first_set_games")) > 11.5,
        "source": clean_str(row.get("source")),
    }


def summarize(matches: list[dict[str, Any]]) -> dict[str, Any]:
    sample = len(matches)

    if sample == 0:
        return {
            "sample_size": 0,
            "avg_first_set_games": None,
            "median_first_set_games": None,
            "first_set_win_rate": None,
            "over_8_5_rate": None,
            "over_9_5_rate": None,
            "over_10_5_rate": None,
            "over_11_5_rate": None,
            "tiebreak_rate": None,
        }

    games = [m["first_set_games"] for m in matches]

    return {
        "sample_size": sample,
        "avg_first_set_games": round(mean(games), 3),
        "median_first_set_games": round(float(median(games)), 3),
        "first_set_win_rate": rate(sum(m["first_set_won"] for m in matches), sample),
        "over_8_5_rate": rate(sum(m["over_8_5"] for m in matches), sample),
        "over_9_5_rate": rate(sum(m["over_9_5"] for m in matches), sample),
        "over_10_5_rate": rate(sum(m["over_10_5"] for m in matches), sample),
        "over_11_5_rate": rate(sum(m["over_11_5"] for m in matches), sample),
        "tiebreak_rate": rate(sum(m["first_set_tiebreak"] for m in matches), sample),
    }


def build_profiles(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    per_player: dict[str, dict[str, Any]] = {}
    counters: dict[str, int] = defaultdict(int)

    valid_rows: list[dict[str, Any]] = []

    for row in rows:
        is_valid, reason = valid_match(row)
        counters[reason] += 1

        if is_valid:
            valid_rows.append(row)

    latest_date = max(
        (parse_date(row.get("date")) for row in valid_rows if parse_date(row.get("date"))),
        default=date.today(),
    )
    recent_cutoff = latest_date - timedelta(days=RECENT_DAYS)

    for row in valid_rows:
        player_1 = clean_str(row.get("player_1"))
        player_2 = clean_str(row.get("player_2"))

        for player_name, opponent_name, side in (
            (player_1, player_2, 1),
            (player_2, player_1, 2),
        ):
            key = player_key(player_name)

            if key not in per_player:
                per_player[key] = {
                    "player_name": player_name,
                    "player_key": key,
                    "matches": [],
                }

            per_player[key]["matches"].append(
                make_match_view(row, player_name, opponent_name, side)
            )

    profiles: list[dict[str, Any]] = []

    for key, payload in per_player.items():
        matches = sorted(payload["matches"], key=lambda item: item["date"])
        overall = summarize(matches)

        surface_profiles = {}
        for surface in ("hard", "clay", "grass", "carpet", "unknown"):
            surface_matches = [m for m in matches if m["surface"] == surface]
            if surface_matches:
                surface_profiles[surface] = summarize(surface_matches)

        recent_matches = [
            m for m in matches
            if parse_date(m["date"]) and parse_date(m["date"]) >= recent_cutoff
        ]
        recent = summarize(recent_matches)

        recent_10_matches = matches[-10:]
        recent_10 = summarize(recent_10_matches)

        profile = {
            "player_name": payload["player_name"],
            "player_key": key,
            "sample_size": overall["sample_size"],
            "confidence": confidence(overall["sample_size"]),
            "maturity": maturity(overall["sample_size"]),
            "last_match_date": matches[-1]["date"] if matches else "",
            "overall": overall,
            "recent_days": RECENT_DAYS,
            "recent": recent,
            "recent_10": recent_10,
            "surface": surface_profiles,
        }

        profiles.append(profile)

    profiles.sort(
        key=lambda item: (
            item["overall"]["sample_size"],
            item["confidence"],
            item["player_name"],
        ),
        reverse=True,
    )

    counters["valid_matches"] = len(valid_rows)
    counters["players"] = len(profiles)
    counters["mature_players"] = sum(p["maturity"] == "mature" for p in profiles)
    counters["usable_players"] = sum(p["maturity"] in {"mature", "usable"} for p in profiles)

    return profiles, dict(counters)


def main() -> None:
    payload = load_json(ARCHIVE_FILE, {})
    rows = payload.get("matches", []) if isinstance(payload, dict) else payload

    if not isinstance(rows, list):
        rows = []

    profiles, counts = build_profiles(rows)

    output = {
        "generated_at": now_iso(),
        "source_file": str(ARCHIVE_FILE),
        "model": "first_set_totals_profiles_v1",
        "settings": {
            "recent_days": RECENT_DAYS,
            "min_valid_first_set_games": MIN_VALID_FIRST_SET_GAMES,
            "max_valid_first_set_games": MAX_VALID_FIRST_SET_GAMES,
        },
        "counts": counts,
        "profiles": profiles,
    }

    report = {
        "generated_at": now_iso(),
        "source_file": str(ARCHIVE_FILE),
        "output_file": str(OUTPUT_FILE),
        "counts": counts,
        "top_sample_players": [
            {
                "player_name": p["player_name"],
                "sample_size": p["sample_size"],
                "confidence": p["confidence"],
                "maturity": p["maturity"],
                "over_9_5_rate": p["overall"]["over_9_5_rate"],
                "avg_first_set_games": p["overall"]["avg_first_set_games"],
            }
            for p in profiles[:30]
        ],
    }

    save_json(OUTPUT_FILE, output)
    save_json(REPORT_FILE, report)

    print("")
    print("BUILD FIRST SET PROFILES DONE")
    print(counts)
    print(f"Output: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")
    print("")


if __name__ == "__main__":
    main()
