from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from tennis_elo.config import ROOT_DIR


CANONICAL_MANIFEST = (
    ROOT_DIR
    / "data"
    / "tle"
    / "processed"
    / "canonical"
    / "tle_matches_manifest.json"
)

OUTPUT = (
    ROOT_DIR
    / "data"
    / "tle"
    / "backtests"
    / "tle_api_overlay_backtest.json"
)

REPORT = (
    ROOT_DIR
    / "data"
    / "tle"
    / "reports"
    / "tle_api_overlay_backtest_report.json"
)

DEFAULT_ELO = 1500.0
GLOBAL_K = 24.0
GLOBAL_SURFACE_K = 20.0
LEVEL_K = 24.0
LEVEL_SURFACE_K = 20.0

VALID_SURFACES = {"hard", "clay", "grass", "carpet"}

MAIN_TOUR_LEVELS = {"atp_wta", "grand_slam"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def parse_date(value: Any) -> date | None:
    text = clean(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_matches(manifest_path: Path):
    manifest = read_json(manifest_path)

    rows = []
    for item in manifest.get("year_files") or []:
        relative = item.get("path")
        if not relative:
            continue

        path = Path(relative)
        if not path.is_absolute():
            path = ROOT_DIR / path

        for match in read_jsonl_gz(path):
            rows.append(match)

    rows.sort(
        key=lambda row: (
            clean(row.get("date")),
            clean(row.get("tle_match_id")),
        )
    )

    yield from rows


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def update_pair(
    winner_rating: float,
    loser_rating: float,
    k_factor: float,
) -> tuple[float, float]:
    winner_expected = expected_score(winner_rating, loser_rating)
    change = k_factor * (1.0 - winner_expected)
    return winner_rating + change, loser_rating - change


def player_identity(player: dict[str, Any], gender: str) -> tuple[str, str]:
    player_id = player.get("sackmann_player_id")
    name = clean(player.get("name"))

    if player_id not in (None, ""):
        return f"{gender}:sackmann:{int(player_id)}", name

    return f"{gender}:name:{name.lower()}", name


def new_surface_state() -> dict[str, Any]:
    return {"elo": DEFAULT_ELO, "matches": 0, "wins": 0}


def new_level_state() -> dict[str, Any]:
    return {
        "overall_elo": DEFAULT_ELO,
        "matches": 0,
        "wins": 0,
        "surfaces": {},
    }


def new_player_state(player_key: str, display_name: str, gender: str) -> dict[str, Any]:
    return {
        "player_key": player_key,
        "display_name": display_name,
        "gender": gender,
        "global": {
            "overall_elo": DEFAULT_ELO,
            "matches": 0,
            "wins": 0,
            "surfaces": {},
        },
        "levels": {},
    }


def ensure_player(
    players: dict[str, dict[str, Any]],
    player_key: str,
    name: str,
    gender: str,
) -> dict[str, Any]:
    if player_key not in players:
        players[player_key] = new_player_state(player_key, name, gender)
    return players[player_key]


def ensure_surface(container: dict[str, Any], surface: str) -> dict[str, Any]:
    if surface not in container["surfaces"]:
        container["surfaces"][surface] = new_surface_state()
    return container["surfaces"][surface]


def ensure_level(player: dict[str, Any], level: str) -> dict[str, Any]:
    if level not in player["levels"]:
        player["levels"][level] = new_level_state()
    return player["levels"][level]


def update_rating_layer(
    winner_container: dict[str, Any],
    loser_container: dict[str, Any],
    rating_field: str,
    k_factor: float,
) -> None:
    winner_rating = float(winner_container[rating_field])
    loser_rating = float(loser_container[rating_field])
    winner_new, loser_new = update_pair(winner_rating, loser_rating, k_factor)
    winner_container[rating_field] = winner_new
    loser_container[rating_field] = loser_new


def update_state_for_match(
    match: dict[str, Any],
    players: dict[str, dict[str, Any]],
) -> None:
    if not match.get("ready_for_tle"):
        return

    gender = clean(match.get("gender")).lower()
    level = clean(match.get("tour_level")).lower()
    surface = clean((match.get("tournament") or {}).get("surface")).lower()

    if gender not in {"men", "women"}:
        return

    winner_raw = match.get("winner") or {}
    loser_raw = match.get("loser") or {}
    if not isinstance(winner_raw, dict) or not isinstance(loser_raw, dict):
        return

    winner_key, winner_name = player_identity(winner_raw, gender)
    loser_key, loser_name = player_identity(loser_raw, gender)

    if not winner_name or not loser_name or winner_key == loser_key:
        return

    winner = ensure_player(players, winner_key, winner_name, gender)
    loser = ensure_player(players, loser_key, loser_name, gender)

    update_rating_layer(winner["global"], loser["global"], "overall_elo", GLOBAL_K)
    winner["global"]["matches"] += 1
    loser["global"]["matches"] += 1
    winner["global"]["wins"] += 1

    if surface in VALID_SURFACES:
        winner_surface = ensure_surface(winner["global"], surface)
        loser_surface = ensure_surface(loser["global"], surface)
        update_rating_layer(winner_surface, loser_surface, "elo", GLOBAL_SURFACE_K)
        winner_surface["matches"] += 1
        loser_surface["matches"] += 1
        winner_surface["wins"] += 1

    winner_level = ensure_level(winner, level)
    loser_level = ensure_level(loser, level)
    update_rating_layer(winner_level, loser_level, "overall_elo", LEVEL_K)
    winner_level["matches"] += 1
    loser_level["matches"] += 1
    winner_level["wins"] += 1

    if surface in VALID_SURFACES:
        winner_level_surface = ensure_surface(winner_level, surface)
        loser_level_surface = ensure_surface(loser_level, surface)
        update_rating_layer(
            winner_level_surface,
            loser_level_surface,
            "elo",
            LEVEL_SURFACE_K,
        )
        winner_level_surface["matches"] += 1
        loser_level_surface["matches"] += 1
        winner_level_surface["wins"] += 1


def get_level_state(player: dict[str, Any], level: str) -> dict[str, Any] | None:
    return player.get("levels", {}).get(level)


def get_level_surface_state(
    player: dict[str, Any],
    level: str,
    surface: str,
) -> dict[str, Any] | None:
    level_state = get_level_state(player, level)
    if not level_state:
        return None
    return level_state.get("surfaces", {}).get(surface)


def probability_for_match(
    winner: dict[str, Any],
    loser: dict[str, Any],
    level: str,
    surface: str,
    args: argparse.Namespace,
) -> tuple[float | None, str, dict[str, Any]]:
    if level in MAIN_TOUR_LEVELS:
        model_level = "atp_wta"

        winner_level = get_level_state(winner, model_level)
        loser_level = get_level_state(loser, model_level)
        winner_surface = get_level_surface_state(winner, model_level, surface)
        loser_surface = get_level_surface_state(loser, model_level, surface)

        if not winner_level or not loser_level:
            return None, "main_tour_missing_level_rating", {}

        if (
            winner_level["matches"] < args.main_min_level_matches
            or loser_level["matches"] < args.main_min_level_matches
        ):
            return None, "main_tour_level_min_sample", {}

        if surface not in VALID_SURFACES:
            return None, "main_tour_unknown_surface", {}

        if not winner_surface or not loser_surface:
            return None, "main_tour_missing_surface_rating", {}

        if (
            winner_surface["matches"] < args.main_min_surface_matches
            or loser_surface["matches"] < args.main_min_surface_matches
        ):
            return None, "main_tour_surface_min_sample", {}

        p_level = expected_score(
            float(winner_level["overall_elo"]),
            float(loser_level["overall_elo"]),
        )
        p_surface = expected_score(
            float(winner_surface["elo"]),
            float(loser_surface["elo"]),
        )
        p = 0.80 * p_level + 0.20 * p_surface

        return p, "main_tour_80_level_20_surface", {
            "winner_level_matches": winner_level["matches"],
            "loser_level_matches": loser_level["matches"],
            "winner_surface_matches": winner_surface["matches"],
            "loser_surface_matches": loser_surface["matches"],
            "p_level": p_level,
            "p_surface": p_surface,
        }

    if level == "itf":
        winner_level = get_level_state(winner, "itf")
        loser_level = get_level_state(loser, "itf")

        if not winner_level or not loser_level:
            return None, "itf_missing_level_rating", {}

        if (
            winner_level["matches"] < args.itf_min_level_matches
            or loser_level["matches"] < args.itf_min_level_matches
        ):
            return None, "itf_level_min_sample", {}

        p = expected_score(
            float(winner_level["overall_elo"]),
            float(loser_level["overall_elo"]),
        )
        return p, "itf_100_level_overall", {
            "winner_level_matches": winner_level["matches"],
            "loser_level_matches": loser_level["matches"],
        }

    if level == "challenger":
        winner_level = get_level_state(winner, "challenger")
        loser_level = get_level_state(loser, "challenger")

        if not winner_level or not loser_level:
            return None, "challenger_missing_level_rating", {}

        if (
            winner_level["matches"] < args.challenger_min_level_matches
            or loser_level["matches"] < args.challenger_min_level_matches
        ):
            return None, "challenger_level_min_sample", {}

        p = expected_score(
            float(winner_level["overall_elo"]),
            float(loser_level["overall_elo"]),
        )
        return p, "challenger_100_level_overall", {
            "winner_level_matches": winner_level["matches"],
            "loser_level_matches": loser_level["matches"],
        }

    if level == "qualifying":
        return None, "qualifying_no_bet", {}

    return None, "unsupported_level", {}


def brier(p: float, y: float) -> float:
    return (p - y) ** 2


def log_loss(p: float, y: float) -> float:
    eps = 1e-15
    p = min(max(p, eps), 1.0 - eps)
    return -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))


def summarize_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        return {
            "sample": 0,
            "accuracy": None,
            "brier": None,
            "log_loss": None,
        }

    return {
        "sample": len(predictions),
        "accuracy": round(
            sum(row["correct_pick"] for row in predictions) / len(predictions),
            6,
        ),
        "brier": round(
            sum(row["brier"] for row in predictions) / len(predictions),
            6,
        ),
        "log_loss": round(
            sum(row["log_loss"] for row in predictions) / len(predictions),
            6,
        ),
        "avg_probability_winner": round(
            sum(row["winner_probability"] for row in predictions) / len(predictions),
            6,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Walk-forward test CORE TLE modela na API overlay tekmah. "
            "Kvote niso vkljuÄene; to je model-only accuracy/Brier/log-loss test."
        )
    )

    parser.add_argument("--manifest", default=str(CANONICAL_MANIFEST))
    parser.add_argument("--from-date", default=None)
    parser.add_argument("--to-date", default=None)
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--report", default=str(REPORT))

    parser.add_argument("--main-min-level-matches", type=int, default=20)
    parser.add_argument("--main-min-surface-matches", type=int, default=10)
    parser.add_argument("--itf-min-level-matches", type=int, default=5)
    parser.add_argument("--challenger-min-level-matches", type=int, default=5)

    args = parser.parse_args()

    from_date = parse_date(args.from_date) if args.from_date else None
    to_date = parse_date(args.to_date) if args.to_date else None

    players: dict[str, dict[str, Any]] = {}
    counters = Counter()
    predictions: list[dict[str, Any]] = []

    for match in iter_matches(Path(args.manifest)):
        match_date = parse_date(match.get("date"))
        if match_date is None:
            counters["bad_date"] += 1
            continue

        gender = clean(match.get("gender")).lower()
        level = clean(match.get("tour_level")).lower()
        surface = clean((match.get("tournament") or {}).get("surface")).lower()
        source = clean(match.get("source"))

        is_test_source = source == "api_tennis"
        in_window = True
        if from_date and match_date < from_date:
            in_window = False
        if to_date and match_date > to_date:
            in_window = False

        if is_test_source and in_window:
            counters["api_test_candidates"] += 1

            winner_raw = match.get("winner") or {}
            loser_raw = match.get("loser") or {}

            winner_key, winner_name = player_identity(winner_raw, gender)
            loser_key, loser_name = player_identity(loser_raw, gender)

            winner = players.get(winner_key)
            loser = players.get(loser_key)

            if not winner or not loser:
                counters["skipped_missing_player_history"] += 1
            else:
                p_winner, model_name, details = probability_for_match(
                    winner,
                    loser,
                    level,
                    surface,
                    args,
                )

                if p_winner is None:
                    counters[f"skipped_{model_name}"] += 1
                else:
                    correct = p_winner >= 0.5

                    prediction = {
                        "date": match_date.isoformat(),
                        "tle_match_id": match.get("tle_match_id"),
                        "source": source,
                        "gender": gender,
                        "tour_level": level,
                        "surface": surface,
                        "tournament": clean((match.get("tournament") or {}).get("name")),
                        "round": clean(match.get("round")),
                        "winner": winner_name,
                        "loser": loser_name,
                        "winner_key": winner_key,
                        "loser_key": loser_key,
                        "winner_probability": round(p_winner, 6),
                        "predicted_winner": winner_name if correct else loser_name,
                        "actual_winner": winner_name,
                        "correct_pick": bool(correct),
                        "brier": brier(p_winner, 1.0),
                        "log_loss": log_loss(p_winner, 1.0),
                        "model": model_name,
                        "details": details,
                    }
                    predictions.append(prediction)
                    counters["predicted"] += 1
                    counters[f"predicted_level_{level}"] += 1

        update_state_for_match(match, players)

    by_level = {}
    for level in sorted({row["tour_level"] for row in predictions}):
        rows = [row for row in predictions if row["tour_level"] == level]
        by_level[level] = summarize_predictions(rows)

    by_gender = {}
    for gender in sorted({row["gender"] for row in predictions}):
        rows = [row for row in predictions if row["gender"] == gender]
        by_gender[gender] = summarize_predictions(rows)

    summary = {
        "generated_at": now_iso(),
        "test_source": "api_tennis",
        "from_date": from_date.isoformat() if from_date else None,
        "to_date": to_date.isoformat() if to_date else None,
        "settings": {
            "main_min_level_matches": args.main_min_level_matches,
            "main_min_surface_matches": args.main_min_surface_matches,
            "itf_min_level_matches": args.itf_min_level_matches,
            "challenger_min_level_matches": args.challenger_min_level_matches,
            "models": {
                "main_tour": "80% level overall + 20% level surface",
                "itf": "100% ITF level overall",
                "challenger": "100% Challenger level overall",
                "qualifying": "NO_BET",
            },
        },
        "counters": dict(sorted(counters.items())),
        "overall": summarize_predictions(predictions),
        "by_level": by_level,
        "by_gender": by_gender,
    }

    output_payload = {
        "schema_version": 1,
        "summary": summary,
        "predictions": predictions,
    }

    report_payload = {
        "schema_version": 1,
        "summary": summary,
        "sample_predictions": predictions[:200],
    }

    save_json(Path(args.output), output_payload)
    save_json(Path(args.report), report_payload)

    print("TLE API OVERLAY BACKTEST DONE")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
