from __future__ import annotations

import argparse
import json
import math
import statistics
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tennis_elo.config import ROOT_DIR


DEFAULT_MODEL_BACKTEST = (
    ROOT_DIR / "data" / "tle" / "backtests" / "tle_api_overlay_backtest.json"
)

DEFAULT_ENRICHED = (
    ROOT_DIR / "data" / "tle" / "source" / "api" / "tle_api_results_backfill_enriched.json"
)

DEFAULT_ODDS = (
    "https://raw.githubusercontent.com/jakobm121/Ai/refs/heads/main/"
    "data/tle/odds_backfill/tle_odds_backfill.json"
)

DEFAULT_OUTPUT = (
    ROOT_DIR / "data" / "tle" / "backtests" / "tle_api_overlay_odds_backtest.json"
)

DEFAULT_REPORT = (
    ROOT_DIR / "data" / "tle" / "reports" / "tle_api_overlay_odds_backtest_report.json"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number) or number <= 1.0:
        return None

    return number


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_json(path_or_url: str | Path) -> dict[str, Any]:
    text = str(path_or_url)

    if text.startswith("http://") or text.startswith("https://"):
        with urllib.request.urlopen(text, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    else:
        path = Path(text)
        if not path.is_absolute():
            path = ROOT_DIR / path
        payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid JSON object: {path_or_url}")

    return payload


def event_key_from_tle_match_id(tle_match_id: Any) -> str:
    text = clean(tle_match_id)
    prefix = "api_tennis_"
    return text[len(prefix):] if text.startswith(prefix) else text


def build_enriched_index(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    matches = payload.get("matches") or []

    index = {}
    for match in matches:
        if isinstance(match, dict) and clean(match.get("event_key")):
            index[clean(match.get("event_key"))] = match

    return index


def odds_market_for_event(
    odds_payload: dict[str, Any],
    event_key: str,
) -> dict[str, Any] | None:
    events = odds_payload.get("odds_by_event_key") or {}
    event = events.get(str(event_key))
    if not isinstance(event, dict):
        return None

    markets = event.get("markets") or {}
    if not isinstance(markets, dict):
        return None

    market = markets.get("Home/Away")
    if not isinstance(market, dict):
        return None

    home = market.get("Home")
    away = market.get("Away")

    if not isinstance(home, dict) or not isinstance(away, dict):
        return None

    return {"Home": home, "Away": away}


def chosen_book_odds(
    market: dict[str, dict[str, Any]],
    side: str,
    bookmaker: str,
    fallback: str,
) -> tuple[float | None, str | None]:
    side_odds = market.get(side) or {}
    value = safe_float(side_odds.get(bookmaker))

    if value is not None:
        return value, bookmaker

    if fallback == "none":
        return None, None

    available = [
        number
        for number in (safe_float(value) for value in side_odds.values())
        if number is not None
    ]

    if not available:
        return None, None

    if fallback == "max":
        return max(available), "fallback_max"

    if fallback == "min":
        return min(available), "fallback_min"

    return float(statistics.median(available)), "fallback_median"


def side_pair_odds(
    market: dict[str, dict[str, Any]],
    bookmaker: str,
    fallback: str,
) -> tuple[float | None, float | None, str | None]:
    home_odds, home_book = chosen_book_odds(market, "Home", bookmaker, fallback)
    away_odds, away_book = chosen_book_odds(market, "Away", bookmaker, fallback)

    if home_odds is not None and away_odds is not None:
        if home_book == bookmaker and away_book == bookmaker:
            return home_odds, away_odds, bookmaker

        if fallback == "none":
            return None, None, None

    def all_side_values(side: str) -> list[float]:
        side_odds = market.get(side) or {}
        return [
            number
            for number in (safe_float(value) for value in side_odds.values())
            if number is not None
        ]

    home_values = all_side_values("Home")
    away_values = all_side_values("Away")

    if not home_values or not away_values:
        return None, None, None

    if fallback == "max":
        return max(home_values), max(away_values), "fallback_max_pair"

    if fallback == "min":
        return min(home_values), min(away_values), "fallback_min_pair"

    return (
        float(statistics.median(home_values)),
        float(statistics.median(away_values)),
        "fallback_median_pair",
    )


def devig_probs(home_odds: float, away_odds: float) -> tuple[float, float, float]:
    home_raw = 1.0 / home_odds
    away_raw = 1.0 / away_odds
    total = home_raw + away_raw
    return home_raw / total, away_raw / total, total - 1.0


def opposite_api_side(side: str) -> str | None:
    if side == "player_1":
        return "player_2"
    if side == "player_2":
        return "player_1"
    return None


def api_side_to_odds_side(side: str) -> str | None:
    # Raw API debug confirmed:
    # Home = event_first_player = player_1
    # Away = event_second_player = player_2
    if side == "player_1":
        return "Home"
    if side == "player_2":
        return "Away"
    return None


def selection_side_from_prediction(
    prediction: dict[str, Any],
    enriched_match: dict[str, Any],
) -> tuple[str | None, str | None]:
    actual_winner_api_side = clean(enriched_match.get("winner_side"))

    if actual_winner_api_side not in {"player_1", "player_2"}:
        return None, "missing_winner_side"

    selected_api_side = (
        actual_winner_api_side
        if bool(prediction.get("correct_pick"))
        else opposite_api_side(actual_winner_api_side)
    )

    selected_odds_side = api_side_to_odds_side(selected_api_side or "")

    if selected_odds_side is None:
        return None, "could_not_map_selected_side"

    return selected_odds_side, actual_winner_api_side


def summarize_bets(bets: list[dict[str, Any]]) -> dict[str, Any]:
    if not bets:
        return {
            "bets": 0,
            "staked": 0,
            "profit": 0,
            "roi": None,
            "hit_rate": None,
            "avg_model_probability": None,
            "avg_odds": None,
            "avg_ev": None,
            "avg_edge": None,
        }

    staked = len(bets)
    profit = sum(row["profit"] for row in bets)
    wins = sum(1 for row in bets if row["won"])

    return {
        "bets": staked,
        "staked": staked,
        "profit": round(profit, 6),
        "roi": round(profit / staked, 6),
        "hit_rate": round(wins / staked, 6),
        "avg_model_probability": round(
            sum(row["model_probability"] for row in bets) / staked,
            6,
        ),
        "avg_odds": round(sum(row["odds"] for row in bets) / staked, 6),
        "avg_ev": round(sum(row["ev"] for row in bets) / staked, 6),
        "avg_edge": round(sum(row["edge"] for row in bets) / staked, 6),
    }


def summarize_by_field(bets: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = sorted({clean(row.get(field)) for row in bets})
    return {
        value: summarize_bets([row for row in bets if clean(row.get(field)) == value])
        for value in values
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Correct ROI backtest for TLE API overlay with historical Home/Away odds. "
            "Raw API debug confirmed Home=event_first_player/player_1 and Away=event_second_player/player_2."
        )
    )

    parser.add_argument("--model-backtest", default=str(DEFAULT_MODEL_BACKTEST))
    parser.add_argument("--enriched", default=str(DEFAULT_ENRICHED))
    parser.add_argument("--odds", default=DEFAULT_ODDS)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--bookmaker", default="Pncl")
    parser.add_argument(
        "--fallback",
        choices=["none", "median", "max", "min"],
        default="median",
    )
    parser.add_argument("--min-edge", type=float, default=0.03)
    parser.add_argument("--min-ev", type=float, default=0.0)

    args = parser.parse_args()

    model_payload = load_json(args.model_backtest)
    odds_payload = load_json(args.odds)
    enriched_index = build_enriched_index(args.enriched)

    predictions = model_payload.get("predictions") or []
    if not isinstance(predictions, list):
        raise RuntimeError("model-backtest nima polja predictions")

    counters = Counter()
    all_rows = []
    value_bets = []

    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue

        counters["model_predictions"] += 1

        event_key = event_key_from_tle_match_id(prediction.get("tle_match_id"))
        enriched = enriched_index.get(event_key)
        if not enriched:
            counters["skipped_missing_enriched_match"] += 1
            continue

        market = odds_market_for_event(odds_payload, event_key)
        if not market:
            counters["skipped_missing_home_away_odds"] += 1
            continue

        selected_side, actual_winner_api_side = selection_side_from_prediction(
            prediction,
            enriched,
        )

        if not selected_side:
            counters[f"skipped_{actual_winner_api_side}"] += 1
            continue

        home_odds, away_odds, odds_source = side_pair_odds(
            market,
            args.bookmaker,
            args.fallback,
        )
        if home_odds is None or away_odds is None:
            counters["skipped_missing_pair_odds"] += 1
            continue

        selected_odds = home_odds if selected_side == "Home" else away_odds
        home_fair, away_fair, overround = devig_probs(home_odds, away_odds)
        book_probability = home_fair if selected_side == "Home" else away_fair

        winner_probability = float(prediction["winner_probability"])
        predicted_is_actual = bool(prediction["correct_pick"])
        model_probability = (
            winner_probability
            if predicted_is_actual
            else 1.0 - winner_probability
        )

        ev = model_probability * selected_odds - 1.0
        edge = model_probability - book_probability
        won = predicted_is_actual
        profit = selected_odds - 1.0 if won else -1.0

        row = {
            "date": prediction.get("date"),
            "event_key": event_key,
            "tle_match_id": prediction.get("tle_match_id"),
            "gender": prediction.get("gender"),
            "tour_level": prediction.get("tour_level"),
            "surface": prediction.get("surface"),
            "tournament": prediction.get("tournament"),
            "model": prediction.get("model"),
            "selection": prediction.get("predicted_winner"),
            "actual_winner": prediction.get("actual_winner"),
            "selected_side": selected_side,
            "actual_winner_api_side": actual_winner_api_side,
            "winner_side_method": "raw_enriched_winner_side",
            "won": won,
            "model_probability": round(model_probability, 6),
            "winner_probability": prediction.get("winner_probability"),
            "book_probability_devig": round(book_probability, 6),
            "edge": round(edge, 6),
            "ev": round(ev, 6),
            "odds": round(selected_odds, 6),
            "home_odds": round(home_odds, 6),
            "away_odds": round(away_odds, 6),
            "overround": round(overround, 6),
            "odds_source": odds_source,
            "profit": round(profit, 6),
            "player_1": enriched.get("player_1"),
            "player_2": enriched.get("player_2"),
            "raw_enriched_winner_side": enriched.get("winner_side"),
        }

        all_rows.append(row)
        counters["priced_predictions"] += 1

        if ev >= args.min_ev and edge >= args.min_edge:
            value_bets.append(row)
            counters["value_bets"] += 1

    thresholds = {}
    for edge_threshold in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]:
        bucket_bets = [
            row
            for row in all_rows
            if row["ev"] >= args.min_ev and row["edge"] >= edge_threshold
        ]
        thresholds[f"edge_gte_{edge_threshold:.2f}"] = summarize_bets(bucket_bets)

    summary = {
        "generated_at": now_iso(),
        "bookmaker": args.bookmaker,
        "fallback": args.fallback,
        "side_mapping": {
            "actual_winner_api_side": "enriched winner_side",
            "Home": "player_1 / event_first_player",
            "Away": "player_2 / event_second_player",
            "raw_debug_confirmed_examples": ["12134273", "12134279"],
        },
        "min_edge": args.min_edge,
        "min_ev": args.min_ev,
        "counters": dict(sorted(counters.items())),
        "all_priced_model_picks": summarize_bets(all_rows),
        "value_bets": summarize_bets(value_bets),
        "value_bets_by_level": summarize_by_field(value_bets, "tour_level"),
        "value_bets_by_gender": summarize_by_field(value_bets, "gender"),
        "thresholds": thresholds,
    }

    output_payload = {
        "schema_version": 3,
        "summary": summary,
        "priced_predictions": all_rows,
        "value_bets": value_bets,
    }

    report_payload = {
        "schema_version": 3,
        "summary": summary,
        "sample_value_bets": value_bets[:200],
        "sample_priced_predictions": all_rows[:200],
    }

    save_json(Path(args.output), output_payload)
    save_json(Path(args.report), report_payload)

    print("TLE API OVERLAY ODDS BACKTEST V3 DONE")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
