from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tennis_elo.config import ROOT_DIR


DEFAULT_ODDS_BACKTEST = (
    ROOT_DIR / "data" / "tle" / "backtests" / "tle_api_overlay_odds_backtest.json"
)

DEFAULT_ENRICHED = (
    ROOT_DIR / "data" / "tle" / "source" / "api" / "tle_api_results_backfill_enriched.json"
)

DEFAULT_ODDS = (
    "https://raw.githubusercontent.com/jakobm121/Ai/refs/heads/main/"
    "data/tle/odds_backfill/tle_odds_backfill.json"
)

DEFAULT_OUTPUT_JSON = (
    ROOT_DIR / "data" / "tle" / "reports" / "tle_api_overlay_odds_audit.json"
)

DEFAULT_OUTPUT_CSV = (
    ROOT_DIR / "data" / "tle" / "reports" / "tle_api_overlay_odds_audit.csv"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalized(value: Any) -> str:
    return " ".join(clean(value).lower().replace("-", " ").split())


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number) or number <= 1.0:
        return None

    return number


def round_or_none(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "audit_group",
        "audit_rank",
        "date",
        "event_key",
        "tour_level",
        "gender",
        "surface",
        "tournament",
        "round",
        "player_1",
        "player_2",
        "raw_enriched_winner_side",
        "actual_winner_api_side",
        "winner_side_method",
        "actual_winner",
        "selection",
        "selected_side",
        "side_check",
        "home_player_assumption",
        "away_player_assumption",
        "home_odds",
        "away_odds",
        "selected_odds",
        "odds_source",
        "model_probability",
        "book_probability_devig",
        "edge",
        "ev",
        "won",
        "profit",
        "score",
        "status",
        "model",
        "tle_match_id",
    ]

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

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


def build_enriched_index(path_or_url: str | Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path_or_url)
    matches = payload.get("matches") or []

    index = {}
    for match in matches:
        if not isinstance(match, dict):
            continue
        event_key = clean(match.get("event_key"))
        if event_key:
            index[event_key] = match

    return index


def raw_market_for_event(
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

    return market


def market_values(market: dict[str, Any], side: str) -> dict[str, float]:
    raw_side = market.get(side) or {}
    if not isinstance(raw_side, dict):
        return {}

    values = {}
    for bookmaker, value in raw_side.items():
        number = safe_float(value)
        if number is not None:
            values[str(bookmaker)] = number

    return values


def median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def side_to_player(side: str, enriched_match: dict[str, Any]) -> str:
    if side == "player_1":
        return clean(enriched_match.get("player_1"))
    if side == "player_2":
        return clean(enriched_match.get("player_2"))
    return ""


def selected_odds_side_to_api_side(selected_side: str) -> str:
    if selected_side == "Home":
        return "player_1"
    if selected_side == "Away":
        return "player_2"
    return ""


def opposite_side(side: str) -> str:
    if side == "player_1":
        return "player_2"
    if side == "player_2":
        return "player_1"
    return ""


def check_side(row: dict[str, Any], enriched_match: dict[str, Any]) -> str:
    actual_side = clean(row.get("actual_winner_api_side"))
    selected_side = clean(row.get("selected_side"))
    selected_api_side = selected_odds_side_to_api_side(selected_side)

    actual_winner = normalized(row.get("actual_winner"))
    selection = normalized(row.get("selection"))

    actual_player = normalized(side_to_player(actual_side, enriched_match))
    selected_player = normalized(side_to_player(selected_api_side, enriched_match))

    if actual_side not in {"player_1", "player_2"}:
        return "CHECK: missing actual_winner_api_side"

    if selected_side not in {"Home", "Away"}:
        return "CHECK: missing selected_side"

    if actual_winner and actual_player and actual_winner != actual_player:
        # Zaradi Sackmann/API imen to ni nujno fatalno, zato je REVIEW namesto hard fail.
        return "REVIEW: actual winner name differs from API side name"

    if selection and selected_player and selection != selected_player:
        return "REVIEW: selection name differs from selected side name"

    return "OK"


def enrich_audit_row(
    bet: dict[str, Any],
    enriched_index: dict[str, dict[str, Any]],
    odds_payload: dict[str, Any],
    audit_group: str,
    audit_rank: int,
) -> dict[str, Any]:
    event_key = clean(bet.get("event_key"))
    enriched = enriched_index.get(event_key, {})
    market = raw_market_for_event(odds_payload, event_key)

    home_values = market_values(market or {}, "Home")
    away_values = market_values(market or {}, "Away")

    actual_side = clean(bet.get("actual_winner_api_side"))
    if not actual_side:
        # kompatibilnost z v1 outputom, Äe ga kdo pomotoma audita
        raw = clean(enriched.get("winner_side"))
        actual_side = opposite_side(raw)

    enriched_row = {
        "audit_group": audit_group,
        "audit_rank": audit_rank,
        "date": bet.get("date"),
        "event_key": event_key,
        "tour_level": bet.get("tour_level"),
        "gender": bet.get("gender"),
        "surface": bet.get("surface"),
        "tournament": bet.get("tournament"),
        "round": bet.get("round"),
        "player_1": clean(enriched.get("player_1")),
        "player_2": clean(enriched.get("player_2")),
        "raw_enriched_winner_side": enriched.get("winner_side"),
        "actual_winner_api_side": actual_side,
        "winner_side_method": bet.get("winner_side_method", "v1_assumed_inverted_for_audit"),
        "actual_winner": bet.get("actual_winner"),
        "selection": bet.get("selection"),
        "selected_side": bet.get("selected_side"),
        "home_player_assumption": "Home = player_1",
        "away_player_assumption": "Away = player_2",
        "home_odds": bet.get("home_odds"),
        "away_odds": bet.get("away_odds"),
        "selected_odds": bet.get("odds"),
        "odds_source": bet.get("odds_source"),
        "model_probability": bet.get("model_probability"),
        "book_probability_devig": bet.get("book_probability_devig"),
        "edge": bet.get("edge"),
        "ev": bet.get("ev"),
        "won": bet.get("won"),
        "profit": bet.get("profit"),
        "score": enriched.get("score"),
        "status": enriched.get("status"),
        "model": bet.get("model"),
        "tle_match_id": bet.get("tle_match_id"),
        "raw_home_books_count": len(home_values),
        "raw_away_books_count": len(away_values),
        "raw_home_min": round(min(home_values.values()), 6) if home_values else None,
        "raw_home_median": round_or_none(median_or_none(list(home_values.values()))),
        "raw_home_max": round(max(home_values.values()), 6) if home_values else None,
        "raw_away_min": round(min(away_values.values()), 6) if away_values else None,
        "raw_away_median": round_or_none(median_or_none(list(away_values.values()))),
        "raw_away_max": round(max(away_values.values()), 6) if away_values else None,
        "raw_home_books": home_values,
        "raw_away_books": away_values,
    }

    enriched_row["side_check"] = check_side(enriched_row, enriched)
    return enriched_row


def add_ranked_rows(
    output: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    enriched_index: dict[str, dict[str, Any]],
    odds_payload: dict[str, Any],
    group_name: str,
    limit: int,
) -> None:
    seen = set()

    for row in rows[:limit]:
        key = (group_name, clean(row.get("event_key")))
        if key in seen:
            continue
        seen.add(key)

        output.append(
            enrich_audit_row(
                row,
                enriched_index,
                odds_payload,
                group_name,
                len(seen),
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "V2 audit CSV/JSON za TLE API overlay ROI backtest. "
            "Pravilno podpira V2 actual_winner_api_side in opozori le na realne review primere."
        )
    )

    parser.add_argument("--odds-backtest", default=str(DEFAULT_ODDS_BACKTEST))
    parser.add_argument("--enriched", default=str(DEFAULT_ENRICHED))
    parser.add_argument("--odds", default=DEFAULT_ODDS)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--top-ev", type=int, default=100)
    parser.add_argument("--top-lost", type=int, default=50)
    parser.add_argument("--top-underdog-won", type=int, default=50)
    parser.add_argument("--random-sample", type=int, default=50)
    parser.add_argument("--seed", type=int, default=121)

    args = parser.parse_args()

    odds_backtest = load_json(args.odds_backtest)
    enriched_index = build_enriched_index(args.enriched)
    odds_payload = load_json(args.odds)

    value_bets = odds_backtest.get("value_bets") or []
    priced_predictions = odds_backtest.get("priced_predictions") or []

    if not isinstance(value_bets, list):
        raise RuntimeError("odds backtest nima list polja value_bets")
    if not isinstance(priced_predictions, list):
        raise RuntimeError("odds backtest nima list polja priced_predictions")

    audit_rows: list[dict[str, Any]] = []

    add_ranked_rows(
        audit_rows,
        sorted(value_bets, key=lambda row: float(row.get("ev", 0)), reverse=True),
        enriched_index,
        odds_payload,
        "top_value_by_ev",
        args.top_ev,
    )

    add_ranked_rows(
        audit_rows,
        sorted(
            [row for row in value_bets if not bool(row.get("won"))],
            key=lambda row: float(row.get("ev", 0)),
            reverse=True,
        ),
        enriched_index,
        odds_payload,
        "top_lost_value_by_ev",
        args.top_lost,
    )

    add_ranked_rows(
        audit_rows,
        sorted(
            [
                row
                for row in value_bets
                if bool(row.get("won")) and float(row.get("odds", 0)) >= 2.0
            ],
            key=lambda row: float(row.get("odds", 0)),
            reverse=True,
        ),
        enriched_index,
        odds_payload,
        "top_won_underdogs",
        args.top_underdog_won,
    )

    random.seed(args.seed)
    random_rows = list(value_bets)
    random.shuffle(random_rows)
    add_ranked_rows(
        audit_rows,
        random_rows,
        enriched_index,
        odds_payload,
        "random_value_sample",
        args.random_sample,
    )

    counters = Counter()
    unique_events = set()

    for row in audit_rows:
        counters[f"side_check_{row['side_check']}"] += 1
        counters[f"group_{row['audit_group']}"] += 1
        unique_events.add(row["event_key"])

    review_rows = [
        row
        for row in audit_rows
        if clean(row.get("side_check")) != "OK"
    ]

    summary = {
        "generated_at": now_iso(),
        "source": {
            "odds_backtest": str(args.odds_backtest),
            "enriched": str(args.enriched),
            "odds": str(args.odds),
        },
        "value_bets_total": len(value_bets),
        "priced_predictions_total": len(priced_predictions),
        "audit_rows": len(audit_rows),
        "unique_events": len(unique_events),
        "review_rows": len(review_rows),
        "counters": dict(sorted(counters.items())),
        "audit_groups": {
            "top_value_by_ev": args.top_ev,
            "top_lost_value_by_ev": args.top_lost,
            "top_won_underdogs": args.top_underdog_won,
            "random_value_sample": args.random_sample,
        },
        "manual_check_instructions": [
            "Home naj pomeni player_1, Away naj pomeni player_2.",
            "Preveri, da selected_side kaÅ¾e na selection.",
            "Preveri nekaj zmag in porazov z actual_winner_api_side + score.",
            "REVIEW vrstice so lahko samo razlika Sackmann/API imen, niso nujno napaka.",
        ],
    }

    payload = {
        "schema_version": 2,
        "summary": summary,
        "review_rows": review_rows,
        "audit_rows": audit_rows,
    }

    save_json(Path(args.output_json), payload)
    save_csv(Path(args.output_csv), audit_rows)

    print("TLE API OVERLAY ODDS AUDIT V2 DONE")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"JSON: {args.output_json}")
    print(f"CSV:  {args.output_csv}")


if __name__ == "__main__":
    main()
