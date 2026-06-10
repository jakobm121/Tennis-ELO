from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import urllib.request
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from tennis_elo.config import ROOT_DIR


CANONICAL_MANIFEST = (
    ROOT_DIR / "data" / "tle" / "processed" / "canonical" / "tle_matches_manifest.json"
)

DEFAULT_ENRICHED = (
    ROOT_DIR / "data" / "tle" / "source" / "api" / "tle_api_results_backfill_enriched.json"
)

DEFAULT_ODDS = (
    "https://raw.githubusercontent.com/jakobm121/Ai/refs/heads/main/"
    "data/tle/odds_backfill/tle_odds_backfill.json"
)

DEFAULT_OUTPUT = (
    ROOT_DIR / "data" / "tle" / "backtests" / "tle_api_overlay_level_only_odds_backtest.json"
)

DEFAULT_REPORT = (
    ROOT_DIR / "data" / "tle" / "reports" / "tle_api_overlay_level_only_odds_backtest_report.json"
)

DEFAULT_ELO = 1500.0
GLOBAL_K = 24.0
GLOBAL_SURFACE_K = 20.0
LEVEL_K = 24.0
LEVEL_SURFACE_K = 20.0

VALID_SURFACES = {"hard", "clay", "grass", "carpet"}


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
        with urllib.request.urlopen(text, timeout=90) as response:
            payload = json.loads(response.read().decode("utf-8"))
    else:
        path = Path(text)
        if not path.is_absolute():
            path = ROOT_DIR / path
        payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid JSON object: {path_or_url}")
    return payload


def read_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_matches(manifest_path: Path):
    manifest = load_json(manifest_path)

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

    rows.sort(key=lambda row: (clean(row.get("date")), clean(row.get("tle_match_id"))))
    yield from rows


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def update_pair(winner_rating: float, loser_rating: float, k_factor: float) -> tuple[float, float]:
    winner_expected = expected_score(winner_rating, loser_rating)
    change = k_factor * (1.0 - winner_expected)
    return winner_rating + change, loser_rating - change


def player_identity(player: dict[str, Any], gender: str) -> tuple[str, str]:
    player_id = player.get("sackmann_player_id")
    name = clean(player.get("name"))

    if player_id not in (None, ""):
        return f"{gender}:sackmann:{int(player_id)}", name

    api_key = player.get("api_player_key")
    if api_key not in (None, ""):
        try:
            return f"{gender}:api:{int(api_key)}", name
        except (TypeError, ValueError):
            pass

    return f"{gender}:name:{name.lower()}", name


def new_surface_state() -> dict[str, Any]:
    return {"elo": DEFAULT_ELO, "matches": 0, "wins": 0}


def new_level_state() -> dict[str, Any]:
    return {"overall_elo": DEFAULT_ELO, "matches": 0, "wins": 0, "surfaces": {}}


def new_player_state(player_key: str, display_name: str, gender: str) -> dict[str, Any]:
    return {
        "player_key": player_key,
        "display_name": display_name,
        "gender": gender,
        "global": {"overall_elo": DEFAULT_ELO, "matches": 0, "wins": 0, "surfaces": {}},
        "levels": {},
    }


def ensure_player(players: dict[str, dict[str, Any]], player_key: str, name: str, gender: str) -> dict[str, Any]:
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


def update_rating_layer(winner_container: dict[str, Any], loser_container: dict[str, Any], rating_field: str, k_factor: float) -> None:
    winner_rating = float(winner_container[rating_field])
    loser_rating = float(loser_container[rating_field])
    winner_new, loser_new = update_pair(winner_rating, loser_rating, k_factor)
    winner_container[rating_field] = winner_new
    loser_container[rating_field] = loser_new


def update_state_for_match(match: dict[str, Any], players: dict[str, dict[str, Any]]) -> None:
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
        update_rating_layer(winner_level_surface, loser_level_surface, "elo", LEVEL_SURFACE_K)
        winner_level_surface["matches"] += 1
        loser_level_surface["matches"] += 1
        winner_level_surface["wins"] += 1


def get_level_state(player: dict[str, Any], level: str) -> dict[str, Any] | None:
    return player.get("levels", {}).get(level)


def level_min_matches(level: str, args: argparse.Namespace) -> int:
    if level in {"atp_wta", "grand_slam"}:
        return args.main_min_level_matches
    if level == "challenger":
        return args.challenger_min_level_matches
    if level == "itf":
        return args.itf_min_level_matches
    if level == "qualifying":
        return args.qualifying_min_level_matches
    return args.default_min_level_matches


def probability_level_only(
    winner: dict[str, Any],
    loser: dict[str, Any],
    level: str,
    args: argparse.Namespace,
) -> tuple[float | None, str, dict[str, Any]]:
    winner_level = get_level_state(winner, level)
    loser_level = get_level_state(loser, level)

    if not winner_level or not loser_level:
        return None, f"{level}_missing_level_rating", {}

    min_matches = level_min_matches(level, args)
    if winner_level["matches"] < min_matches or loser_level["matches"] < min_matches:
        return None, f"{level}_level_min_sample", {
            "min_matches": min_matches,
            "winner_level_matches": winner_level["matches"],
            "loser_level_matches": loser_level["matches"],
        }

    p = expected_score(float(winner_level["overall_elo"]), float(loser_level["overall_elo"]))
    return p, f"{level}_100_level_overall", {
        "min_matches": min_matches,
        "winner_level_matches": winner_level["matches"],
        "loser_level_matches": loser_level["matches"],
        "winner_level_elo": round(float(winner_level["overall_elo"]), 3),
        "loser_level_elo": round(float(loser_level["overall_elo"]), 3),
    }


def brier(p: float, y: float) -> float:
    return (p - y) ** 2


def log_loss(p: float, y: float) -> float:
    eps = 1e-15
    p = min(max(p, eps), 1.0 - eps)
    return -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))


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


def odds_market_for_event(odds_payload: dict[str, Any], event_key: str) -> dict[str, Any] | None:
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


def chosen_book_odds(market: dict[str, dict[str, Any]], side: str, bookmaker: str, fallback: str) -> tuple[float | None, str | None]:
    side_odds = market.get(side) or {}
    value = safe_float(side_odds.get(bookmaker))

    if value is not None:
        return value, bookmaker

    if fallback == "none":
        return None, None

    available = [number for number in (safe_float(value) for value in side_odds.values()) if number is not None]
    if not available:
        return None, None

    if fallback == "max":
        return max(available), "fallback_max"
    if fallback == "min":
        return min(available), "fallback_min"

    return float(statistics.median(available)), "fallback_median"


def side_pair_odds(market: dict[str, dict[str, Any]], bookmaker: str, fallback: str) -> tuple[float | None, float | None, str | None]:
    home_odds, home_book = chosen_book_odds(market, "Home", bookmaker, fallback)
    away_odds, away_book = chosen_book_odds(market, "Away", bookmaker, fallback)

    if home_odds is not None and away_odds is not None:
        if home_book == bookmaker and away_book == bookmaker:
            return home_odds, away_odds, bookmaker
        if fallback == "none":
            return None, None, None

    def all_side_values(side: str) -> list[float]:
        side_odds = market.get(side) or {}
        return [number for number in (safe_float(value) for value in side_odds.values()) if number is not None]

    home_values = all_side_values("Home")
    away_values = all_side_values("Away")

    if not home_values or not away_values:
        return None, None, None

    if fallback == "max":
        return max(home_values), max(away_values), "fallback_max_pair"
    if fallback == "min":
        return min(home_values), min(away_values), "fallback_min_pair"

    return float(statistics.median(home_values)), float(statistics.median(away_values)), "fallback_median_pair"


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


def summarize_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        return {"sample": 0, "accuracy": None, "brier": None, "log_loss": None, "avg_probability_winner": None}

    return {
        "sample": len(predictions),
        "accuracy": round(sum(row["correct_pick"] for row in predictions) / len(predictions), 6),
        "brier": round(sum(row["brier"] for row in predictions) / len(predictions), 6),
        "log_loss": round(sum(row["log_loss"] for row in predictions) / len(predictions), 6),
        "avg_probability_winner": round(sum(row["winner_probability"] for row in predictions) / len(predictions), 6),
    }


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
        "avg_model_probability": round(sum(row["model_probability"] for row in bets) / staked, 6),
        "avg_odds": round(sum(row["odds"] for row in bets) / staked, 6),
        "avg_ev": round(sum(row["ev"] for row in bets) / staked, 6),
        "avg_edge": round(sum(row["edge"] for row in bets) / staked, 6),
    }


def summarize_by_field(rows: list[dict[str, Any]], field: str, summarize_fn) -> dict[str, Any]:
    values = sorted({clean(row.get(field)) for row in rows})
    return {value: summarize_fn([row for row in rows if clean(row.get(field)) == value]) for value in values}


def build_priced_rows(
    predictions: list[dict[str, Any]],
    enriched_index: dict[str, dict[str, Any]],
    odds_payload: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    counters = Counter()
    all_rows = []
    value_bets = []

    for prediction in predictions:
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

        actual_winner_api_side = clean(enriched.get("winner_side"))
        if actual_winner_api_side not in {"player_1", "player_2"}:
            counters["skipped_missing_winner_side"] += 1
            continue

        selected_api_side = actual_winner_api_side if prediction["correct_pick"] else opposite_api_side(actual_winner_api_side)
        selected_odds_side = api_side_to_odds_side(selected_api_side or "")
        if not selected_odds_side:
            counters["skipped_could_not_map_selected_side"] += 1
            continue

        home_odds, away_odds, odds_source = side_pair_odds(market, args.bookmaker, args.fallback)
        if home_odds is None or away_odds is None:
            counters["skipped_missing_pair_odds"] += 1
            continue

        selected_odds = home_odds if selected_odds_side == "Home" else away_odds
        home_fair, away_fair, overround = devig_probs(home_odds, away_odds)
        book_probability = home_fair if selected_odds_side == "Home" else away_fair

        winner_probability = float(prediction["winner_probability"])
        predicted_is_actual = bool(prediction["correct_pick"])
        model_probability = winner_probability if predicted_is_actual else 1.0 - winner_probability

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
            "selected_side": selected_odds_side,
            "actual_winner_api_side": actual_winner_api_side,
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

    return all_rows, value_bets, counters


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Walk-forward API overlay backtest: every tournament type uses only its own level/tournament Elo. "
            "No global Elo and no surface Elo are used. Includes odds ROI test."
        )
    )

    parser.add_argument("--manifest", default=str(CANONICAL_MANIFEST))
    parser.add_argument("--enriched", default=str(DEFAULT_ENRICHED))
    parser.add_argument("--odds", default=DEFAULT_ODDS)
    parser.add_argument("--from-date", default=None)
    parser.add_argument("--to-date", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))

    parser.add_argument("--bookmaker", default="Pncl")
    parser.add_argument("--fallback", choices=["none", "median", "max", "min"], default="median")
    parser.add_argument("--min-edge", type=float, default=0.03)
    parser.add_argument("--min-ev", type=float, default=0.0)

    parser.add_argument("--main-min-level-matches", type=int, default=20)
    parser.add_argument("--itf-min-level-matches", type=int, default=5)
    parser.add_argument("--challenger-min-level-matches", type=int, default=5)
    parser.add_argument("--qualifying-min-level-matches", type=int, default=5)
    parser.add_argument("--default-min-level-matches", type=int, default=5)

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
                p_winner, model_name, details = probability_level_only(winner, loser, level, args)

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

    enriched_index = build_enriched_index(args.enriched)
    odds_payload = load_json(args.odds)
    priced_predictions, value_bets, odds_counters = build_priced_rows(predictions, enriched_index, odds_payload, args)

    counters.update(odds_counters)

    thresholds = {}
    for edge_threshold in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]:
        bucket_bets = [
            row for row in priced_predictions
            if row["ev"] >= args.min_ev and row["edge"] >= edge_threshold
        ]
        thresholds[f"edge_gte_{edge_threshold:.2f}"] = summarize_bets(bucket_bets)

    by_level_model = summarize_by_field(predictions, "tour_level", summarize_predictions)
    by_gender_model = summarize_by_field(predictions, "gender", summarize_predictions)

    summary = {
        "generated_at": now_iso(),
        "test_source": "api_tennis",
        "from_date": from_date.isoformat() if from_date else None,
        "to_date": to_date.isoformat() if to_date else None,
        "bookmaker": args.bookmaker,
        "fallback": args.fallback,
        "min_edge": args.min_edge,
        "min_ev": args.min_ev,
        "model_variant": "level_only_by_tournament_type",
        "settings": {
            "main_min_level_matches": args.main_min_level_matches,
            "itf_min_level_matches": args.itf_min_level_matches,
            "challenger_min_level_matches": args.challenger_min_level_matches,
            "qualifying_min_level_matches": args.qualifying_min_level_matches,
            "default_min_level_matches": args.default_min_level_matches,
            "models": {
                "atp_wta": "100% atp_wta level overall Elo",
                "grand_slam": "100% grand_slam level overall Elo",
                "challenger": "100% challenger level overall Elo",
                "itf": "100% itf level overall Elo",
                "qualifying": "100% qualifying level overall Elo",
                "note": "No global Elo and no surface Elo are used.",
            },
        },
        "counters": dict(sorted(counters.items())),
        "model_only": {
            "overall": summarize_predictions(predictions),
            "by_level": by_level_model,
            "by_gender": by_gender_model,
        },
        "odds": {
            "all_priced_model_picks": summarize_bets(priced_predictions),
            "value_bets": summarize_bets(value_bets),
            "value_bets_by_level": summarize_by_field(value_bets, "tour_level", summarize_bets),
            "value_bets_by_gender": summarize_by_field(value_bets, "gender", summarize_bets),
            "thresholds": thresholds,
        },
    }

    output_payload = {
        "schema_version": 1,
        "summary": summary,
        "predictions": predictions,
        "priced_predictions": priced_predictions,
        "value_bets": value_bets,
    }

    report_payload = {
        "schema_version": 1,
        "summary": summary,
        "sample_predictions": predictions[:200],
        "sample_priced_predictions": priced_predictions[:200],
        "sample_value_bets": value_bets[:200],
    }

    save_json(Path(args.output), output_payload)
    save_json(Path(args.report), report_payload)

    print("TLE API OVERLAY LEVEL-ONLY ODDS BACKTEST DONE")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
