from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import statistics
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from tennis_elo.config import ROOT_DIR


DEFAULT_CANONICAL_MANIFEST = (
    ROOT_DIR / "data" / "tle" / "processed" / "canonical" / "tle_matches_manifest.json"
)

DEFAULT_API_PLAYER_MAPPING = (
    ROOT_DIR / "data" / "tle" / "mappings" / "api_player_to_sackmann.json"
)

DEFAULT_OUTPUT = (
    ROOT_DIR / "data" / "tle" / "backtests" / "tle_backtest_ai_results.json"
)

DEFAULT_REPORT = (
    ROOT_DIR / "data" / "tle" / "reports" / "tle_backtest_ai_results_report.json"
)

DEFAULT_CSV = (
    ROOT_DIR / "data" / "tle" / "backtests" / "tle_backtest_ai_results_by_group.csv"
)

DEFAULT_ELO = 1500.0
GLOBAL_K = 24.0
GLOBAL_SURFACE_K = 20.0
LEVEL_K = 24.0
LEVEL_SURFACE_K = 20.0

VALID_SURFACES = {"hard", "clay", "grass", "carpet"}
MAIN_TOUR_LEVELS = {"atp_wta", "grand_slam", "main_tour"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", clean(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def surname_initial_key(value: Any) -> str:
    tokens = normalize_name(value).split()
    if len(tokens) < 2:
        return ""
    first = tokens[0]
    last = tokens[-1]
    return f"{last}|{first[:1]}"


def parse_date(value: Any) -> date | None:
    text = clean(value)
    match = re.search(r"(20\d{2}-\d{2}-\d{2})", text)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
    fields = [
        "group_type",
        "group",
        "bets",
        "wins",
        "losses",
        "pushes",
        "voids",
        "staked",
        "profit",
        "roi",
        "hit_rate",
        "avg_odds",
        "avg_ai_prob",
        "avg_tle_prob",
        "avg_tle_ev",
        "avg_tle_edge",
        "avg_ai_edge",
        "avg_confidence",
        "avg_quality_score",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    temporary.replace(path)


def load_json(path_or_url: str | Path) -> Any:
    text = str(path_or_url)
    if text.startswith("http://") or text.startswith("https://"):
        with urllib.request.urlopen(text, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))

    path = Path(text)
    if not path.is_absolute():
        path = ROOT_DIR / path

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)

    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if not isinstance(payload, dict):
        return []

    likely_keys = [
        "results",
        "picks",
        "bets",
        "value_bets",
        "totals",
        "predictions",
        "rows",
        "data",
        "items",
    ]

    for key in likely_keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    dict_values = [
        value
        for value in payload.values()
        if isinstance(value, dict)
    ]
    if len(dict_values) >= 5:
        return dict_values

    best: list[dict[str, Any]] = []
    for value in payload.values():
        nested = records_from_payload(value)
        if len(nested) > len(best):
            best = nested

    return best


def normalize_result(value: Any) -> str:
    text = clean(value).lower()

    if text in {"win", "won", "winner", "true", "1"}:
        return "win"

    if text in {"loss", "lost", "lose", "false", "0"}:
        return "loss"

    if text in {"push", "void", "cancelled", "canceled", "refund", "postponed", "retired"}:
        return text

    return text or "unknown"


def compute_profit(row: dict[str, Any]) -> float | None:
    existing = safe_float(row.get("profit"))
    if existing is not None:
        return existing

    stake = safe_float(row.get("stake"), 1.0) or 1.0
    odds = safe_float(row.get("odds"))
    result = normalize_result(row.get("result"))

    if result == "win" and odds is not None:
        return stake * (odds - 1.0)

    if result == "loss":
        return -stake

    if result in {"push", "void", "cancelled", "canceled", "refund", "postponed"}:
        return 0.0

    return None


def infer_pick_type(source_name: str, row: dict[str, Any]) -> str:
    text = source_name.lower()
    bucket = clean(row.get("bucket")).lower()
    market = clean(row.get("market")).lower()
    bet = clean(row.get("bet")).lower()

    if "total" in text or "total" in bucket or bucket in {"over", "under", "totals"}:
        return "totals"

    if "over" in market or "under" in market or "over" in bet or "under" in bet:
        return "totals"

    if "value" in text or bucket == "match_winner":
        return "value"

    return "unknown"


def normalize_level(value: Any, event_type: Any = "") -> str:
    level = clean(value).lower()

    if level in {"main_tour", "atp_wta", "grand_slam", "challenger", "itf", "qualifying"}:
        return level

    text = f"{clean(value)} {clean(event_type)}".lower()

    if "itf" in text:
        return "itf"

    if "challenger" in text:
        return "challenger"

    if "qualif" in text:
        return "qualifying"

    if "atp" in text or "wta" in text:
        return "atp_wta"

    return level or "unknown"


def normalize_pick(
    row: dict[str, Any],
    source_name: str,
) -> dict[str, Any] | None:
    result = normalize_result(row.get("result"))
    profit = compute_profit(row)
    date_value = parse_date(row.get("date") or row.get("event_date") or row.get("created_at"))

    if date_value is None:
        return None

    if result == "unknown" and profit is None:
        return None

    if profit is None:
        return None

    stake = safe_float(row.get("stake"), 1.0) or 1.0
    odds = safe_float(row.get("odds"))
    ai_prob = safe_float(row.get("model_prob") or row.get("model_probability"))
    implied_prob = safe_float(row.get("implied_prob") or row.get("implied_probability"))
    ai_edge = safe_float(row.get("edge"))

    if ai_edge is None and ai_prob is not None and implied_prob is not None:
        ai_edge = ai_prob - implied_prob

    ai_ev = None
    if ai_prob is not None and odds is not None:
        ai_ev = ai_prob * odds - 1.0

    return {
        "source_name": source_name,
        "pick_type": infer_pick_type(source_name, row),
        "pick_id": clean(row.get("pick_id")),
        "event_key": clean(row.get("event_key") or row.get("fixture_id")),
        "model_version": clean(row.get("model_version")),
        "date": date_value.isoformat(),
        "time": clean(row.get("time")),
        "match": clean(row.get("match")),
        "bet": clean(row.get("bet")),
        "bucket": clean(row.get("bucket")),
        "side": clean(row.get("side") or row.get("market_side")).lower(),
        "market_side": clean(row.get("market_side") or row.get("side")),
        "player_key_api": safe_int(row.get("player_key")),
        "opponent_key_api": safe_int(row.get("opponent_key")),
        "player_name": clean(row.get("player_name") or row.get("bet")),
        "opponent_name": clean(row.get("opponent_name")),
        "tour_level": normalize_level(row.get("tour_level"), row.get("event_type")),
        "gender": clean(row.get("gender")).lower(),
        "event_type": clean(row.get("event_type")),
        "surface": clean(row.get("surface")).lower(),
        "tournament": clean(row.get("tournament")),
        "round": clean(row.get("round")),
        "odds": odds,
        "bookmaker": clean(row.get("best_bookmaker") or row.get("bookmaker")),
        "market_median_odds": safe_float(row.get("market_median_odds")),
        "bookmakers_used": safe_int(row.get("bookmakers_used")),
        "ai_prob": ai_prob,
        "implied_prob": implied_prob,
        "ai_edge": ai_edge,
        "ai_ev": ai_ev,
        "confidence": safe_float(row.get("confidence")),
        "quality_score": safe_float(row.get("quality_score")),
        "favorite_type": clean(row.get("favorite_type")),
        "stake_label": clean(row.get("stake_label")),
        "stake": stake,
        "result": result,
        "profit": profit,
        "settled_status": clean(row.get("settled_status")),
        "event_winner": clean(row.get("event_winner")),
        "final_score": clean(row.get("final_score")),
        "created_at": clean(row.get("created_at")),
        "settled_at": clean(row.get("settled_at")),
    }


def load_ai_picks(inputs: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    picks = []
    sources = []

    for item in inputs:
        source_name = item
        payload = load_json(item)
        records = records_from_payload(payload)
        usable = []

        for row in records:
            pick = normalize_pick(row, source_name)
            if pick is not None:
                usable.append(pick)

        picks.extend(usable)
        sources.append(
            {
                "source": source_name,
                "raw_records": len(records),
                "usable_picks": len(usable),
                "pick_types": dict(Counter(row["pick_type"] for row in usable)),
                "date_min": min((row["date"] for row in usable), default=None),
                "date_max": max((row["date"] for row in usable), default=None),
            }
        )

    picks.sort(key=lambda row: (row["date"], row["time"], row["event_key"], row["pick_id"]))
    return picks, sources


def iter_canonical_matches(manifest_path: Path):
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


def player_identity_from_match(player: dict[str, Any], gender: str) -> tuple[str | None, str]:
    name = clean(player.get("name"))
    sackmann_id = player.get("sackmann_player_id")

    if sackmann_id not in {None, ""}:
        try:
            return f"{gender}:sackmann:{int(sackmann_id)}", name
        except (TypeError, ValueError):
            pass

    api_key = player.get("api_player_key")
    if api_key not in {None, ""}:
        try:
            return f"{gender}:api:{int(api_key)}", name
        except (TypeError, ValueError):
            pass

    if name:
        return f"{gender}:name:{normalize_name(name)}", name

    return None, ""


def build_player_alias_index(manifest_path: Path) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, str]]:
    exact = defaultdict(set)
    surname_initial = defaultdict(set)
    display_names = {}

    for match in iter_canonical_matches(manifest_path):
        gender = clean(match.get("gender")).lower()
        if gender not in {"men", "women"}:
            continue

        for side in ("winner", "loser"):
            player = match.get(side) or {}
            if not isinstance(player, dict):
                continue

            player_key, name = player_identity_from_match(player, gender)
            if not player_key or not name:
                continue

            display_names.setdefault(player_key, name)

            norm = normalize_name(name)
            if norm:
                exact[f"{gender}|{norm}"].add(player_key)

            si = surname_initial_key(name)
            if si:
                surname_initial[f"{gender}|{si}"].add(player_key)

    return (
        {key: sorted(values) for key, values in exact.items()},
        {key: sorted(values) for key, values in surname_initial.items()},
        display_names,
    )


def load_api_player_mapping(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    payload = load_json(path)
    players = payload.get("players") if isinstance(payload, dict) else None
    if not isinstance(players, dict):
        return {}

    return {
        str(key): value
        for key, value in players.items()
        if isinstance(value, dict)
    }


def resolve_pick_player(
    api_key: int | None,
    name: str,
    gender: str,
    api_mapping: dict[str, dict[str, Any]],
    exact_index: dict[str, list[str]],
    surname_initial_index: dict[str, list[str]],
) -> tuple[str | None, str]:
    if gender not in {"men", "women"}:
        return None, "invalid_gender"

    if api_key is not None:
        mapping = api_mapping.get(str(api_key))
        if (
            isinstance(mapping, dict)
            and mapping.get("status") == "matched"
            and clean(mapping.get("gender")).lower() == gender
            and mapping.get("sackmann_player_id") not in {None, ""}
        ):
            try:
                return f"{gender}:sackmann:{int(mapping['sackmann_player_id'])}", "api_mapping"
            except (TypeError, ValueError):
                pass

    norm = normalize_name(name)
    if norm:
        candidates = exact_index.get(f"{gender}|{norm}", [])
        if len(candidates) == 1:
            return candidates[0], "exact_name"

    si = surname_initial_key(name)
    if si:
        candidates = surname_initial_index.get(f"{gender}|{si}", [])
        if len(candidates) == 1:
            return candidates[0], "unique_surname_initial"

    if api_key is not None:
        return f"{gender}:api:{api_key}", "api_key_unmapped"

    if norm:
        return f"{gender}:name:{norm}", "name_fallback"

    return None, "unresolved"


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


def ensure_level(player: dict[str, Any], level: str) -> dict[str, Any]:
    if level not in player["levels"]:
        player["levels"][level] = new_level_state()
    return player["levels"][level]


def ensure_surface(container: dict[str, Any], surface: str) -> dict[str, Any]:
    if surface not in container["surfaces"]:
        container["surfaces"][surface] = new_surface_state()
    return container["surfaces"][surface]


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

    winner_key, winner_name = player_identity_from_match(winner_raw, gender)
    loser_key, loser_name = player_identity_from_match(loser_raw, gender)

    if not winner_key or not loser_key or winner_key == loser_key:
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


def get_level_surface_state(
    player: dict[str, Any],
    level: str,
    surface: str,
) -> dict[str, Any] | None:
    level_state = get_level_state(player, level)
    if not level_state:
        return None
    return level_state.get("surfaces", {}).get(surface)


def model_probability_for_selected(
    selected: dict[str, Any],
    opponent: dict[str, Any],
    level: str,
    surface: str,
    args: argparse.Namespace,
) -> tuple[float | None, str, dict[str, Any]]:
    if level in MAIN_TOUR_LEVELS:
        model_level = "atp_wta"
        selected_level = get_level_state(selected, model_level)
        opponent_level = get_level_state(opponent, model_level)

        if not selected_level or not opponent_level:
            return None, "main_tour_missing_level_rating", {}

        if (
            selected_level["matches"] < args.main_min_level_matches
            or opponent_level["matches"] < args.main_min_level_matches
        ):
            return None, "main_tour_level_min_sample", {}

        if surface not in VALID_SURFACES:
            return None, "main_tour_unknown_surface", {}

        selected_surface = get_level_surface_state(selected, model_level, surface)
        opponent_surface = get_level_surface_state(opponent, model_level, surface)

        if not selected_surface or not opponent_surface:
            return None, "main_tour_missing_surface_rating", {}

        if (
            selected_surface["matches"] < args.main_min_surface_matches
            or opponent_surface["matches"] < args.main_min_surface_matches
        ):
            return None, "main_tour_surface_min_sample", {}

        p_level = expected_score(
            float(selected_level["overall_elo"]),
            float(opponent_level["overall_elo"]),
        )
        p_surface = expected_score(
            float(selected_surface["elo"]),
            float(opponent_surface["elo"]),
        )
        p = 0.80 * p_level + 0.20 * p_surface

        return p, "main_tour_80_level_20_surface", {
            "selected_level_matches": selected_level["matches"],
            "opponent_level_matches": opponent_level["matches"],
            "selected_surface_matches": selected_surface["matches"],
            "opponent_surface_matches": opponent_surface["matches"],
            "p_level": p_level,
            "p_surface": p_surface,
        }

    if level == "itf":
        selected_level = get_level_state(selected, "itf")
        opponent_level = get_level_state(opponent, "itf")

        if not selected_level or not opponent_level:
            return None, "itf_missing_level_rating", {}

        if (
            selected_level["matches"] < args.itf_min_level_matches
            or opponent_level["matches"] < args.itf_min_level_matches
        ):
            return None, "itf_level_min_sample", {}

        p = expected_score(
            float(selected_level["overall_elo"]),
            float(opponent_level["overall_elo"]),
        )
        return p, "itf_100_level_overall", {
            "selected_level_matches": selected_level["matches"],
            "opponent_level_matches": opponent_level["matches"],
        }

    if level == "challenger":
        selected_level = get_level_state(selected, "challenger")
        opponent_level = get_level_state(opponent, "challenger")

        if not selected_level or not opponent_level:
            return None, "challenger_missing_level_rating", {}

        if (
            selected_level["matches"] < args.challenger_min_level_matches
            or opponent_level["matches"] < args.challenger_min_level_matches
        ):
            return None, "challenger_level_min_sample", {}

        p = expected_score(
            float(selected_level["overall_elo"]),
            float(opponent_level["overall_elo"]),
        )
        return p, "challenger_100_level_overall", {
            "selected_level_matches": selected_level["matches"],
            "opponent_level_matches": opponent_level["matches"],
        }

    if level == "qualifying":
        return None, "qualifying_no_bet", {}

    return None, "unsupported_level", {}


def odds_bucket(odds: float | None) -> str:
    if odds is None:
        return "missing"
    if odds < 1.30:
        return "<1.30"
    if odds < 1.60:
        return "1.30-1.59"
    if odds < 2.00:
        return "1.60-1.99"
    if odds < 2.50:
        return "2.00-2.49"
    if odds < 3.00:
        return "2.50-2.99"
    if odds < 4.00:
        return "3.00-3.99"
    return ">=4.00"


def edge_bucket(edge: float | None) -> str:
    if edge is None:
        return "missing"
    if edge < 0.00:
        return "<0"
    if edge < 0.03:
        return "0.00-0.029"
    if edge < 0.05:
        return "0.03-0.049"
    if edge < 0.08:
        return "0.05-0.079"
    if edge < 0.10:
        return "0.08-0.099"
    if edge < 0.15:
        return "0.10-0.149"
    return ">=0.15"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "voids": 0,
            "staked": 0,
            "profit": 0,
            "roi": None,
            "hit_rate": None,
            "avg_odds": None,
            "avg_ai_prob": None,
            "avg_tle_prob": None,
            "avg_tle_ev": None,
            "avg_tle_edge": None,
            "avg_ai_edge": None,
            "avg_confidence": None,
            "avg_quality_score": None,
        }

    wins = sum(1 for row in rows if row["result"] == "win")
    losses = sum(1 for row in rows if row["result"] == "loss")
    pushes = sum(1 for row in rows if row["result"] == "push")
    voids = sum(
        1
        for row in rows
        if row["result"] in {"void", "cancelled", "canceled", "refund", "postponed"}
    )
    staked = sum(row["stake"] for row in rows)
    profit = sum(row["profit"] for row in rows)

    def avg(key: str) -> float | None:
        values = [
            row[key]
            for row in rows
            if row.get(key) is not None
        ]
        if not values:
            return None
        return round(sum(values) / len(values), 6)

    return {
        "bets": len(rows),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "voids": voids,
        "staked": round(staked, 6),
        "profit": round(profit, 6),
        "roi": round(profit / staked, 6) if staked else None,
        "hit_rate": round(wins / (wins + losses), 6) if wins + losses else None,
        "avg_odds": avg("odds"),
        "avg_ai_prob": avg("ai_prob"),
        "avg_tle_prob": avg("tle_prob"),
        "avg_tle_ev": avg("tle_ev"),
        "avg_tle_edge": avg("tle_edge"),
        "avg_ai_edge": avg("ai_edge"),
        "avg_confidence": avg("confidence"),
        "avg_quality_score": avg("quality_score"),
    }


def grouped(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups = defaultdict(list)
    for row in rows:
        groups[clean(row.get(key)) or "missing"].append(row)
    return {
        group: summarize(group_rows)
        for group, group_rows in sorted(groups.items())
    }


def grouped_custom(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    groups = defaultdict(list)
    for row in rows:
        if name == "tle_edge_bucket":
            group = edge_bucket(row.get("tle_edge"))
        elif name == "ai_edge_bucket":
            group = edge_bucket(row.get("ai_edge"))
        elif name == "odds_bucket":
            group = odds_bucket(row.get("odds"))
        elif name == "month":
            group = clean(row.get("date"))[:7] or "missing"
        elif name == "tle_prob_bucket":
            p = row.get("tle_prob")
            if p is None:
                group = "missing"
            elif p < 0.45:
                group = "<0.45"
            elif p < 0.50:
                group = "0.45-0.499"
            elif p < 0.55:
                group = "0.50-0.549"
            elif p < 0.60:
                group = "0.55-0.599"
            elif p < 0.65:
                group = "0.60-0.649"
            elif p < 0.70:
                group = "0.65-0.699"
            else:
                group = ">=0.70"
        else:
            group = "unknown"
        groups[group].append(row)

    return {
        group: summarize(group_rows)
        for group, group_rows in sorted(groups.items())
    }


def csv_rows(groups: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for group_type, payload in groups.items():
        for group, summary in payload.items():
            rows.append({"group_type": group_type, "group": group, **summary})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Walk-forward TLE Elo backtest on settled AI tennis results JSONs. "
            "It prices each AI pick with pre-match TLE level Elo, then evaluates actual AI result/profit."
        )
    )

    parser.add_argument(
        "--ai-json",
        required=True,
        help="Comma separated local paths or raw GitHub URLs to AI tennis result JSON files.",
    )
    parser.add_argument("--canonical-manifest", default=str(DEFAULT_CANONICAL_MANIFEST))
    parser.add_argument("--api-player-mapping", default=str(DEFAULT_API_PLAYER_MAPPING))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--min-date", default="")
    parser.add_argument("--max-date", default="")

    parser.add_argument("--min-edge", type=float, default=0.03)
    parser.add_argument("--min-ev", type=float, default=0.0)
    parser.add_argument("--main-min-level-matches", type=int, default=20)
    parser.add_argument("--main-min-surface-matches", type=int, default=10)
    parser.add_argument("--itf-min-level-matches", type=int, default=5)
    parser.add_argument("--challenger-min-level-matches", type=int, default=5)

    args = parser.parse_args()

    ai_inputs = [
        item.strip()
        for item in args.ai_json.split(",")
        if item.strip()
    ]

    if not ai_inputs:
        raise RuntimeError("--ai-json is required")

    picks, source_summaries = load_ai_picks(ai_inputs)

    if args.min_date:
        picks = [row for row in picks if row["date"] >= args.min_date]
    if args.max_date:
        picks = [row for row in picks if row["date"] <= args.max_date]

    canonical_manifest = Path(args.canonical_manifest)
    if not canonical_manifest.is_absolute():
        canonical_manifest = ROOT_DIR / canonical_manifest

    api_mapping_path = Path(args.api_player_mapping)
    if not api_mapping_path.is_absolute():
        api_mapping_path = ROOT_DIR / api_mapping_path

    api_mapping = load_api_player_mapping(api_mapping_path)
    exact_index, surname_initial_index, display_names = build_player_alias_index(canonical_manifest)

    matches = list(iter_canonical_matches(canonical_manifest))
    match_index = 0
    players: dict[str, dict[str, Any]] = {}

    counters = Counter()
    priced_rows = []
    tle_value_rows = []

    for pick in picks:
        pick_date = date.fromisoformat(pick["date"])

        # Conservative walk-forward: only matches with date strictly before pick date.
        while match_index < len(matches):
            match_date = parse_date(matches[match_index].get("date"))
            if match_date is None or match_date >= pick_date:
                break
            update_state_for_match(matches[match_index], players)
            match_index += 1

        counters["ai_picks_seen"] += 1

        if pick["pick_type"] != "value":
            counters["skipped_non_match_winner_pick_type"] += 1
            continue

        gender = pick["gender"]
        level = pick["tour_level"]
        surface = pick["surface"]

        selected_key, selected_method = resolve_pick_player(
            pick["player_key_api"],
            pick["player_name"],
            gender,
            api_mapping,
            exact_index,
            surname_initial_index,
        )
        opponent_key, opponent_method = resolve_pick_player(
            pick["opponent_key_api"],
            pick["opponent_name"],
            gender,
            api_mapping,
            exact_index,
            surname_initial_index,
        )

        if not selected_key or not opponent_key:
            counters["skipped_unresolved_player"] += 1
            continue

        selected = players.get(selected_key)
        opponent = players.get(opponent_key)

        if not selected or not opponent:
            counters["skipped_missing_player_history"] += 1
            counters[f"selected_resolve_{selected_method}"] += 1
            counters[f"opponent_resolve_{opponent_method}"] += 1
            continue

        tle_prob, tle_model, details = model_probability_for_selected(
            selected,
            opponent,
            level,
            surface,
            args,
        )

        counters[f"selected_resolve_{selected_method}"] += 1
        counters[f"opponent_resolve_{opponent_method}"] += 1

        if tle_prob is None:
            counters[f"skipped_{tle_model}"] += 1
            continue

        odds = pick["odds"]
        if odds is None:
            counters["skipped_missing_odds"] += 1
            continue

        implied = pick["implied_prob"]
        if implied is None and odds:
            implied = 1.0 / odds

        tle_ev = tle_prob * odds - 1.0
        tle_edge = tle_prob - implied if implied is not None else None

        row = {
            **pick,
            "selected_tle_key": selected_key,
            "opponent_tle_key": opponent_key,
            "selected_resolve_method": selected_method,
            "opponent_resolve_method": opponent_method,
            "tle_prob": round(tle_prob, 6),
            "tle_ev": round(tle_ev, 6),
            "tle_edge": round(tle_edge, 6) if tle_edge is not None else None,
            "tle_model": tle_model,
            "tle_details": details,
            "tle_selected_display_name": selected.get("display_name"),
            "tle_opponent_display_name": opponent.get("display_name"),
        }

        priced_rows.append(row)
        counters["tle_priced_ai_picks"] += 1
        counters[f"tle_priced_level_{level}"] += 1

        if (
            tle_ev >= args.min_ev
            and tle_edge is not None
            and tle_edge >= args.min_edge
        ):
            tle_value_rows.append(row)
            counters["tle_value_ai_picks"] += 1
            counters[f"tle_value_level_{level}"] += 1

    groups = {
        "all_by_pick_type": grouped(priced_rows, "pick_type"),
        "all_by_model_version": grouped(priced_rows, "model_version"),
        "all_by_tour_level": grouped(priced_rows, "tour_level"),
        "all_by_gender": grouped(priced_rows, "gender"),
        "all_by_favorite_type": grouped(priced_rows, "favorite_type"),
        "all_by_stake_label": grouped(priced_rows, "stake_label"),
        "all_by_bookmaker": grouped(priced_rows, "bookmaker"),
        "all_by_odds_bucket": grouped_custom(priced_rows, "odds_bucket"),
        "all_by_ai_edge_bucket": grouped_custom(priced_rows, "ai_edge_bucket"),
        "all_by_tle_edge_bucket": grouped_custom(priced_rows, "tle_edge_bucket"),
        "all_by_tle_prob_bucket": grouped_custom(priced_rows, "tle_prob_bucket"),
        "all_by_month": grouped_custom(priced_rows, "month"),
        "tle_value_by_tour_level": grouped(tle_value_rows, "tour_level"),
        "tle_value_by_gender": grouped(tle_value_rows, "gender"),
        "tle_value_by_odds_bucket": grouped_custom(tle_value_rows, "odds_bucket"),
        "tle_value_by_tle_edge_bucket": grouped_custom(tle_value_rows, "tle_edge_bucket"),
        "tle_value_by_tle_prob_bucket": grouped_custom(tle_value_rows, "tle_prob_bucket"),
    }

    threshold_summaries = {}
    for edge_threshold in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]:
        filtered = [
            row
            for row in priced_rows
            if row["tle_ev"] >= args.min_ev
            and row["tle_edge"] is not None
            and row["tle_edge"] >= edge_threshold
        ]
        threshold_summaries[f"tle_edge_gte_{edge_threshold:.2f}"] = summarize(filtered)

    summary = {
        "generated_at": now_iso(),
        "settings": {
            "ai_json": ai_inputs,
            "min_date": args.min_date or None,
            "max_date": args.max_date or None,
            "min_edge": args.min_edge,
            "min_ev": args.min_ev,
            "main_min_level_matches": args.main_min_level_matches,
            "main_min_surface_matches": args.main_min_surface_matches,
            "itf_min_level_matches": args.itf_min_level_matches,
            "challenger_min_level_matches": args.challenger_min_level_matches,
            "walk_forward_note": "Only canonical matches with date strictly before pick date are included in TLE state.",
        },
        "source_summaries": source_summaries,
        "counters": dict(sorted(counters.items())),
        "all_ai_picks_input": summarize(picks),
        "tle_priced_ai_picks": summarize(priced_rows),
        "tle_value_ai_picks": summarize(tle_value_rows),
        "thresholds": threshold_summaries,
    }

    payload = {
        "schema_version": 1,
        "summary": summary,
        "groups": groups,
        "sample_tle_value_picks": tle_value_rows[:300],
        "sample_tle_priced_picks": priced_rows[:300],
    }

    report = {
        "schema_version": 1,
        "summary": summary,
        "key_groups": {
            key: groups[key]
            for key in [
                "all_by_tour_level",
                "all_by_odds_bucket",
                "all_by_tle_edge_bucket",
                "tle_value_by_tour_level",
                "tle_value_by_tle_edge_bucket",
            ]
        },
    }

    save_json(Path(args.output), payload)
    save_json(Path(args.report), report)
    save_csv(Path(args.csv), csv_rows(groups))

    print("TLE BACKTEST AI RESULTS DONE")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nKEY GROUPS:")
    print(json.dumps(report["key_groups"], indent=2, ensure_ascii=False))
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")
    print(f"CSV:    {args.csv}")


if __name__ == "__main__":
    main()
