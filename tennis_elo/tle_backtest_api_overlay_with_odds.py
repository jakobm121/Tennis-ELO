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

DEFAULT_ODDS_URL = (
    "https://raw.githubusercontent.com/jakobm121/Ai/refs/heads/main/"
    "data/tle/odds_backfill/tle_odds_backfill.json"
)

DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "tle" / "backtests"

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
    return f"{tokens[-1]}|{tokens[0][:1]}"


def parse_date(value: Any) -> date | None:
    text = clean(value)
    match = re.search(r"(20\d{2}-\d{2}-\d{2})", text)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def safe_decimal_odds(value: Any, min_odds: float, max_odds: float) -> float | None:
    number = safe_float(value)
    if number is None:
        return None
    if number < min_odds or number > max_odds:
        return None
    return number


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_json(path_or_url: str | Path) -> Any:
    text = str(path_or_url)

    if text.startswith("http://") or text.startswith("https://"):
        with urllib.request.urlopen(text, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))

    path = Path(text)
    if not path.is_absolute():
        path = ROOT_DIR / path

    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def read_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def manifest_year_files(manifest: dict[str, Any]) -> list[str]:
    files = []

    for key in ("year_files", "files", "canonical_files"):
        value = manifest.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    path = item.get("path") or item.get("file") or item.get("filename")
                    if path:
                        files.append(str(path))
                elif isinstance(item, str):
                    files.append(item)

    return files


def iter_canonical_matches(manifest_path: Path):
    manifest = load_json(manifest_path)
    rows = []

    for rel in manifest_year_files(manifest):
        path = Path(rel)
        if not path.is_absolute():
            path = ROOT_DIR / path

        if not path.exists():
            continue

        for row in read_jsonl_gz(path):
            rows.append(row)

    rows.sort(
        key=lambda row: (
            clean(row.get("date")),
            clean(row.get("event_key") or row.get("api_event_key") or row.get("tle_match_id")),
        )
    )

    yield from rows


def get_event_key(row: dict[str, Any]) -> str:
    for key in (
        "event_key",
        "event_id",
        "api_event_key",
        "api_event_id",
        "fixture_id",
        "tle_source_event_id",
    ):
        value = clean(row.get(key))
        if value:
            return value

    api = row.get("api")
    if isinstance(api, dict):
        for key in ("event_key", "event_id", "api_event_key", "fixture_id"):
            value = clean(api.get(key))
            if value:
                return value

    source = row.get("source_record")
    if isinstance(source, dict):
        for key in ("event_key", "event_id", "api_event_key", "fixture_id"):
            value = clean(source.get(key))
            if value:
                return value

    return ""


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


def build_alias_indexes(manifest_path: Path) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, str]]:
    exact = defaultdict(set)
    surname_initial = defaultdict(set)
    display = {}

    for match in iter_canonical_matches(manifest_path):
        gender = clean(match.get("gender")).lower()
        if gender not in {"men", "women"}:
            continue

        for side in ("winner", "loser"):
            player = match.get(side) or {}
            if not isinstance(player, dict):
                continue

            key, name = player_identity_from_match(player, gender)
            if not key or not name:
                continue

            display.setdefault(key, name)

            norm = normalize_name(name)
            if norm:
                exact[f"{gender}|{norm}"].add(key)

            si = surname_initial_key(name)
            if si:
                surname_initial[f"{gender}|{si}"].add(key)

    return (
        {key: sorted(values) for key, values in exact.items()},
        {key: sorted(values) for key, values in surname_initial.items()},
        display,
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


def resolve_player(
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
        mapped = api_mapping.get(str(api_key))

        if (
            isinstance(mapped, dict)
            and mapped.get("status") == "matched"
            and clean(mapped.get("gender")).lower() == gender
            and mapped.get("sackmann_player_id") not in {None, ""}
        ):
            try:
                return f"{gender}:sackmann:{int(mapped['sackmann_player_id'])}", "api_mapping"
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


def update_pair(winner_rating: float, loser_rating: float, k: float) -> tuple[float, float]:
    expected = expected_score(winner_rating, loser_rating)
    change = k * (1.0 - expected)
    return winner_rating + change, loser_rating - change


def new_surface_state() -> dict[str, Any]:
    return {"elo": DEFAULT_ELO, "matches": 0, "wins": 0}


def new_level_state() -> dict[str, Any]:
    return {"overall_elo": DEFAULT_ELO, "matches": 0, "wins": 0, "surfaces": {}}


def new_player_state(key: str, name: str, gender: str) -> dict[str, Any]:
    return {
        "player_key": key,
        "display_name": name,
        "gender": gender,
        "global": {
            "overall_elo": DEFAULT_ELO,
            "matches": 0,
            "wins": 0,
            "surfaces": {},
        },
        "levels": {},
    }


def ensure_player(players: dict[str, dict[str, Any]], key: str, name: str, gender: str) -> dict[str, Any]:
    if key not in players:
        players[key] = new_player_state(key, name, gender)

    if name and not players[key].get("display_name"):
        players[key]["display_name"] = name

    return players[key]


def ensure_level(player: dict[str, Any], level: str) -> dict[str, Any]:
    if level not in player["levels"]:
        player["levels"][level] = new_level_state()
    return player["levels"][level]


def ensure_surface(container: dict[str, Any], surface: str) -> dict[str, Any]:
    if surface not in container["surfaces"]:
        container["surfaces"][surface] = new_surface_state()
    return container["surfaces"][surface]


def update_rating_layer(winner: dict[str, Any], loser: dict[str, Any], field: str, k: float) -> None:
    winner_new, loser_new = update_pair(float(winner[field]), float(loser[field]), k)
    winner[field] = winner_new
    loser[field] = loser_new


def update_state_for_match(match: dict[str, Any], players: dict[str, dict[str, Any]]) -> bool:
    if not match.get("ready_for_tle"):
        return False

    gender = clean(match.get("gender")).lower()
    level = clean(match.get("tour_level")).lower()
    surface = clean((match.get("tournament") or {}).get("surface")).lower()

    if not surface:
        surface = clean(match.get("surface")).lower()

    if gender not in {"men", "women"}:
        return False

    winner_raw = match.get("winner") or {}
    loser_raw = match.get("loser") or {}

    if not isinstance(winner_raw, dict) or not isinstance(loser_raw, dict):
        return False

    winner_key, winner_name = player_identity_from_match(winner_raw, gender)
    loser_key, loser_name = player_identity_from_match(loser_raw, gender)

    if not winner_key or not loser_key or winner_key == loser_key:
        return False

    winner = ensure_player(players, winner_key, winner_name, gender)
    loser = ensure_player(players, loser_key, loser_name, gender)

    update_rating_layer(winner["global"], loser["global"], "overall_elo", GLOBAL_K)
    winner["global"]["matches"] += 1
    loser["global"]["matches"] += 1
    winner["global"]["wins"] += 1

    if surface in VALID_SURFACES:
        ws = ensure_surface(winner["global"], surface)
        ls = ensure_surface(loser["global"], surface)

        update_rating_layer(ws, ls, "elo", GLOBAL_SURFACE_K)

        ws["matches"] += 1
        ls["matches"] += 1
        ws["wins"] += 1

    wl = ensure_level(winner, level)
    ll = ensure_level(loser, level)

    update_rating_layer(wl, ll, "overall_elo", LEVEL_K)

    wl["matches"] += 1
    ll["matches"] += 1
    wl["wins"] += 1

    if surface in VALID_SURFACES:
        wls = ensure_surface(wl, surface)
        lls = ensure_surface(ll, surface)

        update_rating_layer(wls, lls, "elo", LEVEL_SURFACE_K)

        wls["matches"] += 1
        lls["matches"] += 1
        wls["wins"] += 1

    return True


def get_level_state(player: dict[str, Any], level: str) -> dict[str, Any] | None:
    return player.get("levels", {}).get(level)


def get_level_surface_state(player: dict[str, Any], level: str, surface: str) -> dict[str, Any] | None:
    level_state = get_level_state(player, level)
    if not level_state:
        return None
    return level_state.get("surfaces", {}).get(surface)


def probability_for_player_1(
    player_1: dict[str, Any],
    player_2: dict[str, Any],
    level: str,
    surface: str,
    args: argparse.Namespace,
) -> tuple[float | None, str, dict[str, Any]]:
    if level in MAIN_TOUR_LEVELS:
        model_level = "atp_wta"

        p1_level = get_level_state(player_1, model_level)
        p2_level = get_level_state(player_2, model_level)

        if not p1_level or not p2_level:
            return None, "main_tour_missing_level_rating", {}

        if (
            p1_level["matches"] < args.main_min_level_matches
            or p2_level["matches"] < args.main_min_level_matches
        ):
            return None, "main_tour_level_min_sample", {}

        if surface not in VALID_SURFACES:
            return None, "main_tour_unknown_surface", {}

        p1_surface = get_level_surface_state(player_1, model_level, surface)
        p2_surface = get_level_surface_state(player_2, model_level, surface)

        if not p1_surface or not p2_surface:
            return None, "main_tour_missing_surface_rating", {}

        if (
            p1_surface["matches"] < args.main_min_surface_matches
            or p2_surface["matches"] < args.main_min_surface_matches
        ):
            return None, "main_tour_surface_min_sample", {}

        p_level = expected_score(float(p1_level["overall_elo"]), float(p2_level["overall_elo"]))
        p_surface = expected_score(float(p1_surface["elo"]), float(p2_surface["elo"]))
        probability = 0.80 * p_level + 0.20 * p_surface

        return probability, "main_tour_80_level_20_surface", {
            "p1_level_matches": p1_level["matches"],
            "p2_level_matches": p2_level["matches"],
            "p1_surface_matches": p1_surface["matches"],
            "p2_surface_matches": p2_surface["matches"],
            "p1_level_elo": round(float(p1_level["overall_elo"]), 3),
            "p2_level_elo": round(float(p2_level["overall_elo"]), 3),
            "p1_surface_elo": round(float(p1_surface["elo"]), 3),
            "p2_surface_elo": round(float(p2_surface["elo"]), 3),
            "p_level": round(p_level, 6),
            "p_surface": round(p_surface, 6),
        }

    if level == "itf":
        p1_level = get_level_state(player_1, "itf")
        p2_level = get_level_state(player_2, "itf")

        if not p1_level or not p2_level:
            return None, "itf_missing_level_rating", {}

        if (
            p1_level["matches"] < args.itf_min_level_matches
            or p2_level["matches"] < args.itf_min_level_matches
        ):
            return None, "itf_level_min_sample", {}

        probability = expected_score(
            float(p1_level["overall_elo"]),
            float(p2_level["overall_elo"]),
        )

        return probability, "itf_100_level_overall", {
            "p1_level_matches": p1_level["matches"],
            "p2_level_matches": p2_level["matches"],
            "p1_level_elo": round(float(p1_level["overall_elo"]), 3),
            "p2_level_elo": round(float(p2_level["overall_elo"]), 3),
        }

    if level == "challenger":
        p1_level = get_level_state(player_1, "challenger")
        p2_level = get_level_state(player_2, "challenger")

        if not p1_level or not p2_level:
            return None, "challenger_missing_level_rating", {}

        if (
            p1_level["matches"] < args.challenger_min_level_matches
            or p2_level["matches"] < args.challenger_min_level_matches
        ):
            return None, "challenger_level_min_sample", {}

        probability = expected_score(
            float(p1_level["overall_elo"]),
            float(p2_level["overall_elo"]),
        )

        return probability, "challenger_100_level_overall", {
            "p1_level_matches": p1_level["matches"],
            "p2_level_matches": p2_level["matches"],
            "p1_level_elo": round(float(p1_level["overall_elo"]), 3),
            "p2_level_elo": round(float(p2_level["overall_elo"]), 3),
        }

    if level == "qualifying":
        return None, "qualifying_no_bet", {}

    return None, "unsupported_level", {}


def collect_book_odds(side_payload: dict[str, Any], min_odds: float, max_odds: float) -> list[float]:
    values: list[float] = []

    books = side_payload.get("books")
    if isinstance(books, dict):
        for value in books.values():
            odds = safe_decimal_odds(value, min_odds, max_odds)
            if odds is not None:
                values.append(odds)

    for key in ("odds", "price", "decimal_odds"):
        odds = safe_decimal_odds(side_payload.get(key), min_odds, max_odds)
        if odds is not None:
            values.append(odds)

    return values


def choose_side_odds(
    side_payload: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[float | None, str | None, int, list[float]]:
    books = side_payload.get("books")

    if args.odds_mode == "bookmaker":
        if isinstance(books, dict):
            odds = safe_decimal_odds(
                books.get(args.bookmaker),
                args.min_odds,
                args.max_odds,
            )
            if odds is not None:
                return odds, args.bookmaker, 1, [odds]

        return None, None, 0, []

    values = collect_book_odds(side_payload, args.min_odds, args.max_odds)

    if not values:
        for key in ("median_odds", "best_odds"):
            odds = safe_decimal_odds(side_payload.get(key), args.min_odds, args.max_odds)
            if odds is not None:
                values.append(odds)

    if not values:
        return None, None, 0, []

    values = sorted(values)

    if args.odds_mode == "clean_average":
        return float(sum(values) / len(values)), "clean_average", len(values), values

    if args.odds_mode == "clean_median":
        return float(statistics.median(values)), "clean_median", len(values), values

    if args.odds_mode == "best":
        return max(values), "best_clean", len(values), values

    if args.odds_mode == "worst":
        return min(values), "worst_clean", len(values), values

    return float(sum(values) / len(values)), "clean_average", len(values), values


def devig_pair(p1_odds: float, p2_odds: float) -> tuple[float, float, float]:
    raw_p1 = 1.0 / p1_odds
    raw_p2 = 1.0 / p2_odds
    total = raw_p1 + raw_p2

    if total <= 0:
        return 0.5, 0.5, 0.0

    return raw_p1 / total, raw_p2 / total, total - 1.0


def extract_matches_from_odds_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]

    if not isinstance(payload, dict):
        return []

    for key in ("matches", "odds", "events", "fixtures", "data", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]

    return []


def normalize_odds_row(row: dict[str, Any]) -> dict[str, Any] | None:
    event_key = get_event_key(row)
    if not event_key:
        return None

    markets = row.get("markets")
    if not isinstance(markets, dict):
        markets = {}

    match_winner = markets.get("match_winner")
    if not isinstance(match_winner, dict):
        match_winner = row.get("match_winner") if isinstance(row.get("match_winner"), dict) else None

    if not isinstance(match_winner, dict):
        return None

    side_1 = match_winner.get("player_1") or match_winner.get("home") or match_winner.get("first_player")
    side_2 = match_winner.get("player_2") or match_winner.get("away") or match_winner.get("second_player")

    if not isinstance(side_1, dict) or not isinstance(side_2, dict):
        return None

    player_1 = (
        clean(row.get("player_1"))
        or clean(row.get("event_first_player"))
        or clean(row.get("home_player"))
        or clean(side_1.get("name"))
    )

    player_2 = (
        clean(row.get("player_2"))
        or clean(row.get("event_second_player"))
        or clean(row.get("away_player"))
        or clean(side_2.get("name"))
    )

    if not player_1 or not player_2:
        return None

    return {
        **row,
        "event_key": event_key,
        "player_1": player_1,
        "player_2": player_2,
        "first_player_key": row.get("first_player_key") or row.get("player_1_key") or row.get("event_first_player_key"),
        "second_player_key": row.get("second_player_key") or row.get("player_2_key") or row.get("event_second_player_key"),
        "markets": {
            **markets,
            "match_winner": {
                "player_1": side_1,
                "player_2": side_2,
            },
        },
    }


def build_odds_index(payload: Any) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows = extract_matches_from_odds_payload(payload)

    index: dict[str, dict[str, Any]] = {}
    counters = Counter()

    for raw in rows:
        counters["raw_odds_rows"] += 1

        row = normalize_odds_row(raw)
        if row is None:
            counters["skipped_unusable_odds_row"] += 1
            continue

        key = clean(row.get("event_key"))
        if not key:
            counters["skipped_missing_event_key"] += 1
            continue

        index[key] = row
        counters["indexed_odds_rows"] += 1

    return index, dict(sorted(counters.items()))


def names_match(a: Any, b: Any) -> bool:
    na = normalize_name(a)
    nb = normalize_name(b)

    if not na or not nb:
        return False

    if na == nb:
        return True

    a_tokens = na.split()
    b_tokens = nb.split()

    if len(a_tokens) >= 2 and len(b_tokens) >= 2:
        if a_tokens[-1] == b_tokens[-1] and a_tokens[0][0] == b_tokens[0][0]:
            return True

    return False


def actual_winner_side_from_match(match: dict[str, Any], odds_row: dict[str, Any]) -> str | None:
    raw_winner = clean(odds_row.get("event_winner")).lower()

    if raw_winner in {"first player", "first", "home", "player_1", "1"}:
        return "player_1"

    if raw_winner in {"second player", "second", "away", "player_2", "2"}:
        return "player_2"

    winner = match.get("winner") or {}
    winner_name = clean(winner.get("name")) if isinstance(winner, dict) else ""

    p1 = clean(odds_row.get("player_1"))
    p2 = clean(odds_row.get("player_2"))

    if names_match(winner_name, p1):
        return "player_1"

    if names_match(winner_name, p2):
        return "player_2"

    return None


def result_for_pick(selected_side: str, winner_side: str) -> tuple[str, float]:
    if selected_side == winner_side:
        return "WIN", 1.0

    return "LOSS", 0.0


def is_test_match(match: dict[str, Any], args: argparse.Namespace) -> bool:
    if not match.get("ready_for_tle"):
        return False

    if args.test_source == "all":
        return True

    source_fields = [
        match.get("source"),
        match.get("data_source"),
        match.get("provider"),
        match.get("source_name"),
    ]

    source = "|".join(clean(x).lower() for x in source_fields if clean(x))

    if args.test_source.lower() in source:
        return True

    return False


def player_display(match: dict[str, Any], side: str) -> str:
    player = match.get(side) or {}
    if isinstance(player, dict):
        return clean(player.get("name"))
    return ""


def make_backtest_row(
    *,
    match: dict[str, Any],
    odds_row: dict[str, Any],
    selected_side: str,
    actual_winner_side: str,
    probability: float,
    odds: float,
    book_probability: float,
    edge: float,
    ev: float,
    model: str,
    model_details: dict[str, Any],
    odds_source: str,
    odds_values_count: int,
    overround: float,
) -> dict[str, Any]:
    event_key = get_event_key(match) or clean(odds_row.get("event_key"))

    p1_name = clean(odds_row.get("player_1"))
    p2_name = clean(odds_row.get("player_2"))

    selection = p1_name if selected_side == "player_1" else p2_name
    opponent = p2_name if selected_side == "player_1" else p1_name

    result, win_unit = result_for_pick(selected_side, actual_winner_side)
    profit = (odds - 1.0) if result == "WIN" else -1.0

    return {
        "event_key": event_key,
        "date": clean(match.get("date")),
        "tour_level": clean(match.get("tour_level")).lower(),
        "gender": clean(match.get("gender")).lower(),
        "surface": clean((match.get("tournament") or {}).get("surface")).lower() or clean(match.get("surface")).lower(),
        "tournament": clean((match.get("tournament") or {}).get("name")) or clean(match.get("tournament")),
        "match": f"{p1_name} - {p2_name}",
        "selection": selection,
        "opponent": opponent,
        "selected_player_side": selected_side,
        "actual_winner_side": actual_winner_side,
        "actual_winner": p1_name if actual_winner_side == "player_1" else p2_name,
        "odds": round(float(odds), 6),
        "odds_source": odds_source,
        "odds_values_count": odds_values_count,
        "book_probability_devig": round(float(book_probability), 6),
        "tle_probability": round(float(probability), 6),
        "tle_edge": round(float(edge), 6),
        "tle_ev": round(float(ev), 6),
        "overround": round(float(overround), 6),
        "result": result,
        "profit": round(float(profit), 6),
        "stake": 1.0,
        "model": model,
        "model_details": model_details,
    }


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bets = len(rows)
    staked = float(sum(float(row.get("stake") or 1.0) for row in rows))
    profit = float(sum(float(row.get("profit") or 0.0) for row in rows))
    wins = sum(1 for row in rows if row.get("result") == "WIN")
    losses = sum(1 for row in rows if row.get("result") == "LOSS")

    avg_odds = (
        float(sum(float(row["odds"]) for row in rows) / bets)
        if bets
        else None
    )

    avg_tle_probability = (
        float(sum(float(row["tle_probability"]) for row in rows) / bets)
        if bets
        else None
    )

    avg_edge = (
        float(sum(float(row["tle_edge"]) for row in rows) / bets)
        if bets
        else None
    )

    avg_ev = (
        float(sum(float(row["tle_ev"]) for row in rows) / bets)
        if bets
        else None
    )

    return {
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "staked": round(staked, 6),
        "profit": round(profit, 6),
        "roi": round(profit / staked, 6) if staked else None,
        "hit_rate": round(wins / bets, 6) if bets else None,
        "avg_odds": round(avg_odds, 6) if avg_odds is not None else None,
        "avg_tle_probability": round(avg_tle_probability, 6) if avg_tle_probability is not None else None,
        "avg_tle_edge": round(avg_edge, 6) if avg_edge is not None else None,
        "avg_tle_ev": round(avg_ev, 6) if avg_ev is not None else None,
    }


def grouped_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        groups[clean(row.get(key)) or "unknown"].append(row)

    return {
        name: metrics(group_rows)
        for name, group_rows in sorted(groups.items())
    }


def threshold_metrics(rows: list[dict[str, Any]], thresholds: list[float]) -> dict[str, Any]:
    result = {}

    for threshold in thresholds:
        selected = [
            row for row in rows
            if float(row.get("tle_edge") or 0.0) >= threshold
        ]
        result[f"edge>={threshold:.2f}"] = metrics(selected)

    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "date",
        "tour_level",
        "gender",
        "surface",
        "tournament",
        "match",
        "selection",
        "odds",
        "tle_probability",
        "book_probability_devig",
        "tle_edge",
        "tle_ev",
        "overround",
        "result",
        "profit",
        "model",
        "odds_source",
        "odds_values_count",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})

    tmp.replace(path)


def pct(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return ""


def write_md(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    m_all = summary.get("all_priced", {})
    m_value = summary.get("value_bets", {})

    lines = [
        "# TLE API Overlay Odds Backtest",
        "",
        f"Generated: `{summary.get('generated_at')}`",
        f"Odds mode: `{summary.get('settings', {}).get('odds_mode')}`",
        f"Odds filter: `{summary.get('settings', {}).get('min_odds')} - {summary.get('settings', {}).get('max_odds')}`",
        f"Overround filter: `{summary.get('settings', {}).get('min_overround')} - {summary.get('settings', {}).get('max_overround')}`",
        "",
        "## Summary",
        "",
        "| Bucket | Bets | Wins | Losses | ROI | Profit | Hit rate | Avg odds | Avg edge |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| All priced | {m_all.get('bets', 0)} | {m_all.get('wins', 0)} | "
            f"{m_all.get('losses', 0)} | {pct(m_all.get('roi'))} | "
            f"{m_all.get('profit', '')} | {pct(m_all.get('hit_rate'))} | "
            f"{m_all.get('avg_odds', '')} | {pct(m_all.get('avg_tle_edge'))} |"
        ),
        (
            f"| Value bets | {m_value.get('bets', 0)} | {m_value.get('wins', 0)} | "
            f"{m_value.get('losses', 0)} | {pct(m_value.get('roi'))} | "
            f"{m_value.get('profit', '')} | {pct(m_value.get('hit_rate'))} | "
            f"{m_value.get('avg_odds', '')} | {pct(m_value.get('avg_tle_edge'))} |"
        ),
        "",
        "## Value bets",
        "",
        "| # | Date | Level | Gender | Match | Pick | Odds | TLE % | Book % | Edge | EV | Result | Profit |",
        "|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]

    for idx, row in enumerate(rows[:250], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    clean(row.get("date")),
                    clean(row.get("tour_level")),
                    clean(row.get("gender")),
                    clean(row.get("match")).replace("|", "-"),
                    clean(row.get("selection")).replace("|", "-"),
                    clean(row.get("odds")),
                    pct(row.get("tle_probability")),
                    pct(row.get("book_probability_devig")),
                    pct(row.get("tle_edge")),
                    pct(row.get("tle_ev")),
                    clean(row.get("result")),
                    clean(row.get("profit")),
                ]
            )
            + " |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TLE walk-forward backtest on API overlay matches with cleaned odds."
    )

    parser.add_argument("--canonical-manifest", default=str(DEFAULT_CANONICAL_MANIFEST))
    parser.add_argument("--api-player-mapping", default=str(DEFAULT_API_PLAYER_MAPPING))
    parser.add_argument("--odds-url", default=DEFAULT_ODDS_URL)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--test-source", default="api_tennis")

    parser.add_argument(
        "--odds-mode",
        choices=["clean_average", "clean_median", "bookmaker", "best", "worst"],
        default="clean_average",
    )
    parser.add_argument("--bookmaker", default="Pncl")
    parser.add_argument("--min-odds", type=float, default=1.20)
    parser.add_argument("--max-odds", type=float, default=8.00)
    parser.add_argument("--min-overround", type=float, default=-0.03)
    parser.add_argument("--max-overround", type=float, default=0.18)

    parser.add_argument("--min-edge", type=float, default=0.03)
    parser.add_argument("--min-ev", type=float, default=0.0)

    parser.add_argument("--main-min-level-matches", type=int, default=20)
    parser.add_argument("--main-min-surface-matches", type=int, default=10)
    parser.add_argument("--itf-min-level-matches", type=int, default=5)
    parser.add_argument("--challenger-min-level-matches", type=int, default=5)

    args = parser.parse_args()

    canonical_manifest = Path(args.canonical_manifest)
    if not canonical_manifest.is_absolute():
        canonical_manifest = ROOT_DIR / canonical_manifest

    api_mapping_path = Path(args.api_player_mapping)
    if not api_mapping_path.is_absolute():
        api_mapping_path = ROOT_DIR / api_mapping_path

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT_DIR / out_dir

    generated_at = now_iso()

    odds_payload = load_json(args.odds_url)
    odds_index, odds_index_summary = build_odds_index(odds_payload)

    api_mapping = load_api_player_mapping(api_mapping_path)
    exact_index, surname_initial_index, _display = build_alias_indexes(canonical_manifest)

    players: dict[str, dict[str, Any]] = {}
    counters = Counter()

    all_priced: list[dict[str, Any]] = []
    value_bets: list[dict[str, Any]] = []

    for match in iter_canonical_matches(canonical_manifest):
        counters["canonical_rows_seen"] += 1

        event_key = get_event_key(match)
        match_date = parse_date(match.get("date"))

        should_test = (
            match_date is not None
            and event_key
            and event_key in odds_index
            and is_test_match(match, args)
        )

        if should_test:
            counters["test_candidates_with_odds"] += 1

            odds_row = odds_index[event_key]

            gender = clean(match.get("gender")).lower()
            level = clean(match.get("tour_level")).lower()
            surface = clean((match.get("tournament") or {}).get("surface")).lower() or clean(match.get("surface")).lower()

            if gender not in {"men", "women"}:
                counters["skipped_gender_unknown"] += 1
            else:
                markets = odds_row.get("markets")
                winner_market = markets.get("match_winner") if isinstance(markets, dict) else None

                if not isinstance(winner_market, dict):
                    counters["skipped_missing_match_winner"] += 1
                else:
                    side_1 = winner_market.get("player_1")
                    side_2 = winner_market.get("player_2")

                    if not isinstance(side_1, dict) or not isinstance(side_2, dict):
                        counters["skipped_missing_match_winner_sides"] += 1
                    else:
                        p1_odds, p1_odds_source, p1_values_count, p1_values = choose_side_odds(side_1, args)
                        p2_odds, p2_odds_source, p2_values_count, p2_values = choose_side_odds(side_2, args)

                        if p1_odds is None or p2_odds is None:
                            counters["skipped_missing_clean_odds_pair"] += 1
                        else:
                            book_p1, book_p2, overround = devig_pair(p1_odds, p2_odds)

                            if overround < args.min_overround or overround > args.max_overround:
                                counters["skipped_overround_filter"] += 1
                            else:
                                p1_name = clean(odds_row.get("player_1"))
                                p2_name = clean(odds_row.get("player_2"))

                                p1_api_key = safe_int(odds_row.get("first_player_key"))
                                p2_api_key = safe_int(odds_row.get("second_player_key"))

                                p1_key, p1_method = resolve_player(
                                    p1_api_key,
                                    p1_name,
                                    gender,
                                    api_mapping,
                                    exact_index,
                                    surname_initial_index,
                                )
                                p2_key, p2_method = resolve_player(
                                    p2_api_key,
                                    p2_name,
                                    gender,
                                    api_mapping,
                                    exact_index,
                                    surname_initial_index,
                                )

                                counters[f"p1_resolve_{p1_method}"] += 1
                                counters[f"p2_resolve_{p2_method}"] += 1

                                if not p1_key or not p2_key:
                                    counters["skipped_unresolved_player"] += 1
                                else:
                                    p1_player = players.get(p1_key)
                                    p2_player = players.get(p2_key)

                                    if not p1_player or not p2_player:
                                        counters["skipped_missing_player_history"] += 1
                                    else:
                                        p1_probability, model, model_details = probability_for_player_1(
                                            p1_player,
                                            p2_player,
                                            level,
                                            surface,
                                            args,
                                        )

                                        if p1_probability is None:
                                            counters[f"skipped_{model}"] += 1
                                        else:
                                            winner_side = actual_winner_side_from_match(match, odds_row)

                                            if winner_side not in {"player_1", "player_2"}:
                                                counters["skipped_unresolved_actual_winner_side"] += 1
                                            else:
                                                p2_probability = 1.0 - p1_probability

                                                odds_source = (
                                                    p1_odds_source
                                                    if p1_odds_source == p2_odds_source
                                                    else f"{p1_odds_source}|{p2_odds_source}"
                                                )

                                                model_details = {
                                                    **model_details,
                                                    "p1_resolve_method": p1_method,
                                                    "p2_resolve_method": p2_method,
                                                    "p1_odds_values": [round(x, 6) for x in p1_values],
                                                    "p2_odds_values": [round(x, 6) for x in p2_values],
                                                    "p1_clean_odds": round(p1_odds, 6),
                                                    "p2_clean_odds": round(p2_odds, 6),
                                                    "overround": round(overround, 6),
                                                }

                                                for side, probability, odds, book_probability, values_count in [
                                                    ("player_1", p1_probability, p1_odds, book_p1, p1_values_count),
                                                    ("player_2", p2_probability, p2_odds, book_p2, p2_values_count),
                                                ]:
                                                    edge = probability - book_probability
                                                    ev = probability * odds - 1.0

                                                    row = make_backtest_row(
                                                        match=match,
                                                        odds_row=odds_row,
                                                        selected_side=side,
                                                        actual_winner_side=winner_side,
                                                        probability=probability,
                                                        odds=odds,
                                                        book_probability=book_probability,
                                                        edge=edge,
                                                        ev=ev,
                                                        model=model,
                                                        model_details=model_details,
                                                        odds_source=odds_source,
                                                        odds_values_count=values_count,
                                                        overround=overround,
                                                    )

                                                    all_priced.append(row)

                                                    if edge >= args.min_edge and ev >= args.min_ev:
                                                        value_bets.append(row)
                                                        counters["value_bets"] += 1
                                                        counters[f"value_level_{level}"] += 1
                                                        counters[f"value_gender_{gender}"] += 1

                                                counters["priced_matches"] += 1
                                                counters[f"priced_level_{level}"] += 1
                                                counters[f"priced_gender_{gender}"] += 1

        if update_state_for_match(match, players):
            counters["state_matches_processed"] += 1

    all_priced.sort(
        key=lambda row: (
            row.get("date") or "",
            -(float(row.get("tle_edge") or 0.0)),
            -(float(row.get("tle_ev") or 0.0)),
        )
    )

    value_bets.sort(
        key=lambda row: (
            -(float(row.get("tle_edge") or 0.0)),
            -(float(row.get("tle_ev") or 0.0)),
            row.get("date") or "",
        )
    )

    thresholds = [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]

    summary = {
        "generated_at": generated_at,
        "odds_url": args.odds_url,
        "odds_index": odds_index_summary,
        "settings": {
            "test_source": args.test_source,
            "odds_mode": args.odds_mode,
            "bookmaker": args.bookmaker,
            "min_odds": args.min_odds,
            "max_odds": args.max_odds,
            "min_overround": args.min_overround,
            "max_overround": args.max_overround,
            "min_edge": args.min_edge,
            "min_ev": args.min_ev,
            "main_min_level_matches": args.main_min_level_matches,
            "main_min_surface_matches": args.main_min_surface_matches,
            "itf_min_level_matches": args.itf_min_level_matches,
            "challenger_min_level_matches": args.challenger_min_level_matches,
            "model_note": "CORE: main tour 80% level + 20% surface; ITF 100% ITF level; Challenger 100% Challenger level; Qualifying no bet.",
            "odds_note": "Clean odds filter removes individual bookmaker odds outside min_odds/max_odds before averaging.",
        },
        "counters": dict(sorted(counters.items())),
        "all_priced": metrics(all_priced),
        "value_bets": metrics(value_bets),
        "all_priced_by_level": grouped_metrics(all_priced, "tour_level"),
        "value_bets_by_level": grouped_metrics(value_bets, "tour_level"),
        "all_priced_by_gender": grouped_metrics(all_priced, "gender"),
        "value_bets_by_gender": grouped_metrics(value_bets, "gender"),
        "thresholds_on_all_priced": threshold_metrics(all_priced, thresholds),
        "top_value_bets": value_bets[:50],
    }

    payload = {
        "schema_version": 4,
        "summary": summary,
        "value_bets": value_bets,
        "all_priced": all_priced,
    }

    json_path = out_dir / "tle_backtest_api_overlay_with_odds_clean.json"
    latest_json_path = out_dir / "tle_backtest_api_overlay_with_odds_latest.json"
    csv_path = out_dir / "tle_backtest_api_overlay_with_odds_clean_value_bets.csv"
    latest_csv_path = out_dir / "tle_backtest_api_overlay_with_odds_latest_value_bets.csv"
    md_path = out_dir / "tle_backtest_api_overlay_with_odds_clean.md"
    latest_md_path = out_dir / "tle_backtest_api_overlay_with_odds_latest.md"

    save_json(json_path, payload)
    save_json(latest_json_path, payload)

    write_csv(csv_path, value_bets)
    write_csv(latest_csv_path, value_bets)

    write_md(md_path, summary, value_bets)
    write_md(latest_md_path, summary, value_bets)

    print("TLE API OVERLAY WITH CLEAN ODDS BACKTEST DONE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"JSON: {json_path}")
    print(f"Latest JSON: {latest_json_path}")
    print(f"CSV: {csv_path}")
    print(f"Latest CSV: {latest_csv_path}")
    print(f"MD: {md_path}")
    print(f"Latest MD: {latest_md_path}")


if __name__ == "__main__":
    main()
