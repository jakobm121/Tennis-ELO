from __future__ import annotations

import argparse
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


DEFAULT_AI_ODDS_URL = (
    "https://raw.githubusercontent.com/jakobm121/Ai/refs/heads/main/"
    "data/tennis_odds_today.json"
)

DEFAULT_CANONICAL_MANIFEST = (
    ROOT_DIR / "data" / "tle" / "processed" / "canonical" / "tle_matches_manifest.json"
)

DEFAULT_API_PLAYER_MAPPING = (
    ROOT_DIR / "data" / "tle" / "mappings" / "api_player_to_sackmann.json"
)

DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "tle" / "predictions"

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
    if not math.isfinite(number) or number <= 1.0:
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
        with urllib.request.urlopen(text, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))

    path = Path(text)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_canonical_matches(manifest_path: Path):
    manifest = load_json(manifest_path)
    rows = []

    for item in manifest.get("year_files") or []:
        rel = item.get("path")
        if not rel:
            continue
        path = Path(rel)
        if not path.is_absolute():
            path = ROOT_DIR / path
        if not path.exists():
            continue
        for row in read_jsonl_gz(path):
            rows.append(row)

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
    return {str(key): value for key, value in players.items() if isinstance(value, dict)}


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
        "global": {"overall_elo": DEFAULT_ELO, "matches": 0, "wins": 0, "surfaces": {}},
        "levels": {},
    }


def ensure_player(players: dict[str, dict[str, Any]], key: str, name: str, gender: str) -> dict[str, Any]:
    if key not in players:
        players[key] = new_player_state(key, name, gender)
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


def build_state_before_date(manifest_path: Path, scan_date: date) -> tuple[dict[str, dict[str, Any]], int]:
    players: dict[str, dict[str, Any]] = {}
    processed = 0

    for match in iter_canonical_matches(manifest_path):
        match_date = parse_date(match.get("date"))
        if match_date is None:
            continue
        if match_date >= scan_date:
            break
        update_state_for_match(match, players)
        processed += 1

    return players, processed


def get_level_state(player: dict[str, Any], level: str) -> dict[str, Any] | None:
    return player.get("levels", {}).get(level)


def get_level_surface_state(player: dict[str, Any], level: str, surface: str) -> dict[str, Any] | None:
    level_state = get_level_state(player, level)
    if not level_state:
        return None
    return level_state.get("surfaces", {}).get(surface)


def probability_for_home(
    home: dict[str, Any],
    away: dict[str, Any],
    level: str,
    surface: str,
    args: argparse.Namespace,
) -> tuple[float | None, str, dict[str, Any]]:
    if level in MAIN_TOUR_LEVELS:
        model_level = "atp_wta"
        home_level = get_level_state(home, model_level)
        away_level = get_level_state(away, model_level)

        if not home_level or not away_level:
            return None, "main_tour_missing_level_rating", {}

        if home_level["matches"] < args.main_min_level_matches or away_level["matches"] < args.main_min_level_matches:
            return None, "main_tour_level_min_sample", {}

        if surface not in VALID_SURFACES:
            return None, "main_tour_unknown_surface", {}

        home_surface = get_level_surface_state(home, model_level, surface)
        away_surface = get_level_surface_state(away, model_level, surface)

        if not home_surface or not away_surface:
            return None, "main_tour_missing_surface_rating", {}

        if home_surface["matches"] < args.main_min_surface_matches or away_surface["matches"] < args.main_min_surface_matches:
            return None, "main_tour_surface_min_sample", {}

        p_level = expected_score(float(home_level["overall_elo"]), float(away_level["overall_elo"]))
        p_surface = expected_score(float(home_surface["elo"]), float(away_surface["elo"]))
        p = 0.80 * p_level + 0.20 * p_surface

        return p, "main_tour_80_level_20_surface", {
            "home_level_matches": home_level["matches"],
            "away_level_matches": away_level["matches"],
            "home_surface_matches": home_surface["matches"],
            "away_surface_matches": away_surface["matches"],
            "home_level_elo": round(float(home_level["overall_elo"]), 3),
            "away_level_elo": round(float(away_level["overall_elo"]), 3),
            "home_surface_elo": round(float(home_surface["elo"]), 3),
            "away_surface_elo": round(float(away_surface["elo"]), 3),
            "p_level": round(p_level, 6),
            "p_surface": round(p_surface, 6),
        }

    if level == "itf":
        home_level = get_level_state(home, "itf")
        away_level = get_level_state(away, "itf")

        if not home_level or not away_level:
            return None, "itf_missing_level_rating", {}

        if home_level["matches"] < args.itf_min_level_matches or away_level["matches"] < args.itf_min_level_matches:
            return None, "itf_level_min_sample", {}

        p = expected_score(float(home_level["overall_elo"]), float(away_level["overall_elo"]))
        return p, "itf_100_level_overall", {
            "home_level_matches": home_level["matches"],
            "away_level_matches": away_level["matches"],
            "home_level_elo": round(float(home_level["overall_elo"]), 3),
            "away_level_elo": round(float(away_level["overall_elo"]), 3),
        }

    if level == "challenger":
        home_level = get_level_state(home, "challenger")
        away_level = get_level_state(away, "challenger")

        if not home_level or not away_level:
            return None, "challenger_missing_level_rating", {}

        if home_level["matches"] < args.challenger_min_level_matches or away_level["matches"] < args.challenger_min_level_matches:
            return None, "challenger_level_min_sample", {}

        p = expected_score(float(home_level["overall_elo"]), float(away_level["overall_elo"]))
        return p, "challenger_100_level_overall", {
            "home_level_matches": home_level["matches"],
            "away_level_matches": away_level["matches"],
            "home_level_elo": round(float(home_level["overall_elo"]), 3),
            "away_level_elo": round(float(away_level["overall_elo"]), 3),
        }

    if level == "qualifying":
        return None, "qualifying_no_bet", {}

    return None, "unsupported_level", {}


def choose_odds(side_payload: dict[str, Any], bookmaker: str, fallback: str) -> tuple[float | None, str | None]:
    books = side_payload.get("books")
    if isinstance(books, dict):
        value = safe_float(books.get(bookmaker))
        if value is not None:
            return value, bookmaker

    if fallback == "none":
        return None, None

    if fallback == "best":
        value = safe_float(side_payload.get("best_odds"))
        source = clean(side_payload.get("best_bookmaker")) or "best"
        return value, source if value is not None else None

    if fallback == "median":
        value = safe_float(side_payload.get("median_odds"))
        return value, "median" if value is not None else None

    values = []
    if isinstance(books, dict):
        values = [v for v in (safe_float(x) for x in books.values()) if v is not None]

    if not values:
        return None, None

    if fallback == "max":
        return max(values), "fallback_max"
    if fallback == "min":
        return min(values), "fallback_min"

    return float(statistics.median(values)), "fallback_median"


def devig_pair(home_odds: float, away_odds: float) -> tuple[float, float, float]:
    raw_home = 1.0 / home_odds
    raw_away = 1.0 / away_odds
    total = raw_home + raw_away
    return raw_home / total, raw_away / total, total - 1.0


def make_pick(
    row: dict[str, Any],
    side: str,
    probability: float,
    odds: float,
    book_probability: float,
    edge: float,
    ev: float,
    odds_source: str,
    model: str,
    model_details: dict[str, Any],
    home_key: str,
    away_key: str,
) -> dict[str, Any]:
    home_name = clean(row.get("player_1"))
    away_name = clean(row.get("player_2"))

    if side == "player_1":
        selected_side = "Home"
        selection = home_name
        opponent = away_name
    else:
        selected_side = "Away"
        selection = away_name
        opponent = home_name

    return {
        "event_key": clean(row.get("event_key") or row.get("event_id")),
        "date": clean(row.get("date")),
        "time": clean(row.get("time")),
        "timezone": clean(row.get("timezone")),
        "tournament": clean(row.get("tournament")),
        "tournament_key": clean(row.get("tournament_key")),
        "round": clean(row.get("round")),
        "event_type": clean(row.get("event_type")),
        "gender": clean(row.get("gender")).lower(),
        "tour_level": clean(row.get("tour_level")).lower(),
        "surface": clean(row.get("surface")).lower(),
        "match": f"{home_name} - {away_name}",
        "selection": selection,
        "opponent": opponent,
        "selected_side": selected_side,
        "selected_player_side": side,
        "home_player": home_name,
        "away_player": away_name,
        "home_tle_key": home_key,
        "away_tle_key": away_key,
        "odds": round(odds, 6),
        "odds_source": odds_source,
        "book_probability_devig": round(book_probability, 6),
        "tle_probability": round(probability, 6),
        "tle_edge": round(edge, 6),
        "tle_ev": round(ev, 6),
        "model": model,
        "model_details": model_details,
    }



def format_pct(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return ""


def write_picks_tables(
    out_dir: Path,
    stem: str,
    latest_stem: str,
    summary: dict[str, Any],
    picks: list[dict[str, Any]],
) -> tuple[Path, Path, Path, Path]:
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"
    latest_csv_path = out_dir / f"{latest_stem}.csv"
    latest_md_path = out_dir / f"{latest_stem}.md"

    fields = [
        "rank",
        "date",
        "time",
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
        "model",
        "odds_source",
    ]

    rows = []
    for rank, pick in enumerate(picks, start=1):
        rows.append(
            {
                "rank": rank,
                "date": pick.get("date"),
                "time": pick.get("time"),
                "tour_level": pick.get("tour_level"),
                "gender": pick.get("gender"),
                "surface": pick.get("surface"),
                "tournament": pick.get("tournament"),
                "match": pick.get("match"),
                "selection": pick.get("selection"),
                "odds": pick.get("odds"),
                "tle_probability": pick.get("tle_probability"),
                "book_probability_devig": pick.get("book_probability_devig"),
                "tle_edge": pick.get("tle_edge"),
                "tle_ev": pick.get("tle_ev"),
                "model": pick.get("model"),
                "odds_source": pick.get("odds_source"),
            }
        )

    import csv

    for path in [csv_path, latest_csv_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)

    md_lines = [
        "# TLE Scanner Picks",
        "",
        f"Generated: `{summary.get('generated_at')}`",
        f"AI odds generated: `{summary.get('ai_generated_at')}`",
        f"Scan date: `{summary.get('scan_date_start')}`",
        f"Picks: `{len(picks)}`",
        "",
        "| # | Time | Level | Gender | Match | Pick | Odds | TLE % | Book % | Edge | EV | Model |",
        "|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]

    for rank, pick in enumerate(picks, start=1):
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    clean(pick.get("time")),
                    clean(pick.get("tour_level")),
                    clean(pick.get("gender")),
                    clean(pick.get("match")).replace("|", "-"),
                    clean(pick.get("selection")).replace("|", "-"),
                    clean(pick.get("odds")),
                    format_pct(pick.get("tle_probability")),
                    format_pct(pick.get("book_probability_devig")),
                    format_pct(pick.get("tle_edge")),
                    format_pct(pick.get("tle_ev")),
                    clean(pick.get("model")),
                ]
            )
            + " |"
        )

    md_text = "\n".join(md_lines) + "\n"

    for path in [md_path, latest_md_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(md_text, encoding="utf-8")
        tmp.replace(path)

    return csv_path, md_path, latest_csv_path, latest_md_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pure TLE scanner from AI repo tennis_odds_today.json."
    )
    parser.add_argument("--ai-odds-url", default=DEFAULT_AI_ODDS_URL)
    parser.add_argument("--bookmaker", default="Pncl")
    parser.add_argument("--fallback", choices=["none", "median", "best", "max", "min"], default="median")
    parser.add_argument("--levels", default="challenger,itf,atp_wta,grand_slam")
    parser.add_argument("--min-edge", type=float, default=0.03)
    parser.add_argument("--min-ev", type=float, default=0.0)
    parser.add_argument("--min-odds", type=float, default=1.01)
    parser.add_argument("--max-odds", type=float, default=100.0)
    parser.add_argument("--canonical-manifest", default=str(DEFAULT_CANONICAL_MANIFEST))
    parser.add_argument("--api-player-mapping", default=str(DEFAULT_API_PLAYER_MAPPING))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    parser.add_argument("--main-min-level-matches", type=int, default=20)
    parser.add_argument("--main-min-surface-matches", type=int, default=10)
    parser.add_argument("--itf-min-level-matches", type=int, default=5)
    parser.add_argument("--challenger-min-level-matches", type=int, default=5)

    args = parser.parse_args()

    levels_to_scan = {clean(x).lower() for x in args.levels.split(",") if clean(x)}
    ai_payload = load_json(args.ai_odds_url)

    matches = ai_payload.get("matches")
    if not isinstance(matches, list):
        raise RuntimeError("AI odds JSON does not contain list field 'matches'.")

    dates = sorted({clean(row.get("date")) for row in matches if isinstance(row, dict) and clean(row.get("date"))})
    if not dates:
        raise RuntimeError("No dates found in AI odds JSON matches.")

    scan_start = date.fromisoformat(dates[0])
    scan_end = date.fromisoformat(dates[-1])

    canonical_manifest = Path(args.canonical_manifest)
    if not canonical_manifest.is_absolute():
        canonical_manifest = ROOT_DIR / canonical_manifest

    api_mapping_path = Path(args.api_player_mapping)
    if not api_mapping_path.is_absolute():
        api_mapping_path = ROOT_DIR / api_mapping_path

    api_mapping = load_api_player_mapping(api_mapping_path)
    exact_index, surname_initial_index, _display = build_alias_indexes(canonical_manifest)
    players, historical_matches = build_state_before_date(canonical_manifest, scan_start)

    counters = Counter()
    all_scored = []
    picks = []

    for row in matches:
        if not isinstance(row, dict):
            continue

        counters["ai_matches_seen"] += 1

        if row.get("live"):
            counters["skipped_live"] += 1
            continue

        status = clean(row.get("status")).lower()
        if status in {"finished", "cancelled", "canceled", "postponed", "retired", "walkover"}:
            counters[f"skipped_status_{status}"] += 1
            continue

        gender = clean(row.get("gender")).lower()
        if gender not in {"men", "women"}:
            counters["skipped_gender_unknown"] += 1
            continue

        level = clean(row.get("tour_level")).lower()
        if level not in levels_to_scan:
            counters[f"skipped_level_{level or 'missing'}"] += 1
            continue

        surface = clean(row.get("surface")).lower()

        markets = row.get("markets")
        if not isinstance(markets, dict):
            counters["skipped_missing_markets"] += 1
            continue

        winner_market = markets.get("match_winner")
        if not isinstance(winner_market, dict):
            counters["skipped_missing_match_winner"] += 1
            continue

        side_1 = winner_market.get("player_1")
        side_2 = winner_market.get("player_2")
        if not isinstance(side_1, dict) or not isinstance(side_2, dict):
            counters["skipped_missing_match_winner_sides"] += 1
            continue

        home_odds, home_source = choose_odds(side_1, args.bookmaker, args.fallback)
        away_odds, away_source = choose_odds(side_2, args.bookmaker, args.fallback)

        if home_odds is None or away_odds is None:
            counters["skipped_missing_pair_odds"] += 1
            continue

        odds_source = args.bookmaker if home_source == args.bookmaker and away_source == args.bookmaker else f"{home_source}|{away_source}"

        home_name = clean(row.get("player_1"))
        away_name = clean(row.get("player_2"))
        home_api_key = safe_int(row.get("first_player_key"))
        away_api_key = safe_int(row.get("second_player_key"))

        home_key, home_method = resolve_player(home_api_key, home_name, gender, api_mapping, exact_index, surname_initial_index)
        away_key, away_method = resolve_player(away_api_key, away_name, gender, api_mapping, exact_index, surname_initial_index)

        counters[f"home_resolve_{home_method}"] += 1
        counters[f"away_resolve_{away_method}"] += 1

        if not home_key or not away_key:
            counters["skipped_unresolved_player"] += 1
            continue

        home_player = players.get(home_key)
        away_player = players.get(away_key)

        if not home_player or not away_player:
            counters["skipped_missing_player_history"] += 1
            continue

        home_prob, model, details = probability_for_home(home_player, away_player, level, surface, args)

        if home_prob is None:
            counters[f"skipped_{model}"] += 1
            continue

        away_prob = 1.0 - home_prob
        book_home_prob, book_away_prob, overround = devig_pair(home_odds, away_odds)

        details = {
            **details,
            "home_resolve_method": home_method,
            "away_resolve_method": away_method,
            "home_odds": round(home_odds, 6),
            "away_odds": round(away_odds, 6),
            "overround": round(overround, 6),
        }

        for side, probability, odds, book_probability in [
            ("player_1", home_prob, home_odds, book_home_prob),
            ("player_2", away_prob, away_odds, book_away_prob),
        ]:
            ev = probability * odds - 1.0
            edge = probability - book_probability

            scored = make_pick(
                row=row,
                side=side,
                probability=probability,
                odds=odds,
                book_probability=book_probability,
                edge=edge,
                ev=ev,
                odds_source=odds_source,
                model=model,
                model_details=details,
                home_key=home_key,
                away_key=away_key,
            )
            all_scored.append(scored)

            if (
                edge >= args.min_edge
                and ev >= args.min_ev
                and odds >= args.min_odds
                and odds <= args.max_odds
            ):
                picks.append(scored)
                counters["picks"] += 1
                counters[f"picks_level_{level}"] += 1
                counters[f"picks_gender_{gender}"] += 1

        counters["scored_matches"] += 1
        counters[f"scored_level_{level}"] += 1

    picks.sort(key=lambda x: (-x["tle_edge"], -x["tle_ev"], x["date"], x["time"], x["event_key"]))
    all_scored.sort(key=lambda x: (-x["tle_edge"], -x["tle_ev"], x["date"], x["time"], x["event_key"]))

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT_DIR / out_dir

    if scan_start == scan_end:
        stem = f"tle_scanner_from_ai_odds_{scan_start.isoformat()}"
    else:
        stem = f"tle_scanner_from_ai_odds_{scan_start.isoformat()}_{scan_end.isoformat()}"

    output_path = out_dir / f"{stem}.json"
    latest_path = out_dir / "tle_scanner_picks_latest.json"
    table_stem = stem.replace("tle_scanner_from_ai_odds", "tle_scanner_table")
    latest_table_stem = "tle_scanner_table_latest"

    summary = {
        "generated_at": now_iso(),
        "ai_odds_url": args.ai_odds_url,
        "ai_generated_at": ai_payload.get("generated_at"),
        "ai_summary": ai_payload.get("summary"),
        "scan_date_start": scan_start.isoformat(),
        "scan_date_end": scan_end.isoformat(),
        "settings": {
            "bookmaker": args.bookmaker,
            "fallback": args.fallback,
            "min_edge": args.min_edge,
            "min_ev": args.min_ev,
            "min_odds": args.min_odds,
            "max_odds": args.max_odds,
            "levels": sorted(levels_to_scan),
            "main_min_level_matches": args.main_min_level_matches,
            "main_min_surface_matches": args.main_min_surface_matches,
            "itf_min_level_matches": args.itf_min_level_matches,
            "challenger_min_level_matches": args.challenger_min_level_matches,
            "state_note": "TLE state uses canonical matches strictly before first AI odds date.",
            "historical_matches_loaded_before_scan_date": historical_matches,
        },
        "counters": dict(sorted(counters.items())),
        "picks_count": len(picks),
        "scored_sides_count": len(all_scored),
        "level_counts": dict(Counter(row["tour_level"] for row in picks)),
        "gender_counts": dict(Counter(row["gender"] for row in picks)),
        "top_edges": [
            {
                "event_key": row["event_key"],
                "date": row["date"],
                "time": row["time"],
                "match": row["match"],
                "selection": row["selection"],
                "tour_level": row["tour_level"],
                "odds": row["odds"],
                "tle_probability": row["tle_probability"],
                "book_probability_devig": row["book_probability_devig"],
                "tle_edge": row["tle_edge"],
                "tle_ev": row["tle_ev"],
                "model": row["model"],
            }
            for row in picks[:25]
        ],
    }

    payload = {
        "schema_version": 1,
        "summary": summary,
        "picks": picks,
        "all_scored_sides": all_scored,
    }

    save_json(output_path, payload)
    save_json(latest_path, payload)

    csv_path, md_path, latest_csv_path, latest_md_path = write_picks_tables(
        out_dir=out_dir,
        stem=table_stem,
        latest_stem=latest_table_stem,
        summary=summary,
        picks=picks,
    )

    print("TLE SCANNER FROM AI ODDS DONE")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Output: {output_path}")
    print(f"Latest: {latest_path}")
    print(f"CSV table: {csv_path}")
    print(f"MD table: {md_path}")
    print(f"Latest CSV table: {latest_csv_path}")
    print(f"Latest MD table: {latest_md_path}")


if __name__ == "__main__":
    main()
