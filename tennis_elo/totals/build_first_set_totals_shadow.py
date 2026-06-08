import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from tennis_elo.config import ROOT_DIR
from tennis_elo.utils import clean_str, load_json, now_iso, save_json


ODDS_URL = os.getenv(
    "TENNIS_ODDS_URL",
    "https://raw.githubusercontent.com/jakobm121/Ai/refs/heads/main/data/tennis_odds_today.json",
)
PROFILES_FILE = ROOT_DIR / "data" / "totals" / "player_first_set_profiles.json"
OUTPUT_FILE = ROOT_DIR / "data" / "predictions" / "first_set_totals_shadow.json"
REPORT_FILE = ROOT_DIR / "data" / "reports" / "first_set_totals_shadow_report.json"

MIN_PLAYER_SAMPLE = int(os.getenv("FIRST_SET_MULTI_MIN_PLAYER_SAMPLE", "10"))
MIN_ONE_PLAYER_SAMPLE = int(os.getenv("FIRST_SET_MULTI_MIN_ONE_PLAYER_SAMPLE", "20"))
MIN_EDGE = float(os.getenv("FIRST_SET_MULTI_MIN_EDGE", "0.05"))
HTTP_TIMEOUT = int(os.getenv("TENNIS_ODDS_HTTP_TIMEOUT", "45"))

STRATEGIES = [
    {
        "strategy_id": "all_fav75_under_9_5",
        "side": "under",
        "line": 9.5,
        "min_odds": 1.80,
        "min_favorite_probability": 0.75,
        "surface": None,
        "gender": None,
        "base_probability": 0.5853,
        "confidence": "medium",
    },
    {
        "strategy_id": "clay_fav75_under_10_5",
        "side": "under",
        "line": 10.5,
        "min_odds": 1.30,
        "min_favorite_probability": 0.75,
        "surface": "clay",
        "gender": None,
        "base_probability": 0.8185,
        "confidence": "high",
    },
    {
        "strategy_id": "clay_fav85_under_9_5",
        "side": "under",
        "line": 9.5,
        "min_odds": 1.65,
        "min_favorite_probability": 0.85,
        "surface": "clay",
        "gender": None,
        "base_probability": 0.6555,
        "confidence": "high",
    },
    {
        "strategy_id": "men_over_8_5",
        "side": "over",
        "line": 8.5,
        "min_odds": 1.42,
        "min_favorite_probability": None,
        "surface": None,
        "gender": "men",
        "base_probability": 0.7544,
        "confidence": "medium",
    },
    {
        "strategy_id": "clay_fav85_under_8_5",
        "side": "under",
        "line": 8.5,
        "min_odds": 2.55,
        "min_favorite_probability": 0.85,
        "surface": "clay",
        "gender": None,
        "base_probability": 0.4202,
        "confidence": "medium",
    },
    {
        "strategy_id": "men_hard_over_10_5_watch",
        "side": "over",
        "line": 10.5,
        "min_odds": 3.20,
        "min_favorite_probability": None,
        "surface": "hard",
        "gender": "men",
        "base_probability": 0.3316,
        "confidence": "watch",
    },
]


def fetch_json(url: str) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "User-Agent": "Tennis-ELO first-set multi-shadow/1.0",
            "Accept": "application/json",
        },
    )

    with urlopen(request, timeout=HTTP_TIMEOUT) as response:
        payload = json.load(response)

    return payload if isinstance(payload, dict) else {}


def compact(value: Any) -> str:
    text = clean_str(value).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def tokens(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", clean_str(value).lower())


def split_name(value: Any) -> tuple[list[str], list[str]]:
    parts = tokens(value)

    if not parts:
        return [], []

    if len(parts) == 1:
        return [], parts

    first_is_initial = len(parts[0]) == 1
    last_is_initial = len(parts[-1]) == 1

    if first_is_initial and not last_is_initial:
        return [parts[0]], parts[1:]

    if last_is_initial and not first_is_initial:
        return [parts[-1]], parts[:-1]

    return [parts[0]], parts[1:]


def name_aliases(value: Any) -> set[str]:
    parts = tokens(value)
    given, surname = split_name(value)
    aliases: set[str] = set()

    if not parts:
        return aliases

    aliases.add("".join(parts))
    aliases.add(" ".join(parts))

    surname_joined = "".join(surname)
    given_joined = "".join(given)
    initial = given[0][:1] if given else ""

    if surname_joined:
        aliases.add(surname_joined)

        if given_joined:
            aliases.add(f"{given_joined}{surname_joined}")
            aliases.add(f"{surname_joined}{given_joined}")

        if initial:
            aliases.add(f"{initial}{surname_joined}")
            aliases.add(f"{surname_joined}{initial}")

    return {compact(alias) for alias in aliases if alias}


def build_profile_index(profiles: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}

    for profile in profiles:
        for alias in name_aliases(profile.get("player_name")):
            index.setdefault(alias, []).append(profile)

    return index


def profile_match_score(query_name: str, profile_name: str) -> int:
    q_given, q_surname = split_name(query_name)
    p_given, p_surname = split_name(profile_name)

    q_surname_joined = "".join(q_surname)
    p_surname_joined = "".join(p_surname)

    score = 0

    if compact(query_name) == compact(profile_name):
        score += 50

    if q_surname_joined and q_surname_joined == p_surname_joined:
        score += 30

    q_initial = q_given[0][:1] if q_given else ""
    p_initial = p_given[0][:1] if p_given else ""

    if q_initial and q_initial == p_initial:
        score += 8

    return score


def find_profile(
    player_name: str,
    index: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any] | None, str]:
    candidates: dict[str, dict[str, Any]] = {}

    for alias in name_aliases(player_name):
        for candidate in index.get(alias, []):
            candidates[clean_str(candidate.get("player_name"))] = candidate

    if not candidates:
        return None, "not_found"

    ranked = sorted(
        candidates.values(),
        key=lambda profile: (
            profile_match_score(
                player_name,
                clean_str(profile.get("player_name")),
            ),
            int(profile.get("sample_size") or 0),
        ),
        reverse=True,
    )

    if len(ranked) > 1:
        first_score = profile_match_score(
            player_name,
            clean_str(ranked[0].get("player_name")),
        )
        second_score = profile_match_score(
            player_name,
            clean_str(ranked[1].get("player_name")),
        )

        if first_score == second_score:
            return None, "ambiguous"

    return ranked[0], "matched"


def extract_match_winner(match: dict[str, Any]) -> dict[str, Any] | None:
    market = (match.get("markets") or {}).get("match_winner")

    if not isinstance(market, dict):
        return None

    de_vig = market.get("de_vig") or {}

    try:
        p1 = float(de_vig.get("player_1_probability"))
        p2 = float(de_vig.get("player_2_probability"))
    except (TypeError, ValueError):
        return None

    return {
        "player_1_probability": p1,
        "player_2_probability": p2,
        "favorite_side": "player_1" if p1 >= p2 else "player_2",
        "favorite_probability": max(p1, p2),
    }


def first_set_market_lines(match: dict[str, Any]) -> dict[float, dict[str, Any]]:
    markets = (match.get("markets") or {}).get("first_set_total_games")
    result: dict[float, dict[str, Any]] = {}

    if not isinstance(markets, list):
        return result

    for market in markets:
        lines = market.get("lines") or {}

        for line_text, payload in lines.items():
            try:
                line = float(line_text)
            except (TypeError, ValueError):
                continue

            if not isinstance(payload, dict):
                continue

            existing = result.setdefault(
                line,
                {
                    "line": line,
                    "source_markets": [],
                    "over": None,
                    "under": None,
                },
            )
            existing["source_markets"].append(market.get("source_market"))

            for side in ("over", "under"):
                side_data = payload.get(side)

                if not isinstance(side_data, dict):
                    continue

                try:
                    best_odds = float(side_data.get("best_odds"))
                except (TypeError, ValueError):
                    continue

                current = existing.get(side)

                if current is None or best_odds > current["odds"]:
                    existing[side] = {
                        "odds": best_odds,
                        "bookmaker": side_data.get("best_bookmaker"),
                        "median_odds": side_data.get("median_odds"),
                    }

    return result


def get_rate(
    profile: dict[str, Any],
    section: str,
    key: str,
    surface: str | None = None,
) -> tuple[float | None, int]:
    if section == "surface":
        payload = (profile.get("surface") or {}).get(surface or "")
    else:
        payload = profile.get(section)

    if not isinstance(payload, dict):
        return None, 0

    try:
        value = float(payload.get(key))
    except (TypeError, ValueError):
        return None, int(payload.get("sample_size") or 0)

    return value, int(payload.get("sample_size") or 0)


def player_market_probability(
    profile: dict[str, Any],
    side: str,
    line: float,
    surface: str,
) -> float | None:
    key = f"over_{str(line).replace('.', '_')}_rate"

    overall, overall_n = get_rate(profile, "overall", key)
    recent, recent_n = get_rate(profile, "recent_10", key)
    surface_rate, surface_n = get_rate(profile, "surface", key, surface)

    weighted: list[tuple[float, float]] = []

    if overall is not None:
        weighted.append((overall, 0.55))

    if surface_rate is not None and surface_n >= 8:
        weighted.append((surface_rate, 0.30))

    if recent is not None and recent_n >= 5:
        weighted.append((recent, 0.15))

    if not weighted:
        return None

    over_probability = (
        sum(value * weight for value, weight in weighted)
        / sum(weight for _, weight in weighted)
    )

    return over_probability if side == "over" else 1 - over_probability


def blended_probability(
    strategy: dict[str, Any],
    profile_1: dict[str, Any],
    profile_2: dict[str, Any],
    surface: str,
) -> tuple[float, dict[str, Any]]:
    player_probabilities = []

    for profile in (profile_1, profile_2):
        probability = player_market_probability(
            profile,
            strategy["side"],
            strategy["line"],
            surface,
        )

        if probability is not None:
            player_probabilities.append(probability)

    player_component = (
        sum(player_probabilities) / len(player_probabilities)
        if player_probabilities
        else strategy["base_probability"]
    )

    probability = (
        0.70 * strategy["base_probability"]
        + 0.30 * player_component
    )
    probability = min(0.95, max(0.05, probability))

    return round(probability, 6), {
        "strategy_base_probability": strategy["base_probability"],
        "player_component_probability": round(player_component, 4),
        "profile_probabilities": [
            round(value, 4) for value in player_probabilities
        ],
    }


def strategy_matches_event(
    strategy: dict[str, Any],
    match: dict[str, Any],
    favorite_probability: float,
) -> list[str]:
    reasons = []
    surface = clean_str(match.get("surface")).lower()
    gender = clean_str(match.get("gender")).lower()

    if strategy.get("surface") and surface != strategy["surface"]:
        reasons.append("surface_mismatch")

    if strategy.get("gender") and gender != strategy["gender"]:
        reasons.append("gender_mismatch")

    threshold = strategy.get("min_favorite_probability")

    if threshold is not None and favorite_probability < threshold:
        reasons.append("favorite_probability_below_strategy_threshold")

    return reasons


def main() -> None:
    odds_payload = fetch_json(ODDS_URL)
    profiles_payload = load_json(PROFILES_FILE, {})

    odds_matches = odds_payload.get("matches", [])
    profiles = profiles_payload.get("profiles", [])

    if not isinstance(odds_matches, list):
        odds_matches = []

    if not isinstance(profiles, list):
        profiles = []

    profile_index = build_profile_index(profiles)

    all_rows: list[dict[str, Any]] = []
    shadow_picks: list[dict[str, Any]] = []
    watch_picks: list[dict[str, Any]] = []
    counters: dict[str, int] = {}

    def count(key: str) -> None:
        counters[key] = counters.get(key, 0) + 1

    for match in odds_matches:
        player_1 = clean_str(match.get("player_1"))
        player_2 = clean_str(match.get("player_2"))
        surface = clean_str(match.get("surface")).lower()
        gender = clean_str(match.get("gender")).lower()

        winner = extract_match_winner(match)

        if not winner:
            count("missing_match_winner")
            continue

        lines = first_set_market_lines(match)

        if not lines:
            count("missing_first_set_totals")
            continue

        profile_1, profile_1_status = find_profile(player_1, profile_index)
        profile_2, profile_2_status = find_profile(player_2, profile_index)

        favorite_probability = winner["favorite_probability"]
        favorite_name = (
            player_1
            if winner["favorite_side"] == "player_1"
            else player_2
        )

        for strategy in STRATEGIES:
            event_reasons = strategy_matches_event(
                strategy,
                match,
                favorite_probability,
            )

            line_data = lines.get(strategy["line"])

            if not line_data:
                event_reasons.append("line_not_available")
                side_data = None
            else:
                side_data = line_data.get(strategy["side"])

                if not side_data:
                    event_reasons.append("side_not_available")

            if profile_1 is None:
                event_reasons.append("player_1_profile_missing")

            if profile_2 is None:
                event_reasons.append("player_2_profile_missing")

            odds = None
            bookmaker = None

            if side_data:
                odds = side_data["odds"]
                bookmaker = side_data["bookmaker"]

                if odds < strategy["min_odds"]:
                    event_reasons.append("odds_below_strategy_threshold")

            model_probability = None
            model_details = None
            fair_odds = None
            edge = None

            if profile_1 and profile_2:
                sample_1 = int(profile_1.get("sample_size") or 0)
                sample_2 = int(profile_2.get("sample_size") or 0)

                if sample_1 < MIN_PLAYER_SAMPLE or sample_2 < MIN_PLAYER_SAMPLE:
                    event_reasons.append("both_players_need_minimum_sample")

                if max(sample_1, sample_2) < MIN_ONE_PLAYER_SAMPLE:
                    event_reasons.append("one_player_needs_larger_sample")

                model_probability, model_details = blended_probability(
                    strategy,
                    profile_1,
                    profile_2,
                    surface,
                )
                fair_odds = round(1 / model_probability, 3)

                if odds is not None:
                    edge = round(model_probability * odds - 1, 4)

                    if edge < MIN_EDGE:
                        event_reasons.append("edge_below_threshold")

            decision = "NO_BET"

            if not event_reasons:
                decision = (
                    "WATCH"
                    if strategy["confidence"] == "watch"
                    else "SHADOW_BET"
                )

            row = {
                "pick_id": (
                    f"{match.get('event_key')}|first_set|"
                    f"{strategy['side']}|{strategy['line']:.1f}|"
                    f"{strategy['strategy_id']}"
                ),
                "event_key": match.get("event_key"),
                "date": match.get("date"),
                "time": match.get("time"),
                "match": f"{player_1} - {player_2}",
                "player_1": player_1,
                "player_2": player_2,
                "tournament": match.get("tournament"),
                "tour_level": match.get("tour_level"),
                "gender": gender,
                "surface": surface,
                "favorite": favorite_name,
                "favorite_probability": round(
                    favorite_probability,
                    6,
                ),
                "strategy_id": strategy["strategy_id"],
                "market": "first_set_total_games",
                "side": strategy["side"],
                "line": strategy["line"],
                "odds": odds,
                "bookmaker": bookmaker,
                "model_probability": model_probability,
                "fair_odds": fair_odds,
                "edge": edge,
                "confidence": strategy["confidence"],
                "decision": decision,
                "reasons": event_reasons,
                "profile_matches": {
                    "player_1_status": profile_1_status,
                    "player_1_profile_name": (
                        profile_1.get("player_name")
                        if profile_1
                        else None
                    ),
                    "player_1_sample": (
                        profile_1.get("sample_size")
                        if profile_1
                        else None
                    ),
                    "player_2_status": profile_2_status,
                    "player_2_profile_name": (
                        profile_2.get("player_name")
                        if profile_2
                        else None
                    ),
                    "player_2_sample": (
                        profile_2.get("sample_size")
                        if profile_2
                        else None
                    ),
                },
                "model_details": model_details,
            }

            all_rows.append(row)

            if decision == "SHADOW_BET":
                shadow_picks.append(row)
                count("shadow_bet")
            elif decision == "WATCH":
                watch_picks.append(row)
                count("watch")
            else:
                count("no_bet")

    shadow_picks.sort(
        key=lambda row: (
            row.get("edge") or -999,
            row.get("model_probability") or 0,
        ),
        reverse=True,
    )
    watch_picks.sort(
        key=lambda row: (
            row.get("edge") or -999,
            row.get("model_probability") or 0,
        ),
        reverse=True,
    )

    output = {
        "generated_at": now_iso(),
        "source_odds_url": ODDS_URL,
        "odds_generated_at": odds_payload.get("generated_at"),
        "model": "first_set_totals_multi_strategy_shadow_v1",
        "settings": {
            "min_player_sample": MIN_PLAYER_SAMPLE,
            "min_one_player_sample": MIN_ONE_PLAYER_SAMPLE,
            "min_edge": MIN_EDGE,
            "strategies": STRATEGIES,
        },
        "summary": {
            "odds_matches": len(odds_matches),
            "evaluated_strategy_rows": len(all_rows),
            "shadow_bets": len(shadow_picks),
            "watch_picks": len(watch_picks),
            **counters,
        },
        "shadow_picks": shadow_picks,
        "watch_picks": watch_picks,
        "all_evaluated": all_rows,
    }

    report = {
        "generated_at": now_iso(),
        "summary": output["summary"],
        "top_shadow_picks": shadow_picks[:30],
        "top_watch_picks": watch_picks[:30],
    }

    save_json(OUTPUT_FILE, output)
    save_json(REPORT_FILE, report)

    print("")
    print("FIRST SET TOTALS MULTI-STRATEGY SHADOW DONE")
    print("SUMMARY:", output["summary"])

    for pick in shadow_picks[:30]:
        print(
            "PICK:",
            pick["match"],
            "| strategy=", pick["strategy_id"],
            "|", pick["side"], pick["line"],
            "| odds=", pick["odds"],
            "| model=", pick["model_probability"],
            "| fair=", pick["fair_odds"],
            "| edge=", pick["edge"],
        )

    for pick in watch_picks[:20]:
        print(
            "WATCH:",
            pick["match"],
            "| strategy=", pick["strategy_id"],
            "|", pick["side"], pick["line"],
            "| odds=", pick["odds"],
            "| model=", pick["model_probability"],
            "| edge=", pick["edge"],
        )

    print(f"Output: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")
    print("")


if __name__ == "__main__":
    main()
