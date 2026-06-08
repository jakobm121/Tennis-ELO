import json
import math
import os
import re
from datetime import datetime, timezone
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
OUTPUT_FILE = ROOT_DIR / "data" / "predictions" / "first_set_under_9_5_shadow.json"
REPORT_FILE = ROOT_DIR / "data" / "reports" / "first_set_under_9_5_shadow_report.json"

MIN_FAVORITE_PROBABILITY = float(
    os.getenv("FIRST_SET_SHADOW_MIN_FAVORITE_PROBABILITY", "0.75")
)
HIGH_CONFIDENCE_FAVORITE_PROBABILITY = float(
    os.getenv("FIRST_SET_SHADOW_HIGH_FAVORITE_PROBABILITY", "0.85")
)
MIN_UNDER_ODDS = float(
    os.getenv("FIRST_SET_SHADOW_MIN_UNDER_ODDS", "1.80")
)
MIN_PLAYER_SAMPLE = int(
    os.getenv("FIRST_SET_SHADOW_MIN_PLAYER_SAMPLE", "15")
)
MIN_ONE_PLAYER_SAMPLE = int(
    os.getenv("FIRST_SET_SHADOW_MIN_ONE_PLAYER_SAMPLE", "30")
)
MIN_EDGE = float(
    os.getenv("FIRST_SET_SHADOW_MIN_EDGE", "0.05")
)
MIN_MODEL_UNDER_PROBABILITY = float(
    os.getenv("FIRST_SET_SHADOW_MIN_MODEL_UNDER_PROBABILITY", "0.56")
)
HTTP_TIMEOUT = int(os.getenv("TENNIS_ODDS_HTTP_TIMEOUT", "45"))


def fetch_json(url: str) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "User-Agent": "Tennis-ELO shadow-picks/1.0",
            "Accept": "application/json",
        },
    )

    with urlopen(request, timeout=HTTP_TIMEOUT) as response:
        payload = json.load(response)

    return payload if isinstance(payload, dict) else {}


def compact(value: Any) -> str:
    text = clean_str(value).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


NAME_PARTICLES = {
    "de", "del", "della", "di", "da", "dos", "das",
    "van", "von", "der", "den", "la", "le", "du",
}


def tokens(value: Any) -> list[str]:
    text = clean_str(value).lower()
    return re.findall(r"[a-z0-9]+", text)


def split_name(value: Any) -> tuple[list[str], list[str]]:
    """
    Supports both:
      I. Surname
      Surname I.
      I. van Assche
      Van Assche L.
      S. De La Fuente
      De La Fuente S.
    """
    parts = tokens(value)

    if not parts:
        return [], []

    if len(parts) == 1:
        return [], parts

    first_is_initial = len(parts[0]) == 1
    last_is_initial = len(parts[-1]) == 1

    if first_is_initial and not last_is_initial:
        given = [parts[0]]
        surname = parts[1:]
        return given, surname

    if last_is_initial and not first_is_initial:
        given = [parts[-1]]
        surname = parts[:-1]
        return given, surname

    # Full names: first token is treated as given name, while all remaining
    # tokens form the surname. This preserves compound surnames.
    given = [parts[0]]
    surname = parts[1:]

    return given, surname


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

    # Also support particle-stripped aliases, but never replace the full
    # compound surname with them.
    stripped_surname = "".join(
        part for part in surname
        if part not in NAME_PARTICLES
    )

    if stripped_surname and stripped_surname != surname_joined:
        aliases.add(stripped_surname)
        if initial:
            aliases.add(f"{initial}{stripped_surname}")
            aliases.add(f"{stripped_surname}{initial}")

    return {compact(alias) for alias in aliases if alias}


def build_profile_index(profiles: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}

    for profile in profiles:
        name = profile.get("player_name")
        for alias in name_aliases(name):
            index.setdefault(alias, []).append(profile)

    return index


def profile_match_score(query_name: str, profile_name: str) -> int:
    query_parts = tokens(query_name)
    profile_parts = tokens(profile_name)
    query_given, query_surname = split_name(query_name)
    profile_given, profile_surname = split_name(profile_name)

    if not query_parts or not profile_parts:
        return -1

    score = 0

    query_surname_joined = "".join(query_surname)
    profile_surname_joined = "".join(profile_surname)

    if query_surname_joined and query_surname_joined == profile_surname_joined:
        score += 30

    query_stripped = "".join(
        part for part in query_surname
        if part not in NAME_PARTICLES
    )
    profile_stripped = "".join(
        part for part in profile_surname
        if part not in NAME_PARTICLES
    )

    if query_stripped and query_stripped == profile_stripped:
        score += 12

    query_initial = query_given[0][:1] if query_given else ""
    profile_initial = profile_given[0][:1] if profile_given else ""

    if query_initial and query_initial == profile_initial:
        score += 8

    if compact(query_name) == compact(profile_name):
        score += 50

    # Penalize clear surname mismatch.
    if (
        query_surname_joined
        and profile_surname_joined
        and query_surname_joined != profile_surname_joined
        and query_stripped != profile_stripped
    ):
        score -= 20

    return score


def find_profile(
    player_name: str,
    index: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any] | None, str]:
    candidates: list[dict[str, Any]] = []

    for alias in name_aliases(player_name):
        candidates.extend(index.get(alias, []))

    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        unique[clean_str(candidate.get("player_name"))] = candidate

    if not unique:
        return None, "not_found"

    ranked = sorted(
        unique.values(),
        key=lambda profile: (
            profile_match_score(player_name, clean_str(profile.get("player_name"))),
            int(profile.get("sample_size") or 0),
        ),
        reverse=True,
    )

    best = ranked[0]

    if len(ranked) > 1:
        top_score = profile_match_score(
            player_name,
            clean_str(ranked[0].get("player_name")),
        )
        second_score = profile_match_score(
            player_name,
            clean_str(ranked[1].get("player_name")),
        )

        if top_score == second_score:
            return None, "ambiguous"

    return best, "matched"


def get_profile_rate(
    profile: dict[str, Any],
    surface: str,
    section: str,
    key: str,
) -> tuple[float | None, int]:
    if section == "surface":
        payload = (
            (profile.get("surface") or {}).get(surface)
            if surface
            else None
        )
    else:
        payload = profile.get(section)

    if not isinstance(payload, dict):
        return None, 0

    try:
        value = float(payload.get(key))
    except (TypeError, ValueError):
        return None, int(payload.get("sample_size") or 0)

    return value, int(payload.get("sample_size") or 0)


def empirical_strong_favorite_under(favorite_probability: float) -> float:
    # Walk-forward segment results:
    # >= 0.75 -> 58.53% Under
    # >= 0.85 -> 61.16% Under
    if favorite_probability >= 0.85:
        return 0.6116
    if favorite_probability >= 0.80:
        return 0.5833
    return 0.5853


def player_under_signal(
    profile: dict[str, Any],
    surface: str,
) -> dict[str, Any]:
    overall_over, overall_sample = get_profile_rate(
        profile,
        surface,
        "overall",
        "over_9_5_rate",
    )
    recent_over, recent_sample = get_profile_rate(
        profile,
        surface,
        "recent_10",
        "over_9_5_rate",
    )
    surface_over, surface_sample = get_profile_rate(
        profile,
        surface,
        "surface",
        "over_9_5_rate",
    )

    overall_under = 1.0 - overall_over if overall_over is not None else None
    recent_under = 1.0 - recent_over if recent_over is not None else None
    surface_under = 1.0 - surface_over if surface_over is not None else None

    weighted: list[tuple[float, float]] = []

    if overall_under is not None:
        weighted.append((overall_under, 0.55))

    if surface_under is not None and surface_sample >= 8:
        weighted.append((surface_under, 0.30))

    if recent_under is not None and recent_sample >= 5:
        weighted.append((recent_under, 0.15))

    if not weighted:
        model_under = None
    else:
        total_weight = sum(weight for _, weight in weighted)
        model_under = sum(value * weight for value, weight in weighted) / total_weight

    return {
        "sample_size": int(profile.get("sample_size") or 0),
        "overall_under_rate": (
            round(overall_under, 4) if overall_under is not None else None
        ),
        "surface_under_rate": (
            round(surface_under, 4) if surface_under is not None else None
        ),
        "surface_sample": surface_sample,
        "recent_10_under_rate": (
            round(recent_under, 4) if recent_under is not None else None
        ),
        "recent_10_sample": recent_sample,
        "blended_under_rate": (
            round(model_under, 4) if model_under is not None else None
        ),
    }


def model_under_probability(
    profile_1: dict[str, Any],
    profile_2: dict[str, Any],
    surface: str,
    favorite_probability: float,
) -> tuple[float, dict[str, Any]]:
    signal_1 = player_under_signal(profile_1, surface)
    signal_2 = player_under_signal(profile_2, surface)

    player_values = [
        value
        for value in [
            signal_1.get("blended_under_rate"),
            signal_2.get("blended_under_rate"),
        ]
        if value is not None
    ]

    player_component = (
        sum(player_values) / len(player_values)
        if player_values
        else 0.55
    )

    segment_component = empirical_strong_favorite_under(
        favorite_probability
    )

    # Keep the empirically validated strong-favorite segment as the anchor.
    probability = (
        0.65 * segment_component
        + 0.35 * player_component
    )

    probability = min(0.70, max(0.50, probability))

    return round(probability, 6), {
        "segment_under_probability": round(segment_component, 4),
        "player_component_under_probability": round(player_component, 4),
        "player_1": signal_1,
        "player_2": signal_2,
    }


def extract_match_winner(
    match: dict[str, Any],
) -> dict[str, Any] | None:
    market = (match.get("markets") or {}).get("match_winner")

    if not isinstance(market, dict):
        return None

    de_vig = market.get("de_vig") or {}

    try:
        p1 = float(de_vig.get("player_1_probability"))
        p2 = float(de_vig.get("player_2_probability"))
    except (TypeError, ValueError):
        return None

    favorite_side = "player_1" if p1 >= p2 else "player_2"
    favorite_probability = max(p1, p2)

    return {
        "player_1_probability": p1,
        "player_2_probability": p2,
        "favorite_side": favorite_side,
        "favorite_probability": favorite_probability,
    }


def find_first_set_line_9_5(
    match: dict[str, Any],
) -> dict[str, Any] | None:
    markets = (match.get("markets") or {}).get("first_set_total_games")

    if not isinstance(markets, list):
        return None

    candidates: list[dict[str, Any]] = []

    for market in markets:
        lines = market.get("lines") or {}
        line = lines.get("9.5")

        if not isinstance(line, dict):
            continue

        under = line.get("under")
        over = line.get("over")

        if not isinstance(under, dict):
            continue

        try:
            under_odds = float(under.get("best_odds"))
        except (TypeError, ValueError):
            continue

        over_odds = None
        if isinstance(over, dict):
            try:
                over_odds = float(over.get("best_odds"))
            except (TypeError, ValueError):
                pass

        candidates.append(
            {
                "source_market": market.get("source_market"),
                "line": 9.5,
                "under_odds": under_odds,
                "under_bookmaker": under.get("best_bookmaker"),
                "under_median_odds": under.get("median_odds"),
                "over_odds": over_odds,
                "over_bookmaker": (
                    over.get("best_bookmaker")
                    if isinstance(over, dict)
                    else None
                ),
            }
        )

    if not candidates:
        return None

    return max(candidates, key=lambda item: item["under_odds"])


def confidence_label(favorite_probability: float) -> str:
    if favorite_probability >= HIGH_CONFIDENCE_FAVORITE_PROBABILITY:
        return "high"
    return "medium"


def decision_reasons(
    favorite_probability: float,
    under_odds: float,
    profile_1: dict[str, Any] | None,
    profile_2: dict[str, Any] | None,
    model_probability: float | None,
    edge: float | None,
) -> list[str]:
    reasons = []

    if favorite_probability < MIN_FAVORITE_PROBABILITY:
        reasons.append("favorite_probability_below_threshold")

    if under_odds < MIN_UNDER_ODDS:
        reasons.append("under_odds_below_threshold")

    if profile_1 is None:
        reasons.append("player_1_profile_missing")

    if profile_2 is None:
        reasons.append("player_2_profile_missing")

    if profile_1 and profile_2:
        sample_1 = int(profile_1.get("sample_size") or 0)
        sample_2 = int(profile_2.get("sample_size") or 0)

        if sample_1 < MIN_PLAYER_SAMPLE or sample_2 < MIN_PLAYER_SAMPLE:
            reasons.append("both_players_need_minimum_sample")

        if max(sample_1, sample_2) < MIN_ONE_PLAYER_SAMPLE:
            reasons.append("one_player_needs_larger_sample")

    if (
        model_probability is not None
        and model_probability < MIN_MODEL_UNDER_PROBABILITY
    ):
        reasons.append("model_probability_below_threshold")

    if edge is not None and edge < MIN_EDGE:
        reasons.append("edge_below_threshold")

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

    rows = []
    shadow_picks = []
    counters: dict[str, int] = {}

    def count(key: str) -> None:
        counters[key] = counters.get(key, 0) + 1

    for match in odds_matches:
        player_1 = clean_str(match.get("player_1"))
        player_2 = clean_str(match.get("player_2"))
        surface = clean_str(match.get("surface")).lower()

        winner_market = extract_match_winner(match)
        first_set_market = find_first_set_line_9_5(match)

        if not winner_market:
            count("missing_match_winner")
            continue

        if not first_set_market:
            count("missing_first_set_9_5")
            continue

        profile_1, profile_1_status = find_profile(
            player_1,
            profile_index,
        )
        profile_2, profile_2_status = find_profile(
            player_2,
            profile_index,
        )

        favorite_probability = winner_market["favorite_probability"]
        under_odds = first_set_market["under_odds"]

        model_probability = None
        model_details = None
        fair_odds = None
        edge = None

        if profile_1 and profile_2:
            model_probability, model_details = model_under_probability(
                profile_1,
                profile_2,
                surface,
                favorite_probability,
            )
            fair_odds = round(1.0 / model_probability, 3)
            edge = round(model_probability * under_odds - 1.0, 4)

        reasons = decision_reasons(
            favorite_probability,
            under_odds,
            profile_1,
            profile_2,
            model_probability,
            edge,
        )

        decision = "SHADOW_BET" if not reasons else "NO_BET"

        favorite_name = (
            player_1
            if winner_market["favorite_side"] == "player_1"
            else player_2
        )

        row = {
            "event_key": match.get("event_key"),
            "date": match.get("date"),
            "time": match.get("time"),
            "player_1": player_1,
            "player_2": player_2,
            "match": f"{player_1} - {player_2}",
            "tournament": match.get("tournament"),
            "tour_level": match.get("tour_level"),
            "gender": match.get("gender"),
            "surface": surface,
            "favorite": favorite_name,
            "favorite_side": winner_market["favorite_side"],
            "favorite_probability": round(favorite_probability, 6),
            "market": "1st Set Under 9.5 Games",
            "line": 9.5,
            "under_odds": under_odds,
            "under_bookmaker": first_set_market.get(
                "under_bookmaker"
            ),
            "over_odds": first_set_market.get("over_odds"),
            "model_under_probability": model_probability,
            "fair_odds_under": fair_odds,
            "edge": edge,
            "confidence": confidence_label(
                favorite_probability
            ),
            "decision": decision,
            "reasons": reasons,
            "profile_matches": {
                "player_1_status": profile_1_status,
                "player_1_profile_name": (
                    profile_1.get("player_name")
                    if profile_1
                    else None
                ),
                "player_2_status": profile_2_status,
                "player_2_profile_name": (
                    profile_2.get("player_name")
                    if profile_2
                    else None
                ),
            },
            "model_details": model_details,
        }

        rows.append(row)

        if decision == "SHADOW_BET":
            shadow_picks.append(row)
            count("shadow_bet")
        else:
            count("no_bet")

    shadow_picks.sort(
        key=lambda row: (
            row.get("edge") or -1,
            row.get("favorite_probability") or 0,
        ),
        reverse=True,
    )

    output = {
        "generated_at": now_iso(),
        "source_odds_url": ODDS_URL,
        "odds_generated_at": odds_payload.get("generated_at"),
        "model": "first_set_under_9_5_shadow_v1",
        "settings": {
            "min_favorite_probability": MIN_FAVORITE_PROBABILITY,
            "high_confidence_favorite_probability": (
                HIGH_CONFIDENCE_FAVORITE_PROBABILITY
            ),
            "min_under_odds": MIN_UNDER_ODDS,
            "min_player_sample": MIN_PLAYER_SAMPLE,
            "min_one_player_sample": MIN_ONE_PLAYER_SAMPLE,
            "min_model_under_probability": (
                MIN_MODEL_UNDER_PROBABILITY
            ),
            "min_edge": MIN_EDGE,
        },
        "summary": {
            "odds_matches": len(odds_matches),
            "evaluated_matches": len(rows),
            "shadow_bets": len(shadow_picks),
            "no_bets": len(rows) - len(shadow_picks),
            **counters,
        },
        "shadow_picks": shadow_picks,
        "all_evaluated": rows,
    }

    report = {
        "generated_at": now_iso(),
        "source_odds_url": ODDS_URL,
        "summary": output["summary"],
        "top_shadow_picks": shadow_picks[:20],
    }

    save_json(OUTPUT_FILE, output)
    save_json(REPORT_FILE, report)

    print("")
    print("FIRST SET UNDER 9.5 SHADOW PICKS DONE")
    print("SUMMARY:", output["summary"])

    for pick in shadow_picks[:20]:
        print(
            "PICK:",
            pick["match"],
            "favorite_prob=",
            pick["favorite_probability"],
            "under_odds=",
            pick["under_odds"],
            "model_prob=",
            pick["model_under_probability"],
            "edge=",
            pick["edge"],
            "confidence=",
            pick["confidence"],
        )

    print(f"Output: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")
    print("")


if __name__ == "__main__":
    main()
