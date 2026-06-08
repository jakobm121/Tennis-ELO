import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from tennis_elo.config import DEFAULT_ELO, ROOT_DIR
from tennis_elo.utils import clean_str, load_json, now_iso, save_json


ODDS_URL = os.getenv(
    "TENNIS_ODDS_URL",
    "https://raw.githubusercontent.com/jakobm121/Ai/refs/heads/main/data/tennis_odds_today.json",
)
RATINGS_FILE = (
    ROOT_DIR
    / "data"
    / "ratings"
    / "player_ratings_latest.json"
)
OUTPUT_FILE = (
    ROOT_DIR
    / "data"
    / "predictions"
    / "match_winner_elo_shadow.json"
)
REPORT_FILE = (
    ROOT_DIR
    / "data"
    / "reports"
    / "match_winner_elo_shadow_report.json"
)

MIN_PLAYER_MATCHES = int(
    os.getenv("MATCH_WINNER_MIN_PLAYER_MATCHES", "10")
)
MIN_MODEL_PROBABILITY = float(
    os.getenv("MATCH_WINNER_MIN_MODEL_PROBABILITY", "0.55")
)
MIN_PROBABILITY_EDGE = float(
    os.getenv("MATCH_WINNER_MIN_PROBABILITY_EDGE", "0.03")
)
MIN_EV = float(
    os.getenv("MATCH_WINNER_MIN_EV", "0.05")
)
MIN_ODDS = float(
    os.getenv("MATCH_WINNER_MIN_ODDS", "1.50")
)
MAX_ODDS = float(
    os.getenv("MATCH_WINNER_MAX_ODDS", "5.00")
)
OVERALL_WEIGHT = float(
    os.getenv("MATCH_WINNER_OVERALL_WEIGHT", "0.60")
)
SURFACE_WEIGHT = float(
    os.getenv("MATCH_WINNER_SURFACE_WEIGHT", "0.40")
)
SURFACE_CONFIDENCE_MATCHES = int(
    os.getenv("MATCH_WINNER_SURFACE_CONFIDENCE_MATCHES", "20")
)
HTTP_TIMEOUT = int(
    os.getenv("TENNIS_ODDS_HTTP_TIMEOUT", "45")
)

KNOWN_SURFACES = {"hard", "clay", "grass", "carpet"}
BAD_STATUSES = {
    "finished",
    "walk over",
    "walkover",
    "cancelled",
    "canceled",
    "postponed",
    "retired",
    "abandoned",
}


def fetch_json(url: str) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "User-Agent": "Tennis-ELO match-winner shadow/1.0",
            "Accept": "application/json",
        },
    )

    with urlopen(request, timeout=HTTP_TIMEOUT) as response:
        payload = json.load(response)

    return payload if isinstance(payload, dict) else {}


def compact(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        clean_str(value).lower(),
    )


def tokens(value: Any) -> list[str]:
    return re.findall(
        r"[a-z0-9]+",
        clean_str(value).lower(),
    )


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


def aliases(value: Any) -> set[str]:
    parts = tokens(value)
    given, surname = split_name(value)
    output: set[str] = set()

    if not parts:
        return output

    output.add("".join(parts))
    output.add(" ".join(parts))

    surname_text = "".join(surname)
    given_text = "".join(given)
    initial = given[0][:1] if given else ""

    if surname_text:
        output.add(surname_text)

        if given_text:
            output.add(f"{given_text}{surname_text}")
            output.add(f"{surname_text}{given_text}")

        if initial:
            output.add(f"{initial}{surname_text}")
            output.add(f"{surname_text}{initial}")

    return {
        compact(alias)
        for alias in output
        if alias
    }


def match_score(
    query_name: str,
    rating_name: str,
) -> int:
    q_given, q_surname = split_name(query_name)
    r_given, r_surname = split_name(rating_name)

    score = 0

    if compact(query_name) == compact(rating_name):
        score += 50

    if (
        q_surname
        and r_surname
        and "".join(q_surname) == "".join(r_surname)
    ):
        score += 30

    q_initial = q_given[0][:1] if q_given else ""
    r_initial = r_given[0][:1] if r_given else ""

    if q_initial and q_initial == r_initial:
        score += 8

    return score


def build_rating_index(
    ratings: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for rating in ratings:
        names = {
            clean_str(rating.get("player_name")),
            clean_str(rating.get("player_key")),
        }

        for name in names:
            for alias in aliases(name):
                index[alias].append(rating)

    return dict(index)


def find_rating(
    player_name: str,
    index: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any] | None, str]:
    candidates: dict[str, dict[str, Any]] = {}

    for alias in aliases(player_name):
        for candidate in index.get(alias, []):
            key = clean_str(
                candidate.get("player_key")
                or candidate.get("player_name")
            )
            candidates[key] = candidate

    if not candidates:
        return None, "not_found"

    ranked = sorted(
        candidates.values(),
        key=lambda rating: (
            match_score(
                player_name,
                clean_str(rating.get("player_name")),
            ),
            int(rating.get("matches_total") or 0),
        ),
        reverse=True,
    )

    if len(ranked) > 1:
        first_score = match_score(
            player_name,
            clean_str(ranked[0].get("player_name")),
        )
        second_score = match_score(
            player_name,
            clean_str(ranked[1].get("player_name")),
        )

        if first_score == second_score:
            return None, "ambiguous"

    return ranked[0], "matched"


def expected_score(
    rating_a: float,
    rating_b: float,
) -> float:
    return 1.0 / (
        1.0
        + math.pow(
            10.0,
            (rating_b - rating_a) / 400.0,
        )
    )


def normalized_surface(value: Any) -> str:
    surface = clean_str(value).lower()

    aliases_map = {
        "hardcourt": "hard",
        "hard court": "hard",
        "red clay": "clay",
        "green clay": "clay",
        "lawn": "grass",
    }

    surface = aliases_map.get(surface, surface)

    return (
        surface
        if surface in KNOWN_SURFACES
        else "unknown"
    )


def rating_value(
    rating: dict[str, Any],
    key: str,
    default: float = DEFAULT_ELO,
) -> float:
    try:
        return float(rating.get(key))
    except (TypeError, ValueError):
        return float(default)


def surface_rating(
    rating: dict[str, Any],
    surface: str,
) -> float:
    payload = rating.get("surface_elo") or {}

    try:
        return float(payload.get(surface, DEFAULT_ELO))
    except (TypeError, ValueError):
        return float(DEFAULT_ELO)


def surface_matches(
    rating: dict[str, Any],
    surface: str,
) -> int:
    payload = rating.get("surface_matches") or {}

    try:
        return int(payload.get(surface, 0))
    except (TypeError, ValueError):
        return 0


def model_probabilities(
    rating_1: dict[str, Any],
    rating_2: dict[str, Any],
    surface: str,
) -> dict[str, Any]:
    p1_overall_elo = rating_value(
        rating_1,
        "overall_elo",
    )
    p2_overall_elo = rating_value(
        rating_2,
        "overall_elo",
    )
    overall_p1 = expected_score(
        p1_overall_elo,
        p2_overall_elo,
    )

    p1_surface_elo = p1_overall_elo
    p2_surface_elo = p2_overall_elo
    p1_surface_matches = 0
    p2_surface_matches = 0
    surface_p1 = overall_p1
    effective_surface_weight = 0.0

    if surface in KNOWN_SURFACES:
        p1_surface_elo = surface_rating(
            rating_1,
            surface,
        )
        p2_surface_elo = surface_rating(
            rating_2,
            surface,
        )
        p1_surface_matches = surface_matches(
            rating_1,
            surface,
        )
        p2_surface_matches = surface_matches(
            rating_2,
            surface,
        )
        surface_p1 = expected_score(
            p1_surface_elo,
            p2_surface_elo,
        )

        confidence = min(
            1.0,
            min(
                p1_surface_matches,
                p2_surface_matches,
            )
            / max(1, SURFACE_CONFIDENCE_MATCHES),
        )
        effective_surface_weight = (
            SURFACE_WEIGHT * confidence
        )

    effective_overall_weight = (
        OVERALL_WEIGHT
        + SURFACE_WEIGHT
        - effective_surface_weight
    )
    total_weight = (
        effective_overall_weight
        + effective_surface_weight
    )

    blended_p1 = (
        overall_p1 * effective_overall_weight
        + surface_p1 * effective_surface_weight
    ) / total_weight

    return {
        "overall_p1": overall_p1,
        "surface_p1": surface_p1,
        "blended_p1": blended_p1,
        "player_1_overall_elo": p1_overall_elo,
        "player_2_overall_elo": p2_overall_elo,
        "player_1_surface_elo": p1_surface_elo,
        "player_2_surface_elo": p2_surface_elo,
        "player_1_surface_matches": p1_surface_matches,
        "player_2_surface_matches": p2_surface_matches,
        "effective_overall_weight": effective_overall_weight,
        "effective_surface_weight": effective_surface_weight,
    }


def match_winner_market(
    match: dict[str, Any],
) -> dict[str, Any] | None:
    market = (
        (match.get("markets") or {})
        .get("match_winner")
    )

    if not isinstance(market, dict):
        return None

    de_vig = market.get("de_vig") or {}

    try:
        p1_market_probability = float(
            de_vig.get("player_1_probability")
        )
        p2_market_probability = float(
            de_vig.get("player_2_probability")
        )
    except (TypeError, ValueError):
        return None

    sides = {}

    for side in ("player_1", "player_2"):
        payload = market.get(side)

        if not isinstance(payload, dict):
            return None

        try:
            best_odds = float(payload.get("best_odds"))
        except (TypeError, ValueError):
            return None

        sides[side] = {
            "best_odds": best_odds,
            "best_bookmaker": payload.get("best_bookmaker"),
            "median_odds": payload.get("median_odds"),
            "bookmakers_count": payload.get("bookmakers_count"),
        }

    return {
        "player_1_probability": p1_market_probability,
        "player_2_probability": p2_market_probability,
        "player_1": sides["player_1"],
        "player_2": sides["player_2"],
    }


def evaluate_side(
    side: str,
    player_name: str,
    model_probability: float,
    market_probability: float,
    odds_data: dict[str, Any],
) -> dict[str, Any]:
    odds = float(odds_data["best_odds"])
    probability_edge = (
        model_probability - market_probability
    )
    ev = model_probability * odds - 1.0
    fair_odds = 1.0 / model_probability

    reasons = []

    if model_probability < MIN_MODEL_PROBABILITY:
        reasons.append("model_probability_below_threshold")

    if probability_edge < MIN_PROBABILITY_EDGE:
        reasons.append("probability_edge_below_threshold")

    if ev < MIN_EV:
        reasons.append("ev_below_threshold")

    if odds < MIN_ODDS:
        reasons.append("odds_below_threshold")

    if odds > MAX_ODDS:
        reasons.append("odds_above_threshold")

    return {
        "pick_side": side,
        "pick_player": player_name,
        "odds": round(odds, 3),
        "bookmaker": odds_data.get("best_bookmaker"),
        "median_odds": odds_data.get("median_odds"),
        "bookmakers_count": odds_data.get("bookmakers_count"),
        "model_probability": round(
            model_probability,
            6,
        ),
        "bookmaker_probability": round(
            market_probability,
            6,
        ),
        "probability_edge": round(
            probability_edge,
            6,
        ),
        "fair_odds": round(fair_odds, 3),
        "ev": round(ev, 6),
        "reasons": reasons,
    }


def status_is_eligible(match: dict[str, Any]) -> bool:
    status = clean_str(match.get("status")).lower()

    if status in BAD_STATUSES:
        return False

    return not bool(match.get("live"))


def main() -> None:
    odds_payload = fetch_json(ODDS_URL)
    ratings_payload = load_json(
        RATINGS_FILE,
        {},
    )

    matches = odds_payload.get("matches", [])
    ratings = ratings_payload.get("ratings", [])

    if not isinstance(matches, list):
        matches = []

    if not isinstance(ratings, list):
        ratings = []

    rating_index = build_rating_index(ratings)

    picks: list[dict[str, Any]] = []
    all_evaluated: list[dict[str, Any]] = []
    counters: dict[str, int] = defaultdict(int)

    for match in matches:
        if not isinstance(match, dict):
            continue

        if not status_is_eligible(match):
            counters["ineligible_status"] += 1
            continue

        player_1 = clean_str(match.get("player_1"))
        player_2 = clean_str(match.get("player_2"))

        if not player_1 or not player_2:
            counters["missing_players"] += 1
            continue

        market = match_winner_market(match)

        if market is None:
            counters["missing_match_winner_market"] += 1
            continue

        rating_1, rating_1_status = find_rating(
            player_1,
            rating_index,
        )
        rating_2, rating_2_status = find_rating(
            player_2,
            rating_index,
        )

        common_reasons = []

        if rating_1 is None:
            common_reasons.append("player_1_rating_missing")

        if rating_2 is None:
            common_reasons.append("player_2_rating_missing")

        if rating_1 is None or rating_2 is None:
            row = {
                "event_key": match.get("event_key"),
                "match": f"{player_1} - {player_2}",
                "player_1": player_1,
                "player_2": player_2,
                "surface": normalized_surface(
                    match.get("surface")
                ),
                "decision": "NO_BET",
                "reasons": common_reasons,
                "rating_matches": {
                    "player_1_status": rating_1_status,
                    "player_2_status": rating_2_status,
                },
            }
            all_evaluated.append(row)

            for reason in common_reasons:
                counters[reason] += 1

            continue

        p1_matches = int(
            rating_1.get("matches_total") or 0
        )
        p2_matches = int(
            rating_2.get("matches_total") or 0
        )

        if p1_matches < MIN_PLAYER_MATCHES:
            common_reasons.append(
                "player_1_insufficient_history"
            )

        if p2_matches < MIN_PLAYER_MATCHES:
            common_reasons.append(
                "player_2_insufficient_history"
            )

        surface = normalized_surface(
            match.get("surface")
        )
        probabilities = model_probabilities(
            rating_1,
            rating_2,
            surface,
        )

        p1_model = probabilities["blended_p1"]
        p2_model = 1.0 - p1_model

        p1_evaluation = evaluate_side(
            "player_1",
            player_1,
            p1_model,
            market["player_1_probability"],
            market["player_1"],
        )
        p2_evaluation = evaluate_side(
            "player_2",
            player_2,
            p2_model,
            market["player_2_probability"],
            market["player_2"],
        )

        # Pick at most one side per match: the side with higher EV.
        selected = max(
            (p1_evaluation, p2_evaluation),
            key=lambda item: item["ev"],
        )

        reasons = [
            *common_reasons,
            *selected["reasons"],
        ]
        decision = (
            "SHADOW_BET"
            if not reasons
            else "NO_BET"
        )

        row = {
            "pick_id": (
                f"{match.get('event_key')}|match_winner|"
                f"{selected['pick_side']}"
            ),
            "event_key": match.get("event_key"),
            "date": match.get("date"),
            "time": match.get("time"),
            "match": f"{player_1} - {player_2}",
            "player_1": player_1,
            "player_2": player_2,
            "tournament": match.get("tournament"),
            "tournament_key": match.get("tournament_key"),
            "round": match.get("round"),
            "gender": match.get("gender"),
            "tour_level": match.get("tour_level"),
            "surface": surface,
            "market": "match_winner",
            "pick_side": selected["pick_side"],
            "pick_player": selected["pick_player"],
            "odds": selected["odds"],
            "bookmaker": selected["bookmaker"],
            "median_odds": selected["median_odds"],
            "bookmakers_count": selected["bookmakers_count"],
            "model_probability": selected[
                "model_probability"
            ],
            "bookmaker_probability": selected[
                "bookmaker_probability"
            ],
            "probability_edge": selected[
                "probability_edge"
            ],
            "fair_odds": selected["fair_odds"],
            "ev": selected["ev"],
            "decision": decision,
            "confidence": (
                "high"
                if selected["ev"] >= 0.10
                and selected["probability_edge"] >= 0.05
                else "medium"
            ),
            "reasons": reasons,
            "player_1_market": p1_evaluation,
            "player_2_market": p2_evaluation,
            "rating_matches": {
                "player_1_status": rating_1_status,
                "player_1_rating_name": rating_1.get(
                    "player_name"
                ),
                "player_1_matches": p1_matches,
                "player_2_status": rating_2_status,
                "player_2_rating_name": rating_2.get(
                    "player_name"
                ),
                "player_2_matches": p2_matches,
            },
            "elo_details": {
                "player_1_overall_elo": round(
                    probabilities["player_1_overall_elo"],
                    3,
                ),
                "player_2_overall_elo": round(
                    probabilities["player_2_overall_elo"],
                    3,
                ),
                "player_1_surface_elo": round(
                    probabilities["player_1_surface_elo"],
                    3,
                ),
                "player_2_surface_elo": round(
                    probabilities["player_2_surface_elo"],
                    3,
                ),
                "player_1_surface_matches": probabilities[
                    "player_1_surface_matches"
                ],
                "player_2_surface_matches": probabilities[
                    "player_2_surface_matches"
                ],
                "overall_p1_probability": round(
                    probabilities["overall_p1"],
                    6,
                ),
                "surface_p1_probability": round(
                    probabilities["surface_p1"],
                    6,
                ),
                "blended_p1_probability": round(
                    probabilities["blended_p1"],
                    6,
                ),
                "effective_overall_weight": round(
                    probabilities[
                        "effective_overall_weight"
                    ],
                    4,
                ),
                "effective_surface_weight": round(
                    probabilities[
                        "effective_surface_weight"
                    ],
                    4,
                ),
            },
        }

        all_evaluated.append(row)

        if decision == "SHADOW_BET":
            picks.append(row)
            counters["shadow_bet"] += 1
        else:
            counters["no_bet"] += 1

            for reason in reasons:
                counters[reason] += 1

    picks.sort(
        key=lambda row: (
            row.get("ev") or -999,
            row.get("probability_edge") or -999,
        ),
        reverse=True,
    )

    output = {
        "generated_at": now_iso(),
        "source_odds_url": ODDS_URL,
        "odds_generated_at": odds_payload.get(
            "generated_at"
        ),
        "ratings_generated_at": ratings_payload.get(
            "generated_at"
        ),
        "model": "match_winner_elo_shadow_v1",
        "settings": {
            "min_player_matches": MIN_PLAYER_MATCHES,
            "min_model_probability": (
                MIN_MODEL_PROBABILITY
            ),
            "min_probability_edge": (
                MIN_PROBABILITY_EDGE
            ),
            "min_ev": MIN_EV,
            "min_odds": MIN_ODDS,
            "max_odds": MAX_ODDS,
            "overall_weight": OVERALL_WEIGHT,
            "surface_weight": SURFACE_WEIGHT,
            "surface_confidence_matches": (
                SURFACE_CONFIDENCE_MATCHES
            ),
        },
        "summary": {
            "odds_matches": len(matches),
            "ratings_players": len(ratings),
            "evaluated_matches": len(all_evaluated),
            "shadow_bets": len(picks),
            "no_bets": (
                len(all_evaluated) - len(picks)
            ),
            **dict(counters),
        },
        "shadow_picks": picks,
        "all_evaluated": all_evaluated,
    }

    report = {
        "generated_at": now_iso(),
        "summary": output["summary"],
        "settings": output["settings"],
        "top_shadow_picks": picks[:30],
    }

    save_json(OUTPUT_FILE, output)
    save_json(REPORT_FILE, report)

    print("")
    print("MATCH WINNER ELO SHADOW DONE")
    print("SUMMARY:", output["summary"])

    for pick in picks[:30]:
        print(
            "PICK:",
            pick["match"],
            "| pick=", pick["pick_player"],
            "| odds=", pick["odds"],
            "| model=", pick["model_probability"],
            "| book=", pick["bookmaker_probability"],
            "| edge=", pick["probability_edge"],
            "| ev=", pick["ev"],
            "| confidence=", pick["confidence"],
        )

    print(f"Output: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")
    print("")


if __name__ == "__main__":
    main()
