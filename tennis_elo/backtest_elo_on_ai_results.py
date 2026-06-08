import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from statistics import mean
from typing import Any
from urllib.request import Request, urlopen

from tennis_elo.config import (
    CANONICAL_MATCHES_FILE,
    DEFAULT_ELO,
    K_FACTOR,
    ROOT_DIR,
    SURFACE_K_FACTOR,
)
from tennis_elo.utils import clean_str, load_json, now_iso, save_json


AI_RESULTS_URL = os.getenv(
    "AI_TENNIS_RESULTS_URL",
    "https://raw.githubusercontent.com/jakobm121/Ai/refs/heads/main/data/tennis_results.json",
)
OUTPUT_FILE = (
    ROOT_DIR
    / "data"
    / "backtests"
    / "elo_on_ai_results_backtest.json"
)
REPORT_FILE = (
    ROOT_DIR
    / "data"
    / "reports"
    / "elo_on_ai_results_backtest_report.json"
)

HTTP_TIMEOUT = int(os.getenv("AI_RESULTS_HTTP_TIMEOUT", "60"))
MIN_PLAYER_MATCHES = int(
    os.getenv("ELO_AI_RESULTS_MIN_PLAYER_MATCHES", "10")
)
OVERALL_WEIGHT = float(
    os.getenv("ELO_AI_RESULTS_OVERALL_WEIGHT", "0.60")
)
SURFACE_WEIGHT = float(
    os.getenv("ELO_AI_RESULTS_SURFACE_WEIGHT", "0.40")
)
SURFACE_CONFIDENCE_MATCHES = int(
    os.getenv("ELO_AI_RESULTS_SURFACE_CONFIDENCE_MATCHES", "20")
)
EV_THRESHOLDS = [
    float(value.strip())
    for value in os.getenv(
        "ELO_AI_RESULTS_EV_THRESHOLDS",
        "0.00,0.03,0.05,0.08,0.10,0.15",
    ).split(",")
    if value.strip()
]
MIN_MODEL_PROBABILITIES = [
    float(value.strip())
    for value in os.getenv(
        "ELO_AI_RESULTS_MIN_MODEL_PROBABILITIES",
        "0.50,0.55,0.60",
    ).split(",")
    if value.strip()
]
MIN_ODDS_VALUES = [
    float(value.strip())
    for value in os.getenv(
        "ELO_AI_RESULTS_MIN_ODDS_VALUES",
        "1.00,1.50,1.75,2.00",
    ).split(",")
    if value.strip()
]

KNOWN_SURFACES = {"hard", "clay", "grass", "carpet"}
SETTLED_RESULTS = {"win", "loss"}


def fetch_json(url: str) -> Any:
    request = Request(
        url,
        headers={
            "User-Agent": "Tennis-ELO AI-results backtest/1.0",
            "Accept": "application/json",
        },
    )

    with urlopen(request, timeout=HTTP_TIMEOUT) as response:
        return json.load(response)


def parse_date(raw: Any):
    value = clean_str(raw)

    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%d.%m.%Y",
        "%d.%m.%y",
        "%Y%m%d",
    ):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        ).date()
    except ValueError:
        return None


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


def name_aliases(value: Any) -> set[str]:
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


def canonical_name(value: Any) -> str:
    aliases = sorted(name_aliases(value))

    if not aliases:
        return ""

    # Prefer the longest alias because it preserves compound surnames.
    return max(aliases, key=len)


def expected_score(a: float, b: float) -> float:
    return 1.0 / (
        1.0 + math.pow(10.0, (b - a) / 400.0)
    )


def update_pair(
    winner: float,
    loser: float,
    k_factor: float,
) -> tuple[float, float]:
    probability = expected_score(winner, loser)

    return (
        winner + k_factor * (1.0 - probability),
        loser - k_factor * (1.0 - probability),
    )


def normalized_surface(value: Any) -> str:
    surface = clean_str(value).lower()

    aliases = {
        "hardcourt": "hard",
        "hard court": "hard",
        "cement": "hard",
        "red clay": "clay",
        "green clay": "clay",
        "lawn": "grass",
    }

    surface = aliases.get(surface, surface)

    return (
        surface
        if surface in KNOWN_SURFACES
        else "unknown"
    )


def new_player_state() -> dict[str, Any]:
    return {
        "overall_elo": float(DEFAULT_ELO),
        "surface_elo": {},
        "matches_total": 0,
        "surface_matches": defaultdict(int),
    }


def get_surface_rating(
    player: dict[str, Any],
    surface: str,
) -> float:
    return float(
        player["surface_elo"].get(
            surface,
            DEFAULT_ELO,
        )
    )


def blended_probability(
    p1: dict[str, Any],
    p2: dict[str, Any],
    surface: str,
) -> dict[str, float]:
    p1_overall = float(p1["overall_elo"])
    p2_overall = float(p2["overall_elo"])
    overall_p1 = expected_score(
        p1_overall,
        p2_overall,
    )

    p1_surface = p1_overall
    p2_surface = p2_overall
    surface_p1 = overall_p1
    p1_surface_matches = 0
    p2_surface_matches = 0
    effective_surface_weight = 0.0

    if surface in KNOWN_SURFACES:
        p1_surface = get_surface_rating(
            p1,
            surface,
        )
        p2_surface = get_surface_rating(
            p2,
            surface,
        )
        p1_surface_matches = int(
            p1["surface_matches"][surface]
        )
        p2_surface_matches = int(
            p2["surface_matches"][surface]
        )
        surface_p1 = expected_score(
            p1_surface,
            p2_surface,
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
        "p1_overall_elo": p1_overall,
        "p2_overall_elo": p2_overall,
        "p1_surface_elo": p1_surface,
        "p2_surface_elo": p2_surface,
        "p1_surface_matches": p1_surface_matches,
        "p2_surface_matches": p2_surface_matches,
    }


def valid_canonical_match(row: dict[str, Any]) -> bool:
    if not parse_date(row.get("date")):
        return False

    player_1 = clean_str(row.get("player_1"))
    player_2 = clean_str(row.get("player_2"))
    winner = clean_str(row.get("winner"))

    if not player_1 or not player_2 or not winner:
        return False

    p1_key = canonical_name(player_1)
    p2_key = canonical_name(player_2)
    winner_key = canonical_name(winner)

    if (
        not p1_key
        or not p2_key
        or p1_key == p2_key
        or winner_key not in {p1_key, p2_key}
    ):
        return False

    if row.get("ready_for_elo") is False:
        return False

    if row.get("retired") or row.get("walkover"):
        return False

    return True


def valid_ai_result(row: dict[str, Any]) -> bool:
    if clean_str(row.get("result")).lower() not in SETTLED_RESULTS:
        return False

    if not parse_date(row.get("date")):
        return False

    if not clean_str(row.get("player_name")):
        return False

    if not clean_str(row.get("opponent_name")):
        return False

    try:
        odds = float(row.get("odds"))
    except (TypeError, ValueError):
        return False

    return odds > 1.0


def flat_profit(result: str, odds: float) -> float:
    if result == "win":
        return odds - 1.0

    if result == "loss":
        return -1.0

    return 0.0


def summarize_bets(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    wins = sum(row["result"] == "win" for row in rows)
    losses = sum(row["result"] == "loss" for row in rows)
    profit = round(
        sum(float(row["flat_profit"]) for row in rows),
        4,
    )
    sample = len(rows)

    return {
        "bets": sample,
        "wins": wins,
        "losses": losses,
        "hit_rate": (
            round(wins / sample, 4)
            if sample
            else None
        ),
        "profit": profit,
        "roi": (
            round(profit / sample, 4)
            if sample
            else None
        ),
        "avg_odds": (
            round(mean(row["odds"] for row in rows), 3)
            if sample
            else None
        ),
        "avg_model_probability": (
            round(
                mean(
                    row["elo_model_probability"]
                    for row in rows
                ),
                4,
            )
            if sample
            else None
        ),
        "avg_ev": (
            round(mean(row["elo_ev"] for row in rows), 4)
            if sample
            else None
        ),
    }


def threshold_grid(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []

    for min_ev in EV_THRESHOLDS:
        for min_probability in MIN_MODEL_PROBABILITIES:
            for min_odds in MIN_ODDS_VALUES:
                subset = [
                    row
                    for row in rows
                    if row["elo_ev"] >= min_ev
                    and row["elo_model_probability"]
                    >= min_probability
                    and row["odds"] >= min_odds
                ]

                summary = summarize_bets(subset)

                output.append(
                    {
                        "min_ev": min_ev,
                        "min_model_probability": (
                            min_probability
                        ),
                        "min_odds": min_odds,
                        **summary,
                    }
                )

    output.sort(
        key=lambda row: (
            row["roi"] if row["roi"] is not None else -999,
            row["bets"],
        ),
        reverse=True,
    )

    return output


def main() -> None:
    canonical_payload = load_json(
        CANONICAL_MATCHES_FILE,
        {},
    )
    canonical_matches = (
        canonical_payload.get("matches", [])
        if isinstance(canonical_payload, dict)
        else []
    )

    if not isinstance(canonical_matches, list):
        canonical_matches = []

    canonical_matches = [
        row
        for row in canonical_matches
        if isinstance(row, dict)
        and valid_canonical_match(row)
    ]
    canonical_matches.sort(
        key=lambda row: (
            parse_date(row.get("date")),
            clean_str(
                row.get("tournament")
                or row.get("tournament_code")
            ),
            clean_str(
                row.get("canonical_match_id")
            ),
        )
    )

    ai_payload = fetch_json(AI_RESULTS_URL)
    ai_results = (
        ai_payload
        if isinstance(ai_payload, list)
        else ai_payload.get("results", [])
        if isinstance(ai_payload, dict)
        else []
    )

    ai_results = [
        row
        for row in ai_results
        if isinstance(row, dict)
        and valid_ai_result(row)
    ]

    by_date: dict[Any, list[dict[str, Any]]] = defaultdict(list)

    for row in ai_results:
        by_date[parse_date(row["date"])].append(row)

    players: dict[str, dict[str, Any]] = defaultdict(
        new_player_state
    )
    evaluated: list[dict[str, Any]] = []
    counters: dict[str, int] = defaultdict(int)

    ai_dates = sorted(by_date)
    ai_date_set = set(ai_dates)

    # Walk forward through canonical history.
    for row in canonical_matches:
        match_date = parse_date(row["date"])
        player_1 = clean_str(row["player_1"])
        player_2 = clean_str(row["player_2"])
        winner = clean_str(row["winner"])

        p1_key = canonical_name(player_1)
        p2_key = canonical_name(player_2)
        winner_key = canonical_name(winner)
        surface = normalized_surface(row.get("surface"))

        # Before updating this date's match, evaluate any AI picks for this date
        # that correspond to this player pair. This keeps the ELO strictly pre-match.
        if match_date in ai_date_set:
            remaining = []

            for pick in by_date[match_date]:
                pick_player = clean_str(pick["player_name"])
                opponent = clean_str(pick["opponent_name"])
                pick_key = canonical_name(pick_player)
                opponent_key = canonical_name(opponent)

                pair_matches = {
                    pick_key,
                    opponent_key,
                } == {
                    p1_key,
                    p2_key,
                }

                if not pair_matches:
                    remaining.append(pick)
                    continue

                pick_state = players[pick_key]
                opponent_state = players[opponent_key]

                if (
                    int(pick_state["matches_total"])
                    < MIN_PLAYER_MATCHES
                    or int(opponent_state["matches_total"])
                    < MIN_PLAYER_MATCHES
                ):
                    counters[
                        "insufficient_player_history"
                    ] += 1
                    continue

                model = blended_probability(
                    pick_state,
                    opponent_state,
                    surface,
                )
                model_probability = model["blended_p1"]
                odds = float(pick["odds"])
                result = clean_str(
                    pick["result"]
                ).lower()
                ev = model_probability * odds - 1.0

                evaluated.append(
                    {
                        "pick_id": pick.get("pick_id"),
                        "event_key": pick.get("event_key"),
                        "date": match_date.isoformat(),
                        "match": pick.get("match"),
                        "pick_player": pick_player,
                        "opponent": opponent,
                        "tournament": pick.get("tournament"),
                        "tour_level": pick.get("tour_level"),
                        "gender": pick.get("gender"),
                        "surface": surface,
                        "odds": odds,
                        "result": result,
                        "flat_profit": round(
                            flat_profit(result, odds),
                            4,
                        ),
                        "original_stake": pick.get("stake"),
                        "original_profit": pick.get("profit"),
                        "ai_model_probability": pick.get(
                            "model_prob"
                        ),
                        "ai_implied_probability": pick.get(
                            "implied_prob"
                        ),
                        "elo_model_probability": round(
                            model_probability,
                            6,
                        ),
                        "elo_fair_odds": round(
                            1.0 / model_probability,
                            3,
                        ),
                        "elo_ev": round(ev, 6),
                        "pick_matches_before": int(
                            pick_state["matches_total"]
                        ),
                        "opponent_matches_before": int(
                            opponent_state["matches_total"]
                        ),
                        "elo_details": model,
                    }
                )
                counters["matched_ai_pick"] += 1

            by_date[match_date] = remaining

        # Update ELO only after pre-match evaluation.
        p1_state = players[p1_key]
        p2_state = players[p2_key]

        if winner_key == p1_key:
            winner_state = p1_state
            loser_state = p2_state
        else:
            winner_state = p2_state
            loser_state = p1_state

        winner_overall = float(
            winner_state["overall_elo"]
        )
        loser_overall = float(
            loser_state["overall_elo"]
        )

        (
            winner_state["overall_elo"],
            loser_state["overall_elo"],
        ) = update_pair(
            winner_overall,
            loser_overall,
            K_FACTOR,
        )

        if surface in KNOWN_SURFACES:
            winner_surface = get_surface_rating(
                winner_state,
                surface,
            )
            loser_surface = get_surface_rating(
                loser_state,
                surface,
            )

            (
                winner_state["surface_elo"][surface],
                loser_state["surface_elo"][surface],
            ) = update_pair(
                winner_surface,
                loser_surface,
                SURFACE_K_FACTOR,
            )

            p1_state["surface_matches"][surface] += 1
            p2_state["surface_matches"][surface] += 1

        p1_state["matches_total"] += 1
        p2_state["matches_total"] += 1

    unmatched = sum(
        len(rows)
        for rows in by_date.values()
    )
    counters["unmatched_ai_results"] = unmatched

    all_summary = summarize_bets(evaluated)
    grid = threshold_grid(evaluated)

    by_gender = {}
    for gender in sorted(
        {
            clean_str(row.get("gender")) or "unknown"
            for row in evaluated
        }
    ):
        subset = [
            row
            for row in evaluated
            if (clean_str(row.get("gender")) or "unknown")
            == gender
        ]
        by_gender[gender] = summarize_bets(subset)

    by_tour_level = {}
    for level in sorted(
        {
            clean_str(row.get("tour_level")) or "unknown"
            for row in evaluated
        }
    ):
        subset = [
            row
            for row in evaluated
            if (
                clean_str(row.get("tour_level"))
                or "unknown"
            ) == level
        ]
        by_tour_level[level] = summarize_bets(subset)

    by_surface = {}
    for surface in sorted(
        {
            clean_str(row.get("surface")) or "unknown"
            for row in evaluated
        }
    ):
        subset = [
            row
            for row in evaluated
            if (
                clean_str(row.get("surface"))
                or "unknown"
            ) == surface
        ]
        by_surface[surface] = summarize_bets(subset)

    output = {
        "generated_at": now_iso(),
        "model": "elo_on_ai_results_walk_forward_v1",
        "source_ai_results_url": AI_RESULTS_URL,
        "source_canonical_file": str(
            CANONICAL_MATCHES_FILE
        ),
        "settings": {
            "min_player_matches": MIN_PLAYER_MATCHES,
            "overall_weight": OVERALL_WEIGHT,
            "surface_weight": SURFACE_WEIGHT,
            "surface_confidence_matches": (
                SURFACE_CONFIDENCE_MATCHES
            ),
            "ev_thresholds": EV_THRESHOLDS,
            "min_model_probabilities": (
                MIN_MODEL_PROBABILITIES
            ),
            "min_odds_values": MIN_ODDS_VALUES,
        },
        "counts": {
            "canonical_matches": len(
                canonical_matches
            ),
            "ai_results_total": len(ai_results),
            "evaluated_ai_results": len(evaluated),
            **dict(counters),
        },
        "all_ai_results_summary": all_summary,
        "best_thresholds": grid[:25],
        "threshold_grid": grid,
        "by_gender": by_gender,
        "by_tour_level": by_tour_level,
        "by_surface": by_surface,
        "evaluated_results": evaluated,
    }

    report = {
        "generated_at": now_iso(),
        "model": output["model"],
        "settings": output["settings"],
        "counts": output["counts"],
        "all_ai_results_summary": all_summary,
        "best_thresholds": grid[:25],
        "by_gender": by_gender,
        "by_tour_level": by_tour_level,
        "by_surface": by_surface,
    }

    save_json(OUTPUT_FILE, output)
    save_json(REPORT_FILE, report)

    print("")
    print("ELO ON AI RESULTS BACKTEST DONE")
    print("COUNTS:", output["counts"])
    print(
        "ALL RESULTS:",
        output["all_ai_results_summary"],
    )

    print("\nBEST THRESHOLDS:")

    for row in output["best_thresholds"][:20]:
        print(row)

    print("\nBY GENDER:", by_gender)
    print("BY TOUR LEVEL:", by_tour_level)
    print("BY SURFACE:", by_surface)
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")
    print("")


if __name__ == "__main__":
    main()
