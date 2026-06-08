import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
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
MAX_DATE_SHIFT_DAYS = int(
    os.getenv("ELO_AI_RESULTS_MAX_DATE_SHIFT_DAYS", "1")
)
MIN_NAME_MATCH_SCORE = int(
    os.getenv("ELO_AI_RESULTS_MIN_NAME_MATCH_SCORE", "70")
)
MAX_DIAGNOSTIC_ROWS = int(
    os.getenv("ELO_AI_RESULTS_MAX_DIAGNOSTIC_ROWS", "250")
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
            "User-Agent": "Tennis-ELO AI-results backtest/2.0",
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


def surname_text(value: Any) -> str:
    _, surname = split_name(value)
    return "".join(surname)


def initial_text(value: Any) -> str:
    given, _ = split_name(value)
    return given[0][:1] if given else ""


def name_match_score(query: Any, candidate: Any) -> int:
    query_clean = compact(query)
    candidate_clean = compact(candidate)

    if not query_clean or not candidate_clean:
        return 0

    if query_clean == candidate_clean:
        return 100

    q_surname = surname_text(query)
    c_surname = surname_text(candidate)
    q_initial = initial_text(query)
    c_initial = initial_text(candidate)

    if q_surname and q_surname == c_surname:
        if q_initial and c_initial and q_initial == c_initial:
            return 85
        return 65

    if (
        q_surname
        and c_surname
        and (
            q_surname in c_surname
            or c_surname in q_surname
        )
    ):
        if q_initial and c_initial and q_initial == c_initial:
            return 75
        return 50

    return 0


def tournament_match_score(query: Any, candidate: Any) -> int:
    query_clean = compact(query)
    candidate_clean = compact(candidate)

    if not query_clean or not candidate_clean:
        return 0

    if query_clean == candidate_clean:
        return 15

    if (
        query_clean in candidate_clean
        or candidate_clean in query_clean
    ):
        return 8

    return 0


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


def player_key(value: Any) -> str:
    text = compact(value)
    return text


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
        "effective_overall_weight": effective_overall_weight,
        "effective_surface_weight": effective_surface_weight,
    }


def valid_canonical_match(row: dict[str, Any]) -> bool:
    match_date = parse_date(row.get("date"))
    player_1 = clean_str(row.get("player_1"))
    player_2 = clean_str(row.get("player_2"))
    winner = clean_str(row.get("winner"))

    if not match_date or not player_1 or not player_2 or not winner:
        return False

    p1_key = player_key(player_1)
    p2_key = player_key(player_2)
    winner_key = player_key(winner)

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
    result = clean_str(row.get("result")).lower()

    if result not in SETTLED_RESULTS:
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
    return odds - 1.0 if result == "win" else -1.0


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

                output.append(
                    {
                        "min_ev": min_ev,
                        "min_model_probability": min_probability,
                        "min_odds": min_odds,
                        **summarize_bets(subset),
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


def build_pre_match_snapshots(
    canonical_matches: list[dict[str, Any]],
) -> tuple[
    dict[Any, list[dict[str, Any]]],
    dict[str, Any],
]:
    players: dict[str, dict[str, Any]] = defaultdict(
        new_player_state
    )
    snapshots_by_date: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    counters: dict[str, int] = defaultdict(int)

    for row in canonical_matches:
        match_date = parse_date(row["date"])
        player_1 = clean_str(row["player_1"])
        player_2 = clean_str(row["player_2"])
        winner = clean_str(row["winner"])
        p1_key = player_key(player_1)
        p2_key = player_key(player_2)
        winner_key = player_key(winner)
        surface = normalized_surface(row.get("surface"))

        p1_state = players[p1_key]
        p2_state = players[p2_key]
        model = blended_probability(
            p1_state,
            p2_state,
            surface,
        )

        snapshots_by_date[match_date].append(
            {
                "date": match_date,
                "player_1": player_1,
                "player_2": player_2,
                "player_1_key": p1_key,
                "player_2_key": p2_key,
                "winner": winner,
                "surface": surface,
                "tournament": (
                    row.get("tournament")
                    or row.get("tournament_code")
                    or ""
                ),
                "tour_level": row.get("tour_level"),
                "gender": row.get("gender"),
                "canonical_match_id": row.get(
                    "canonical_match_id"
                ),
                "player_1_matches_before": int(
                    p1_state["matches_total"]
                ),
                "player_2_matches_before": int(
                    p2_state["matches_total"]
                ),
                "model": model,
            }
        )

        if winner_key == p1_key:
            winner_state = p1_state
            loser_state = p2_state
        else:
            winner_state = p2_state
            loser_state = p1_state

        (
            winner_state["overall_elo"],
            loser_state["overall_elo"],
        ) = update_pair(
            float(winner_state["overall_elo"]),
            float(loser_state["overall_elo"]),
            K_FACTOR,
        )

        if surface in KNOWN_SURFACES:
            (
                winner_state["surface_elo"][surface],
                loser_state["surface_elo"][surface],
            ) = update_pair(
                get_surface_rating(
                    winner_state,
                    surface,
                ),
                get_surface_rating(
                    loser_state,
                    surface,
                ),
                SURFACE_K_FACTOR,
            )

            p1_state["surface_matches"][surface] += 1
            p2_state["surface_matches"][surface] += 1
        else:
            counters["unknown_surface_matches"] += 1

        p1_state["matches_total"] += 1
        p2_state["matches_total"] += 1

    return dict(snapshots_by_date), {
        "players_final": len(players),
        **dict(counters),
    }


def candidate_score(
    ai_row: dict[str, Any],
    snapshot: dict[str, Any],
    date_shift: int,
) -> tuple[int, str, int, int]:
    pick_player = clean_str(ai_row.get("player_name"))
    opponent = clean_str(ai_row.get("opponent_name"))

    direct_pick_score = name_match_score(
        pick_player,
        snapshot["player_1"],
    )
    direct_opponent_score = name_match_score(
        opponent,
        snapshot["player_2"],
    )
    reverse_pick_score = name_match_score(
        pick_player,
        snapshot["player_2"],
    )
    reverse_opponent_score = name_match_score(
        opponent,
        snapshot["player_1"],
    )

    direct_total = direct_pick_score + direct_opponent_score
    reverse_total = reverse_pick_score + reverse_opponent_score

    if direct_total >= reverse_total:
        orientation = "player_1"
        pick_score = direct_pick_score
        opponent_score = direct_opponent_score
        total = direct_total
    else:
        orientation = "player_2"
        pick_score = reverse_pick_score
        opponent_score = reverse_opponent_score
        total = reverse_total

    total += 20 if date_shift == 0 else 5
    total += tournament_match_score(
        ai_row.get("tournament"),
        snapshot.get("tournament"),
    )

    return total, orientation, pick_score, opponent_score


def find_snapshot(
    ai_row: dict[str, Any],
    snapshots_by_date: dict[Any, list[dict[str, Any]]],
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any],
]:
    ai_date = parse_date(ai_row.get("date"))

    if not ai_date:
        return None, {
            "reason": "invalid_ai_date",
        }

    candidates = []

    for shift in range(
        -MAX_DATE_SHIFT_DAYS,
        MAX_DATE_SHIFT_DAYS + 1,
    ):
        candidate_date = ai_date + timedelta(days=shift)

        for snapshot in snapshots_by_date.get(
            candidate_date,
            [],
        ):
            (
                score,
                orientation,
                pick_score,
                opponent_score,
            ) = candidate_score(
                ai_row,
                snapshot,
                abs(shift),
            )

            if (
                pick_score >= MIN_NAME_MATCH_SCORE
                and opponent_score >= MIN_NAME_MATCH_SCORE
            ):
                candidates.append(
                    {
                        "score": score,
                        "orientation": orientation,
                        "pick_name_score": pick_score,
                        "opponent_name_score": opponent_score,
                        "date_shift_days": shift,
                        "snapshot": snapshot,
                    }
                )

    if not candidates:
        same_date_rows = snapshots_by_date.get(ai_date, [])
        return None, {
            "reason": (
                "no_pair_match_within_date_window"
                if same_date_rows
                else "no_canonical_matches_near_date"
            ),
            "same_date_candidates": len(same_date_rows),
        }

    candidates.sort(
        key=lambda item: (
            item["score"],
            -abs(item["date_shift_days"]),
        ),
        reverse=True,
    )

    best = candidates[0]

    if (
        len(candidates) > 1
        and candidates[1]["score"] == best["score"]
        and candidates[1]["snapshot"].get(
            "canonical_match_id"
        ) != best["snapshot"].get(
            "canonical_match_id"
        )
    ):
        return None, {
            "reason": "ambiguous_pair_match",
            "best_score": best["score"],
            "candidate_count": len(candidates),
        }

    return best["snapshot"], {
        "reason": "matched",
        "score": best["score"],
        "orientation": best["orientation"],
        "pick_name_score": best["pick_name_score"],
        "opponent_name_score": best["opponent_name_score"],
        "date_shift_days": best["date_shift_days"],
        "candidate_count": len(candidates),
    }


def grouped_summary(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    output = {}

    for value in sorted(
        {
            clean_str(row.get(field)) or "unknown"
            for row in rows
        }
    ):
        subset = [
            row
            for row in rows
            if (
                clean_str(row.get(field))
                or "unknown"
            ) == value
        ]
        output[value] = summarize_bets(subset)

    return output


def main() -> None:
    canonical_payload = load_json(
        CANONICAL_MATCHES_FILE,
        {},
    )
    raw_canonical = (
        canonical_payload.get("matches", [])
        if isinstance(canonical_payload, dict)
        else []
    )

    if not isinstance(raw_canonical, list):
        raw_canonical = []

    canonical_matches = [
        row
        for row in raw_canonical
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

    snapshots_by_date, snapshot_counts = (
        build_pre_match_snapshots(
            canonical_matches
        )
    )

    ai_payload = fetch_json(AI_RESULTS_URL)
    raw_ai_results = (
        ai_payload
        if isinstance(ai_payload, list)
        else ai_payload.get("results", [])
        if isinstance(ai_payload, dict)
        else []
    )

    if not isinstance(raw_ai_results, list):
        raw_ai_results = []

    ai_results = [
        row
        for row in raw_ai_results
        if isinstance(row, dict)
        and valid_ai_result(row)
    ]

    evaluated: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    counters: dict[str, int] = defaultdict(int)

    canonical_dates = sorted(snapshots_by_date)
    canonical_min_date = (
        canonical_dates[0]
        if canonical_dates
        else None
    )
    canonical_max_date = (
        canonical_dates[-1]
        if canonical_dates
        else None
    )

    for ai_row in ai_results:
        snapshot, match_info = find_snapshot(
            ai_row,
            snapshots_by_date,
        )

        if snapshot is None:
            reason = match_info["reason"]
            counters[reason] += 1

            if len(diagnostics) < MAX_DIAGNOSTIC_ROWS:
                diagnostics.append(
                    {
                        "status": "unmatched",
                        "reason": reason,
                        "date": ai_row.get("date"),
                        "event_key": ai_row.get("event_key"),
                        "player_name": ai_row.get("player_name"),
                        "opponent_name": ai_row.get("opponent_name"),
                        "tournament": ai_row.get("tournament"),
                        "details": match_info,
                    }
                )
            continue

        orientation = match_info["orientation"]

        if orientation == "player_1":
            pick_matches_before = snapshot[
                "player_1_matches_before"
            ]
            opponent_matches_before = snapshot[
                "player_2_matches_before"
            ]
            model_probability = snapshot[
                "model"
            ]["blended_p1"]
        else:
            pick_matches_before = snapshot[
                "player_2_matches_before"
            ]
            opponent_matches_before = snapshot[
                "player_1_matches_before"
            ]
            model_probability = (
                1.0
                - snapshot["model"]["blended_p1"]
            )

        if (
            pick_matches_before < MIN_PLAYER_MATCHES
            or opponent_matches_before < MIN_PLAYER_MATCHES
        ):
            counters[
                "insufficient_player_history"
            ] += 1

            if len(diagnostics) < MAX_DIAGNOSTIC_ROWS:
                diagnostics.append(
                    {
                        "status": "matched_but_ineligible",
                        "reason": "insufficient_player_history",
                        "date": ai_row.get("date"),
                        "event_key": ai_row.get("event_key"),
                        "player_name": ai_row.get("player_name"),
                        "opponent_name": ai_row.get("opponent_name"),
                        "canonical_match_id": snapshot.get(
                            "canonical_match_id"
                        ),
                        "pick_matches_before": pick_matches_before,
                        "opponent_matches_before": opponent_matches_before,
                        "join": match_info,
                    }
                )
            continue

        odds = float(ai_row["odds"])
        result = clean_str(
            ai_row["result"]
        ).lower()
        ev = model_probability * odds - 1.0

        evaluated.append(
            {
                "pick_id": ai_row.get("pick_id"),
                "event_key": ai_row.get("event_key"),
                "date": parse_date(
                    ai_row["date"]
                ).isoformat(),
                "canonical_date": snapshot[
                    "date"
                ].isoformat(),
                "date_shift_days": match_info[
                    "date_shift_days"
                ],
                "canonical_match_id": snapshot.get(
                    "canonical_match_id"
                ),
                "match": ai_row.get("match"),
                "pick_player": clean_str(
                    ai_row.get("player_name")
                ),
                "opponent": clean_str(
                    ai_row.get("opponent_name")
                ),
                "canonical_player_1": snapshot[
                    "player_1"
                ],
                "canonical_player_2": snapshot[
                    "player_2"
                ],
                "orientation": orientation,
                "tournament": (
                    ai_row.get("tournament")
                    or snapshot.get("tournament")
                ),
                "tour_level": (
                    ai_row.get("tour_level")
                    or snapshot.get("tour_level")
                ),
                "gender": (
                    ai_row.get("gender")
                    or snapshot.get("gender")
                ),
                "surface": snapshot["surface"],
                "odds": odds,
                "result": result,
                "flat_profit": round(
                    flat_profit(result, odds),
                    4,
                ),
                "original_stake": ai_row.get("stake"),
                "original_profit": ai_row.get("profit"),
                "ai_model_probability": ai_row.get(
                    "model_prob"
                ),
                "ai_implied_probability": ai_row.get(
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
                "pick_matches_before": pick_matches_before,
                "opponent_matches_before": opponent_matches_before,
                "join_score": match_info["score"],
                "pick_name_score": match_info[
                    "pick_name_score"
                ],
                "opponent_name_score": match_info[
                    "opponent_name_score"
                ],
                "elo_details": snapshot["model"],
            }
        )
        counters["matched_ai_pick"] += 1

    grid = threshold_grid(evaluated)

    unmatched_count = sum(
        value
        for key, value in counters.items()
        if key in {
            "invalid_ai_date",
            "no_pair_match_within_date_window",
            "no_canonical_matches_near_date",
            "ambiguous_pair_match",
        }
    )

    output = {
        "generated_at": now_iso(),
        "model": "elo_on_ai_results_walk_forward_v2",
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
            "max_date_shift_days": MAX_DATE_SHIFT_DAYS,
            "min_name_match_score": MIN_NAME_MATCH_SCORE,
            "ev_thresholds": EV_THRESHOLDS,
            "min_model_probabilities": (
                MIN_MODEL_PROBABILITIES
            ),
            "min_odds_values": MIN_ODDS_VALUES,
        },
        "coverage": {
            "canonical_min_date": (
                canonical_min_date.isoformat()
                if canonical_min_date
                else None
            ),
            "canonical_max_date": (
                canonical_max_date.isoformat()
                if canonical_max_date
                else None
            ),
            "ai_min_date": (
                min(
                    parse_date(row["date"])
                    for row in ai_results
                ).isoformat()
                if ai_results
                else None
            ),
            "ai_max_date": (
                max(
                    parse_date(row["date"])
                    for row in ai_results
                ).isoformat()
                if ai_results
                else None
            ),
        },
        "counts": {
            "canonical_matches_raw": len(
                raw_canonical
            ),
            "canonical_matches_valid": len(
                canonical_matches
            ),
            "ai_results_raw": len(raw_ai_results),
            "ai_results_valid": len(ai_results),
            "evaluated_ai_results": len(evaluated),
            "unmatched_ai_results": unmatched_count,
            **snapshot_counts,
            **dict(counters),
        },
        "all_ai_results_summary": summarize_bets(
            evaluated
        ),
        "best_thresholds": grid[:25],
        "threshold_grid": grid,
        "by_gender": grouped_summary(
            evaluated,
            "gender",
        ),
        "by_tour_level": grouped_summary(
            evaluated,
            "tour_level",
        ),
        "by_surface": grouped_summary(
            evaluated,
            "surface",
        ),
        "join_diagnostics": {
            "reason_counts": dict(
                Counter(
                    row["reason"]
                    for row in diagnostics
                )
            ),
            "sample_rows": diagnostics,
        },
        "evaluated_results": evaluated,
    }

    report = {
        "generated_at": now_iso(),
        "model": output["model"],
        "settings": output["settings"],
        "coverage": output["coverage"],
        "counts": output["counts"],
        "all_ai_results_summary": output[
            "all_ai_results_summary"
        ],
        "best_thresholds": output[
            "best_thresholds"
        ],
        "by_gender": output["by_gender"],
        "by_tour_level": output[
            "by_tour_level"
        ],
        "by_surface": output["by_surface"],
        "join_diagnostics": output[
            "join_diagnostics"
        ],
    }

    save_json(OUTPUT_FILE, output)
    save_json(REPORT_FILE, report)

    print("")
    print("ELO ON AI RESULTS BACKTEST V2 DONE")
    print("COVERAGE:", output["coverage"])
    print("COUNTS:", output["counts"])
    print(
        "ALL RESULTS:",
        output["all_ai_results_summary"],
    )

    print("\nJOIN REASONS:")

    for reason, count in output[
        "join_diagnostics"
    ]["reason_counts"].items():
        print(reason, count)

    print("\nBEST THRESHOLDS:")

    for row in output["best_thresholds"][:20]:
        print(row)

    print("\nBY GENDER:", output["by_gender"])
    print(
        "BY TOUR LEVEL:",
        output["by_tour_level"],
    )
    print("BY SURFACE:", output["by_surface"])
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")
    print("")


if __name__ == "__main__":
    main()
