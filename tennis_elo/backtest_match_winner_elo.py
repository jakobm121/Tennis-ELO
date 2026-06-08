import math
import os
from collections import defaultdict
from datetime import datetime
from statistics import mean
from typing import Any

from tennis_elo.config import (
    CANONICAL_MATCHES_FILE,
    DEFAULT_ELO,
    K_FACTOR,
    ROOT_DIR,
    SURFACE_K_FACTOR,
)
from tennis_elo.utils import (
    canonical_player_name,
    clean_str,
    load_json,
    now_iso,
    save_json,
)


OUTPUT_FILE = (
    ROOT_DIR
    / "data"
    / "backtests"
    / "match_winner_elo_backtest.json"
)
REPORT_FILE = (
    ROOT_DIR
    / "data"
    / "reports"
    / "match_winner_elo_backtest_report.json"
)

TEST_START = os.getenv(
    "MATCH_WINNER_ELO_TEST_START",
    "2026-03-01",
)
MIN_PLAYER_MATCHES = int(
    os.getenv("MATCH_WINNER_ELO_MIN_PLAYER_MATCHES", "10")
)
BLEND_OVERALL_WEIGHT = float(
    os.getenv("MATCH_WINNER_ELO_BLEND_OVERALL_WEIGHT", "0.60")
)
BLEND_SURFACE_WEIGHT = float(
    os.getenv("MATCH_WINNER_ELO_BLEND_SURFACE_WEIGHT", "0.40")
)
SURFACE_CONFIDENCE_MATCHES = int(
    os.getenv("MATCH_WINNER_ELO_SURFACE_CONFIDENCE_MATCHES", "20")
)

KNOWN_SURFACES = {"hard", "clay", "grass", "carpet"}
MODEL_NAMES = (
    "overall_elo",
    "surface_elo",
    "blended_elo",
)


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


def expected_score(a: float, b: float) -> float:
    return 1.0 / (
        1.0 + math.pow(10.0, (b - a) / 400.0)
    )


def update_pair(
    winner: float,
    loser: float,
    k: float,
) -> tuple[float, float]:
    probability = expected_score(winner, loser)

    return (
        winner + k * (1.0 - probability),
        loser - k * (1.0 - probability),
    )


def clip_probability(value: float) -> float:
    return min(0.999999, max(0.000001, value))


def accuracy(
    probabilities: list[float],
    outcomes: list[int],
) -> float | None:
    if not probabilities:
        return None

    correct = sum(
        (probability >= 0.5) == bool(outcome)
        for probability, outcome
        in zip(probabilities, outcomes)
    )

    return round(correct / len(probabilities), 4)


def brier_score(
    probabilities: list[float],
    outcomes: list[int],
) -> float | None:
    if not probabilities:
        return None

    return round(
        mean(
            (probability - outcome) ** 2
            for probability, outcome
            in zip(probabilities, outcomes)
        ),
        6,
    )


def log_loss(
    probabilities: list[float],
    outcomes: list[int],
) -> float | None:
    if not probabilities:
        return None

    values = []

    for probability, outcome in zip(
        probabilities,
        outcomes,
    ):
        probability = clip_probability(probability)

        values.append(
            -(
                outcome * math.log(probability)
                + (1 - outcome)
                * math.log(1 - probability)
            )
        )

    return round(mean(values), 6)


def calibration(
    probabilities: list[float],
    outcomes: list[int],
) -> list[dict[str, Any]]:
    rows = []

    for index in range(10):
        low = index / 10
        high = low + 0.1

        bucket = [
            (probability, outcome)
            for probability, outcome
            in zip(probabilities, outcomes)
            if (
                low <= probability < high
                or (
                    high >= 1.0
                    and probability == 1.0
                )
            )
        ]

        if not bucket:
            continue

        rows.append(
            {
                "bucket": f"{low:.1f}-{high:.1f}",
                "sample_size": len(bucket),
                "avg_predicted": round(
                    mean(item[0] for item in bucket),
                    4,
                ),
                "actual_win_rate": round(
                    mean(item[1] for item in bucket),
                    4,
                ),
            }
        )

    return rows


def summarize(
    rows: list[dict[str, Any]],
    model_name: str,
) -> dict[str, Any]:
    key = f"{model_name}_p1_probability"
    probabilities = [
        float(row[key])
        for row in rows
    ]
    outcomes = [
        int(row["player_1_won"])
        for row in rows
    ]

    return {
        "sample_size": len(rows),
        "accuracy": accuracy(probabilities, outcomes),
        "brier_score": brier_score(
            probabilities,
            outcomes,
        ),
        "log_loss": log_loss(
            probabilities,
            outcomes,
        ),
        "avg_predicted_p1": (
            round(mean(probabilities), 4)
            if probabilities
            else None
        ),
        "actual_p1_win_rate": (
            round(mean(outcomes), 4)
            if outcomes
            else None
        ),
        "calibration": calibration(
            probabilities,
            outcomes,
        ),
    }


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


def normalized_gender(row: dict[str, Any]) -> str:
    value = clean_str(
        row.get("gender")
        or row.get("sex")
        or row.get("tour_gender")
    ).lower()

    if value in {"m", "male", "men", "atp"}:
        return "men"

    if value in {"f", "female", "women", "wta"}:
        return "women"

    source = " ".join(
        clean_str(row.get(key)).lower()
        for key in (
            "tour",
            "tour_level",
            "tournament",
            "tournament_code",
            "source",
        )
    )

    if "wta" in source or "women" in source:
        return "women"

    if "atp" in source or "men" in source:
        return "men"

    return "unknown"


def normalized_tour_level(
    row: dict[str, Any],
) -> str:
    value = clean_str(
        row.get("tour_level")
        or row.get("level")
        or row.get("series")
    ).lower()

    if value:
        return value

    source = " ".join(
        clean_str(row.get(key)).lower()
        for key in (
            "tournament",
            "tournament_code",
            "source",
        )
    )

    if "challenger" in source:
        return "challenger"
    if "itf" in source:
        return "itf"
    if "wta" in source:
        return "wta"
    if "atp" in source:
        return "atp"

    return "unknown"


def valid_match(
    row: dict[str, Any],
    counters: dict[str, int],
) -> bool:
    if not parse_date(row.get("date")):
        counters["invalid_date"] += 1
        return False

    # V2 deliberately requires the original sides. No winner/loser fallback.
    player_1 = clean_str(row.get("player_1"))
    player_2 = clean_str(row.get("player_2"))

    if not player_1 or not player_2:
        counters["missing_original_players"] += 1
        return False

    p1_key = canonical_player_name(player_1)
    p2_key = canonical_player_name(player_2)

    if not p1_key or not p2_key or p1_key == p2_key:
        counters["invalid_player_pair"] += 1
        return False

    winner = clean_str(row.get("winner"))
    winner_key = canonical_player_name(winner)

    if not winner_key:
        counters["missing_winner"] += 1
        return False

    if winner_key not in {p1_key, p2_key}:
        counters["winner_not_one_of_players"] += 1
        return False

    if row.get("ready_for_elo") is False:
        counters["not_ready_for_elo"] += 1
        return False

    if row.get("retired") or row.get("walkover"):
        counters["retired_or_walkover"] += 1
        return False

    return True


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
    overall_probability: float,
    surface_probability: float,
    surface: str,
    p1_surface_matches: int,
    p2_surface_matches: int,
) -> float:
    # Unknown surface must never create or influence a surface-specific model.
    if surface not in KNOWN_SURFACES:
        return overall_probability

    confidence = min(
        1.0,
        min(
            p1_surface_matches,
            p2_surface_matches,
        )
        / max(1, SURFACE_CONFIDENCE_MATCHES),
    )

    surface_weight = (
        BLEND_SURFACE_WEIGHT * confidence
    )
    overall_weight = (
        BLEND_OVERALL_WEIGHT
        + BLEND_SURFACE_WEIGHT
        * (1.0 - confidence)
    )
    total = overall_weight + surface_weight

    return (
        overall_probability * overall_weight
        + surface_probability * surface_weight
    ) / total


def favorite_bucket(probability: float) -> str:
    value = max(probability, 1.0 - probability)

    if value < 0.55:
        return "0.50-0.55"
    if value < 0.60:
        return "0.55-0.60"
    if value < 0.65:
        return "0.60-0.65"
    if value < 0.70:
        return "0.65-0.70"
    if value < 0.75:
        return "0.70-0.75"
    if value < 0.80:
        return "0.75-0.80"
    if value < 0.85:
        return "0.80-0.85"

    return "0.85+"


def grouped_summary(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    result = {}

    for value in sorted(
        {clean_str(row.get(field)) or "unknown" for row in rows}
    ):
        subset = [
            row
            for row in rows
            if (clean_str(row.get(field)) or "unknown")
            == value
        ]

        result[value] = {
            model_name: summarize(
                subset,
                model_name,
            )
            for model_name in MODEL_NAMES
        }

    return result


def main() -> None:
    payload = load_json(
        CANONICAL_MATCHES_FILE,
        {},
    )
    raw_matches = (
        payload.get("matches", [])
        if isinstance(payload, dict)
        else []
    )

    if not isinstance(raw_matches, list):
        raw_matches = []

    counters: dict[str, int] = defaultdict(int)
    matches = []

    for row in raw_matches:
        if (
            isinstance(row, dict)
            and valid_match(row, counters)
        ):
            matches.append(row)

    matches.sort(
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

    test_start = parse_date(TEST_START)

    if not test_start:
        raise SystemExit(
            f"Invalid MATCH_WINNER_ELO_TEST_START: "
            f"{TEST_START}"
        )

    players: dict[str, dict[str, Any]] = defaultdict(
        new_player_state
    )
    predictions: list[dict[str, Any]] = []

    for row in matches:
        match_date = parse_date(row.get("date"))
        player_1 = clean_str(row.get("player_1"))
        player_2 = clean_str(row.get("player_2"))
        winner = clean_str(row.get("winner"))

        p1_key = canonical_player_name(player_1)
        p2_key = canonical_player_name(player_2)
        winner_key = canonical_player_name(winner)

        surface = normalized_surface(
            row.get("surface")
        )
        gender = normalized_gender(row)
        tour_level = normalized_tour_level(row)

        p1 = players[p1_key]
        p2 = players[p2_key]

        p1_overall = float(p1["overall_elo"])
        p2_overall = float(p2["overall_elo"])

        overall_probability = expected_score(
            p1_overall,
            p2_overall,
        )

        p1_surface_matches = int(
            p1["surface_matches"][surface]
        )
        p2_surface_matches = int(
            p2["surface_matches"][surface]
        )

        if surface in KNOWN_SURFACES:
            p1_surface = get_surface_rating(
                p1,
                surface,
            )
            p2_surface = get_surface_rating(
                p2,
                surface,
            )
            surface_probability = expected_score(
                p1_surface,
                p2_surface,
            )
        else:
            # No fake "unknown surface ELO".
            p1_surface = p1_overall
            p2_surface = p2_overall
            surface_probability = overall_probability

        blend_probability = blended_probability(
            overall_probability,
            surface_probability,
            surface,
            p1_surface_matches,
            p2_surface_matches,
        )

        eligible = (
            match_date >= test_start
            and int(p1["matches_total"])
            >= MIN_PLAYER_MATCHES
            and int(p2["matches_total"])
            >= MIN_PLAYER_MATCHES
        )

        if eligible:
            player_1_won = int(
                winner_key == p1_key
            )

            predictions.append(
                {
                    "date": match_date.isoformat(),
                    "canonical_match_id": row.get(
                        "canonical_match_id"
                    ),
                    "tournament": (
                        row.get("tournament")
                        or row.get("tournament_code")
                        or ""
                    ),
                    "gender": gender,
                    "tour_level": tour_level,
                    "surface": surface,
                    "player_1": player_1,
                    "player_2": player_2,
                    "winner": winner,
                    "player_1_won": player_1_won,
                    "player_1_matches_before": int(
                        p1["matches_total"]
                    ),
                    "player_2_matches_before": int(
                        p2["matches_total"]
                    ),
                    "player_1_surface_matches_before": (
                        p1_surface_matches
                    ),
                    "player_2_surface_matches_before": (
                        p2_surface_matches
                    ),
                    "player_1_overall_elo_before": round(
                        p1_overall,
                        3,
                    ),
                    "player_2_overall_elo_before": round(
                        p2_overall,
                        3,
                    ),
                    "player_1_surface_elo_before": round(
                        p1_surface,
                        3,
                    ),
                    "player_2_surface_elo_before": round(
                        p2_surface,
                        3,
                    ),
                    "overall_elo_p1_probability": round(
                        overall_probability,
                        6,
                    ),
                    "surface_elo_p1_probability": round(
                        surface_probability,
                        6,
                    ),
                    "blended_elo_p1_probability": round(
                        blend_probability,
                        6,
                    ),
                    "overall_favorite_bucket": (
                        favorite_bucket(
                            overall_probability
                        )
                    ),
                    "surface_favorite_bucket": (
                        favorite_bucket(
                            surface_probability
                        )
                    ),
                    "blended_favorite_bucket": (
                        favorite_bucket(
                            blend_probability
                        )
                    ),
                }
            )
        elif match_date >= test_start:
            counters[
                "insufficient_player_history"
            ] += 1

        if winner_key == p1_key:
            winner_state = p1
            loser_state = p2
            winner_overall = p1_overall
            loser_overall = p2_overall
            winner_surface = p1_surface
            loser_surface = p2_surface
        else:
            winner_state = p2
            loser_state = p1
            winner_overall = p2_overall
            loser_overall = p1_overall
            winner_surface = p2_surface
            loser_surface = p1_surface

        (
            winner_state["overall_elo"],
            loser_state["overall_elo"],
        ) = update_pair(
            winner_overall,
            loser_overall,
            K_FACTOR,
        )

        # Only real surfaces update a surface-specific rating.
        if surface in KNOWN_SURFACES:
            (
                winner_state["surface_elo"][surface],
                loser_state["surface_elo"][surface],
            ) = update_pair(
                winner_surface,
                loser_surface,
                SURFACE_K_FACTOR,
            )

            p1["surface_matches"][surface] += 1
            p2["surface_matches"][surface] += 1
        else:
            counters["unknown_surface_matches"] += 1

        p1["matches_total"] += 1
        p2["matches_total"] += 1

    summaries = {
        model_name: summarize(
            predictions,
            model_name,
        )
        for model_name in MODEL_NAMES
    }

    by_surface = grouped_summary(
        predictions,
        "surface",
    )
    by_gender = grouped_summary(
        predictions,
        "gender",
    )
    by_tour_level = grouped_summary(
        predictions,
        "tour_level",
    )

    by_bucket = {}

    for model_name in MODEL_NAMES:
        prefix = model_name.replace("_elo", "")
        key = f"{prefix}_favorite_bucket"
        by_bucket[model_name] = {}

        for bucket in (
            "0.50-0.55",
            "0.55-0.60",
            "0.60-0.65",
            "0.65-0.70",
            "0.70-0.75",
            "0.75-0.80",
            "0.80-0.85",
            "0.85+",
        ):
            subset = [
                row
                for row in predictions
                if row.get(key) == bucket
            ]
            by_bucket[model_name][bucket] = (
                summarize(
                    subset,
                    model_name,
                )
            )

    ranking = sorted(
        MODEL_NAMES,
        key=lambda name: (
            summaries[name]["brier_score"]
            if summaries[name]["brier_score"]
            is not None
            else 999,
            summaries[name]["log_loss"]
            if summaries[name]["log_loss"]
            is not None
            else 999,
            -(
                summaries[name]["accuracy"]
                if summaries[name]["accuracy"]
                is not None
                else 0
            ),
        ),
    )

    output = {
        "generated_at": now_iso(),
        "model": "match_winner_elo_walk_forward_v2",
        "source_file": str(
            CANONICAL_MATCHES_FILE
        ),
        "settings": {
            "test_start": test_start.isoformat(),
            "default_elo": DEFAULT_ELO,
            "overall_k_factor": K_FACTOR,
            "surface_k_factor": (
                SURFACE_K_FACTOR
            ),
            "min_player_matches": (
                MIN_PLAYER_MATCHES
            ),
            "blend_overall_weight": (
                BLEND_OVERALL_WEIGHT
            ),
            "blend_surface_weight": (
                BLEND_SURFACE_WEIGHT
            ),
            "surface_confidence_matches": (
                SURFACE_CONFIDENCE_MATCHES
            ),
            "known_surfaces": sorted(
                KNOWN_SURFACES
            ),
            "strict_original_players": True,
            "unknown_surface_uses_overall_only": True,
        },
        "counts": {
            "raw_input_matches": len(raw_matches),
            "valid_input_matches": len(matches),
            "test_predictions": len(predictions),
            "players_final": len(players),
            **dict(counters),
        },
        "best_model": ranking[0] if ranking else None,
        "model_ranking": ranking,
        "summary": summaries,
        "by_surface": by_surface,
        "by_gender": by_gender,
        "by_tour_level": by_tour_level,
        "by_favorite_probability_bucket": (
            by_bucket
        ),
        "predictions": predictions,
    }

    report = {
        "generated_at": now_iso(),
        "model": output["model"],
        "settings": output["settings"],
        "counts": output["counts"],
        "best_model": output["best_model"],
        "model_ranking": ranking,
        "summary": summaries,
        "by_surface": by_surface,
        "by_gender": by_gender,
        "by_tour_level": by_tour_level,
        "by_favorite_probability_bucket": (
            by_bucket
        ),
    }

    save_json(OUTPUT_FILE, output)
    save_json(REPORT_FILE, report)

    print("")
    print("MATCH WINNER ELO BACKTEST V2 DONE")
    print("COUNTS:", output["counts"])
    print("BEST MODEL:", output["best_model"])
    print("MODEL RANKING:", ranking)

    for model_name in MODEL_NAMES:
        print(
            model_name.upper() + ":",
            summaries[model_name],
        )

    print("BY SURFACE:", by_surface)
    print("BY GENDER:", by_gender)
    print("BY TOUR LEVEL:", by_tour_level)
    print(f"Output: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")
    print("")


if __name__ == "__main__":
    main()
