import gzip
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tennis_elo.config import ROOT_DIR
from tennis_elo.utils import now_iso, save_json


SOURCE_MANIFEST = (
    ROOT_DIR
    / "data"
    / "tle"
    / "processed"
    / "sackmann"
    / "tle_sackmann_manifest.json"
)

OUTPUT_FILE = (
    ROOT_DIR
    / "data"
    / "tle"
    / "backtests"
    / "tle_model_search_v3.json"
)

REPORT_FILE = (
    ROOT_DIR
    / "data"
    / "tle"
    / "reports"
    / "tle_model_search_v3_report.json"
)

DEFAULT_ELO = float(os.getenv("TLE_DEFAULT_ELO", "1500"))
GLOBAL_K = float(os.getenv("TLE_GLOBAL_K", "24"))
GLOBAL_SURFACE_K = float(os.getenv("TLE_GLOBAL_SURFACE_K", "20"))
LEVEL_K = float(os.getenv("TLE_LEVEL_K", "24"))
LEVEL_SURFACE_K = float(os.getenv("TLE_LEVEL_SURFACE_K", "20"))
TEST_START_DATE = os.getenv("TLE_TEST_START_DATE", "2025-01-01")

LEVEL_SAMPLE_GRID = [
    int(value)
    for value in os.getenv(
        "TLE_LEVEL_SAMPLE_GRID",
        "5,10,15,20",
    ).split(",")
    if value.strip()
]
SURFACE_SAMPLE_GRID = [
    int(value)
    for value in os.getenv(
        "TLE_SURFACE_SAMPLE_GRID",
        "3,5,10",
    ).split(",")
    if value.strip()
]
SURFACE_WEIGHT_GRID = [
    float(value)
    for value in os.getenv(
        "TLE_SURFACE_WEIGHT_GRID",
        "0,0.1,0.15,0.2,0.25,0.3,0.4",
    ).split(",")
    if value.strip()
]
GLOBAL_WEIGHT_GRID = [
    float(value)
    for value in os.getenv(
        "TLE_GLOBAL_WEIGHT_GRID",
        "0,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    ).split(",")
    if value.strip()
]

VALID_SURFACES = {"hard", "clay", "grass", "carpet"}
TARGET_LEVELS = {"main_tour", "challenger", "itf", "qualifying"}


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def player_key(player: dict[str, Any], gender: str) -> str:
    player_id = player.get("sackmann_player_id")
    name = clean(player.get("name"))

    if player_id not in (None, ""):
        return f"{gender}:sackmann:{int(player_id)}"

    normalized = re.sub(
        r"[^a-z0-9]+",
        " ",
        name.lower(),
    ).strip()

    return f"{gender}:name:{normalized}"


def expected_score(a: float, b: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (b - a) / 400.0))


def update_pair(
    winner_rating: float,
    loser_rating: float,
    k_factor: float,
) -> tuple[float, float]:
    p = expected_score(winner_rating, loser_rating)
    change = k_factor * (1.0 - p)
    return winner_rating + change, loser_rating - change


def normalize_level(match: dict[str, Any]) -> str:
    raw = clean(match.get("tour_level")).lower()

    if raw in {"grand_slam", "atp_wta"}:
        return "main_tour"

    if raw in {"challenger", "itf", "qualifying"}:
        return raw

    return "unknown"


def new_surface_state() -> dict[str, Any]:
    return {"elo": DEFAULT_ELO, "matches": 0}


def new_level_state() -> dict[str, Any]:
    return {
        "overall_elo": DEFAULT_ELO,
        "matches": 0,
        "surfaces": {},
    }


def new_player_state() -> dict[str, Any]:
    return {
        "global_overall_elo": DEFAULT_ELO,
        "global_matches": 0,
        "global_surfaces": {},
        "levels": {},
    }


def ensure_surface(
    container: dict[str, Any],
    surface: str,
) -> dict[str, Any]:
    if surface not in container:
        container[surface] = new_surface_state()

    return container[surface]


def ensure_level(
    player: dict[str, Any],
    level: str,
) -> dict[str, Any]:
    if level not in player["levels"]:
        player["levels"][level] = new_level_state()

    return player["levels"][level]


def iter_matches():
    manifest = json.loads(
        SOURCE_MANIFEST.read_text(encoding="utf-8")
    )

    for item in sorted(
        manifest.get("year_files", []),
        key=lambda row: int(row.get("year", 0)),
    ):
        path = Path(item["path"])

        if not path.is_absolute():
            path = ROOT_DIR / path

        with gzip.open(
            path,
            "rt",
            encoding="utf-8",
        ) as handle:
            rows = [
                json.loads(line)
                for line in handle
                if line.strip()
            ]

        rows.sort(
            key=lambda row: (
                clean(row.get("date")),
                clean(row.get("tle_match_id")),
            )
        )

        for row in rows:
            yield row


def brier(probability: float) -> float:
    return (probability - 1.0) ** 2


def log_loss(probability: float) -> float:
    p = min(max(probability, 1e-12), 1 - 1e-12)
    return -math.log(p)


def calibration(probabilities: list[float]) -> list[dict[str, Any]]:
    buckets = []

    prediction_rows = [
        {
            "confidence": max(p, 1.0 - p),
            "correct": int(p >= 0.5),
        }
        for p in probabilities
    ]

    for lower in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        upper = round(min(lower + 0.05, 1.000001), 6)
        values = [
            row
            for row in prediction_rows
            if lower <= row["confidence"] < upper
        ]

        if not values:
            continue

        buckets.append(
            {
                "bucket": f"{lower:.2f}-{min(upper, 1.0):.2f}",
                "sample_size": len(values),
                "avg_predicted_confidence": round(
                    sum(row["confidence"] for row in values)
                    / len(values),
                    4,
                ),
                "actual_accuracy": round(
                    sum(row["correct"] for row in values)
                    / len(values),
                    4,
                ),
                "calibration_gap": round(
                    (
                        sum(row["correct"] for row in values)
                        / len(values)
                    )
                    - (
                        sum(row["confidence"] for row in values)
                        / len(values)
                    ),
                    4,
                ),
            }
        )

    return buckets


def summarize(probabilities: list[float]) -> dict[str, Any]:
    if not probabilities:
        return {
            "sample_size": 0,
            "accuracy": None,
            "brier_score": None,
            "log_loss": None,
            "avg_predicted_winner_probability": None,
            "avg_prediction_confidence": None,
            "calibration": [],
        }

    return {
        "sample_size": len(probabilities),
        "accuracy": round(
            sum(p >= 0.5 for p in probabilities)
            / len(probabilities),
            4,
        ),
        "brier_score": round(
            sum(brier(p) for p in probabilities)
            / len(probabilities),
            6,
        ),
        "log_loss": round(
            sum(log_loss(p) for p in probabilities)
            / len(probabilities),
            6,
        ),
        "avg_predicted_winner_probability": round(
            sum(probabilities) / len(probabilities),
            4,
        ),
        "avg_prediction_confidence": round(
            sum(max(p, 1.0 - p) for p in probabilities)
            / len(probabilities),
            4,
        ),
        "calibration": calibration(probabilities),
    }


def snapshot(
    winner: dict[str, Any],
    loser: dict[str, Any],
    level: str,
    surface: str,
) -> dict[str, Any]:
    winner_global = float(winner["global_overall_elo"])
    loser_global = float(loser["global_overall_elo"])

    global_overall = expected_score(
        winner_global,
        loser_global,
    )

    global_surface = None
    global_surface_matches = None

    if (
        surface in VALID_SURFACES
        and surface in winner["global_surfaces"]
        and surface in loser["global_surfaces"]
    ):
        ws = winner["global_surfaces"][surface]
        ls = loser["global_surfaces"][surface]
        global_surface = expected_score(
            float(ws["elo"]),
            float(ls["elo"]),
        )
        global_surface_matches = min(
            int(ws["matches"]),
            int(ls["matches"]),
        )

    winner_level = winner["levels"].get(level)
    loser_level = loser["levels"].get(level)

    level_overall = None
    level_matches = None
    level_surface = None
    level_surface_matches = None

    if winner_level and loser_level:
        level_overall = expected_score(
            float(winner_level["overall_elo"]),
            float(loser_level["overall_elo"]),
        )
        level_matches = min(
            int(winner_level["matches"]),
            int(loser_level["matches"]),
        )

        if (
            surface in VALID_SURFACES
            and surface in winner_level["surfaces"]
            and surface in loser_level["surfaces"]
        ):
            ws = winner_level["surfaces"][surface]
            ls = loser_level["surfaces"][surface]
            level_surface = expected_score(
                float(ws["elo"]),
                float(ls["elo"]),
            )
            level_surface_matches = min(
                int(ws["matches"]),
                int(ls["matches"]),
            )

    return {
        "global_overall": global_overall,
        "global_surface": global_surface,
        "global_surface_matches": global_surface_matches,
        "level_overall": level_overall,
        "level_matches": level_matches,
        "level_surface": level_surface,
        "level_surface_matches": level_surface_matches,
    }


def blended_probability(
    snap: dict[str, Any],
    min_level_matches: int,
    min_surface_matches: int,
    surface_weight: float,
    global_weight: float,
) -> float | None:
    level_prob = snap["level_overall"]
    level_matches = snap["level_matches"]

    if (
        level_prob is None
        or level_matches is None
        or level_matches < min_level_matches
    ):
        return None

    level_blend = float(level_prob)

    if (
        surface_weight > 0
        and snap["level_surface"] is not None
        and snap["level_surface_matches"] is not None
        and snap["level_surface_matches"] >= min_surface_matches
    ):
        level_blend = (
            (1.0 - surface_weight) * level_blend
            + surface_weight * float(snap["level_surface"])
        )

    global_prob = float(snap["global_overall"])

    if (
        surface_weight > 0
        and snap["global_surface"] is not None
        and snap["global_surface_matches"] is not None
        and snap["global_surface_matches"] >= min_surface_matches
    ):
        global_prob = (
            (1.0 - surface_weight) * global_prob
            + surface_weight * float(snap["global_surface"])
        )

    return (
        (1.0 - global_weight) * level_blend
        + global_weight * global_prob
    )


def update_ratings(
    winner: dict[str, Any],
    loser: dict[str, Any],
    level: str,
    surface: str,
) -> None:
    (
        winner["global_overall_elo"],
        loser["global_overall_elo"],
    ) = update_pair(
        float(winner["global_overall_elo"]),
        float(loser["global_overall_elo"]),
        GLOBAL_K,
    )

    winner["global_matches"] += 1
    loser["global_matches"] += 1

    if surface in VALID_SURFACES:
        ws = ensure_surface(
            winner["global_surfaces"],
            surface,
        )
        ls = ensure_surface(
            loser["global_surfaces"],
            surface,
        )
        ws["elo"], ls["elo"] = update_pair(
            float(ws["elo"]),
            float(ls["elo"]),
            GLOBAL_SURFACE_K,
        )
        ws["matches"] += 1
        ls["matches"] += 1

    winner_level = ensure_level(winner, level)
    loser_level = ensure_level(loser, level)

    (
        winner_level["overall_elo"],
        loser_level["overall_elo"],
    ) = update_pair(
        float(winner_level["overall_elo"]),
        float(loser_level["overall_elo"]),
        LEVEL_K,
    )
    winner_level["matches"] += 1
    loser_level["matches"] += 1

    if surface in VALID_SURFACES:
        ws = ensure_surface(
            winner_level["surfaces"],
            surface,
        )
        ls = ensure_surface(
            loser_level["surfaces"],
            surface,
        )
        ws["elo"], ls["elo"] = update_pair(
            float(ws["elo"]),
            float(ls["elo"]),
            LEVEL_SURFACE_K,
        )
        ws["matches"] += 1
        ls["matches"] += 1


def main() -> None:
    players = defaultdict(new_player_state)
    snapshots = []
    counters = Counter()

    for match in iter_matches():
        if not match.get("ready_for_tle"):
            counters["not_ready_for_tle"] += 1
            continue

        level = normalize_level(match)
        surface = clean(
            match.get("tournament", {}).get("surface")
        ).lower()
        gender = clean(match.get("gender")).lower()
        date = clean(match.get("date"))

        if level not in TARGET_LEVELS:
            counters["invalid_level"] += 1
            continue

        winner_raw = match.get("winner", {})
        loser_raw = match.get("loser", {})

        winner = players[player_key(winner_raw, gender)]
        loser = players[player_key(loser_raw, gender)]

        snap = snapshot(
            winner,
            loser,
            level,
            surface,
        )

        if date >= TEST_START_DATE:
            snapshots.append(
                {
                    "date": date,
                    "gender": gender,
                    "level": level,
                    "surface": surface,
                    **snap,
                }
            )

        update_ratings(
            winner,
            loser,
            level,
            surface,
        )
        counters["processed_matches"] += 1
        counters[f"level_{level}"] += 1

    grid_results = []
    best_by_level = {}
    common_sample_results = {}
    baselines_by_level = {}

    for level in sorted(TARGET_LEVELS):
        level_rows = [
            row for row in snapshots
            if row["level"] == level
        ]

        level_results = []

        for min_level_matches in LEVEL_SAMPLE_GRID:
            for min_surface_matches in SURFACE_SAMPLE_GRID:
                for surface_weight in SURFACE_WEIGHT_GRID:
                    for global_weight in GLOBAL_WEIGHT_GRID:
                        probabilities = []

                        for row in level_rows:
                            probability = blended_probability(
                                row,
                                min_level_matches=min_level_matches,
                                min_surface_matches=min_surface_matches,
                                surface_weight=surface_weight,
                                global_weight=global_weight,
                            )

                            if probability is not None:
                                probabilities.append(probability)

                        summary = summarize(probabilities)

                        result = {
                            "level": level,
                            "min_level_matches": min_level_matches,
                            "min_surface_matches": min_surface_matches,
                            "surface_weight": surface_weight,
                            "global_weight": global_weight,
                            **summary,
                        }

                        level_results.append(result)
                        grid_results.append(result)

        level_results.sort(
            key=lambda row: (
                row["log_loss"]
                if row["log_loss"] is not None
                else 999,
                row["brier_score"]
                if row["brier_score"] is not None
                else 999,
                -row["sample_size"],
            )
        )

        best_by_level[level] = level_results[:20]

        baseline_probabilities = {
            "global_overall": [],
            "global_surface_20": [],
            "level_overall": [],
            "level_surface_20": [],
        }

        for row in level_rows:
            baseline_probabilities["global_overall"].append(
                float(row["global_overall"])
            )

            global_surface = float(row["global_overall"])
            if (
                row["global_surface"] is not None
                and row["global_surface_matches"] is not None
                and row["global_surface_matches"] >= 5
            ):
                global_surface = (
                    0.8 * float(row["global_overall"])
                    + 0.2 * float(row["global_surface"])
                )
            baseline_probabilities["global_surface_20"].append(
                global_surface
            )

            if (
                row["level_overall"] is not None
                and row["level_matches"] is not None
                and row["level_matches"] >= 10
            ):
                baseline_probabilities["level_overall"].append(
                    float(row["level_overall"])
                )

                level_surface = float(row["level_overall"])
                if (
                    row["level_surface"] is not None
                    and row["level_surface_matches"] is not None
                    and row["level_surface_matches"] >= 5
                ):
                    level_surface = (
                        0.8 * float(row["level_overall"])
                        + 0.2 * float(row["level_surface"])
                    )
                baseline_probabilities["level_surface_20"].append(
                    level_surface
                )

        # Common sample: all tested configs use the strictest thresholds.
        common_rows = [
            row
            for row in level_rows
            if (
                row["level_overall"] is not None
                and row["level_matches"] is not None
                and row["level_matches"] >= max(LEVEL_SAMPLE_GRID)
            )
        ]

        common_results = []

        for surface_weight in SURFACE_WEIGHT_GRID:
            for global_weight in GLOBAL_WEIGHT_GRID:
                probabilities = []

                for row in common_rows:
                    probability = blended_probability(
                        row,
                        min_level_matches=max(LEVEL_SAMPLE_GRID),
                        min_surface_matches=max(SURFACE_SAMPLE_GRID),
                        surface_weight=surface_weight,
                        global_weight=global_weight,
                    )

                    if probability is not None:
                        probabilities.append(probability)

                common_results.append(
                    {
                        "surface_weight": surface_weight,
                        "global_weight": global_weight,
                        **summarize(probabilities),
                    }
                )

        common_results.sort(
            key=lambda row: (
                row["log_loss"]
                if row["log_loss"] is not None
                else 999,
                row["brier_score"]
                if row["brier_score"] is not None
                else 999,
            )
        )

        common_sample_results[level] = common_results[:20]
        baselines_by_level[level] = {
            name: summarize(values)
            for name, values in baseline_probabilities.items()
        }

    output = {
        "generated_at": now_iso(),
        "model_family": "tle",
        "model": "tle_model_search_v3",
        "settings": {
            "test_start_date": TEST_START_DATE,
            "level_sample_grid": LEVEL_SAMPLE_GRID,
            "surface_sample_grid": SURFACE_SAMPLE_GRID,
            "surface_weight_grid": SURFACE_WEIGHT_GRID,
            "global_weight_grid": GLOBAL_WEIGHT_GRID,
            "grand_slam_mapped_to": "main_tour",
        },
        "counts": {
            **dict(counters),
            "test_snapshots": len(snapshots),
            "players_final": len(players),
            "grid_rows": len(grid_results),
        },
        "best_by_level": best_by_level,
        "common_sample_best_by_level": common_sample_results,
        "baselines_by_level": baselines_by_level,
    }

    report = {
        "generated_at": output["generated_at"],
        "model": output["model"],
        "settings": output["settings"],
        "counts": output["counts"],
        "best_by_level": output["best_by_level"],
        "common_sample_best_by_level": output[
            "common_sample_best_by_level"
        ],
        "baselines_by_level": output["baselines_by_level"],
    }

    save_json(OUTPUT_FILE, output)
    save_json(REPORT_FILE, report)

    print("")
    print("TLE MODEL SEARCH V3 DONE")
    print("COUNTS:", output["counts"])

    for level in sorted(TARGET_LEVELS):
        print(f"\nBEST {level.upper()}:")

        for row in best_by_level[level][:10]:
            print(row)

        print(f"\nCOMMON SAMPLE {level.upper()}:")

        for row in common_sample_results[level][:10]:
            print(row)

        print(f"\nBASELINES {level.upper()}:")

        for name, summary in baselines_by_level[level].items():
            print(name, summary)

    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")
    print("")


if __name__ == "__main__":
    main()
