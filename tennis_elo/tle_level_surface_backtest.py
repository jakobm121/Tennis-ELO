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
    / "tle_level_surface_backtest.json"
)

REPORT_FILE = (
    ROOT_DIR
    / "data"
    / "tle"
    / "reports"
    / "tle_level_surface_backtest_report.json"
)

DEFAULT_ELO = float(os.getenv("TLE_DEFAULT_ELO", "1500"))
LEVEL_K = float(os.getenv("TLE_LEVEL_K", "24"))
LEVEL_SURFACE_K = float(
    os.getenv("TLE_LEVEL_SURFACE_K", "20")
)

TEST_START_DATE = os.getenv(
    "TLE_TEST_START_DATE",
    "2025-01-01",
)

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
        "3,5,10,15",
    ).split(",")
    if value.strip()
]

SURFACE_WEIGHT_GRID = [
    float(value)
    for value in os.getenv(
        "TLE_SURFACE_WEIGHT_GRID",
        "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    ).split(",")
    if value.strip()
]

VALID_SURFACES = {
    "hard",
    "clay",
    "grass",
    "carpet",
}

TARGET_LEVELS = {
    "main_tour",
    "challenger",
    "itf",
    "qualifying",
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_player_key(
    player: dict[str, Any],
    gender: str,
) -> str:
    player_id = player.get(
        "sackmann_player_id"
    )
    name = clean(player.get("name"))

    if player_id not in (None, ""):
        return (
            f"{gender}:sackmann:"
            f"{int(player_id)}"
        )

    normalized = re.sub(
        r"[^a-z0-9]+",
        " ",
        name.lower(),
    ).strip()

    return f"{gender}:name:{normalized}"


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


def update_pair(
    winner_rating: float,
    loser_rating: float,
    k_factor: float,
) -> tuple[float, float]:
    winner_probability = expected_score(
        winner_rating,
        loser_rating,
    )
    change = (
        k_factor
        * (1.0 - winner_probability)
    )

    return (
        winner_rating + change,
        loser_rating - change,
    )


def normalize_level(
    match: dict[str, Any],
) -> str:
    raw = clean(
        match.get("tour_level")
    ).lower()

    if raw in {
        "grand_slam",
        "atp_wta",
    }:
        return "main_tour"

    if raw in {
        "challenger",
        "itf",
        "qualifying",
    }:
        return raw

    return "unknown"


def new_surface_state() -> dict[str, Any]:
    return {
        "elo": DEFAULT_ELO,
        "matches": 0,
    }


def new_level_state() -> dict[str, Any]:
    return {
        "overall_elo": DEFAULT_ELO,
        "matches": 0,
        "surfaces": {},
    }


def new_player_state() -> dict[str, Any]:
    return {
        "levels": {},
    }


def ensure_level(
    player: dict[str, Any],
    level: str,
) -> dict[str, Any]:
    if level not in player["levels"]:
        player["levels"][
            level
        ] = new_level_state()

    return player["levels"][level]


def ensure_surface(
    level_state: dict[str, Any],
    surface: str,
) -> dict[str, Any]:
    surfaces = level_state["surfaces"]

    if surface not in surfaces:
        surfaces[
            surface
        ] = new_surface_state()

    return surfaces[surface]


def iter_matches():
    if not SOURCE_MANIFEST.exists():
        raise FileNotFoundError(
            f"Missing source manifest: "
            f"{SOURCE_MANIFEST}"
        )

    manifest = json.loads(
        SOURCE_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    year_files = manifest.get(
        "year_files",
        [],
    )

    for item in sorted(
        year_files,
        key=lambda row: int(
            row.get("year", 0)
        ),
    ):
        path = Path(item["path"])

        if not path.is_absolute():
            path = ROOT_DIR / path

        if not path.exists():
            raise FileNotFoundError(
                f"Missing source file: {path}"
            )

        rows = []

        with gzip.open(
            path,
            "rt",
            encoding="utf-8",
        ) as handle:
            for line in handle:
                line = line.strip()

                if line:
                    rows.append(
                        json.loads(line)
                    )

        rows.sort(
            key=lambda row: (
                clean(row.get("date")),
                clean(
                    row.get(
                        "tle_match_id"
                    )
                ),
            )
        )

        for row in rows:
            yield row


def snapshot(
    winner: dict[str, Any],
    loser: dict[str, Any],
    level: str,
    surface: str,
) -> dict[str, Any]:
    winner_level = winner[
        "levels"
    ].get(level)
    loser_level = loser[
        "levels"
    ].get(level)

    if (
        winner_level is None
        or loser_level is None
    ):
        return {
            "level_probability": None,
            "level_matches": None,
            "surface_probability": None,
            "surface_matches": None,
        }

    level_probability = expected_score(
        float(
            winner_level[
                "overall_elo"
            ]
        ),
        float(
            loser_level[
                "overall_elo"
            ]
        ),
    )

    level_matches = min(
        int(
            winner_level[
                "matches"
            ]
        ),
        int(
            loser_level[
                "matches"
            ]
        ),
    )

    surface_probability = None
    surface_matches = None

    if (
        surface in VALID_SURFACES
        and surface
        in winner_level[
            "surfaces"
        ]
        and surface
        in loser_level[
            "surfaces"
        ]
    ):
        winner_surface = (
            winner_level[
                "surfaces"
            ][surface]
        )
        loser_surface = (
            loser_level[
                "surfaces"
            ][surface]
        )

        surface_probability = expected_score(
            float(
                winner_surface[
                    "elo"
                ]
            ),
            float(
                loser_surface[
                    "elo"
                ]
            ),
        )

        surface_matches = min(
            int(
                winner_surface[
                    "matches"
                ]
            ),
            int(
                loser_surface[
                    "matches"
                ]
            ),
        )

    return {
        "level_probability": (
            level_probability
        ),
        "level_matches": level_matches,
        "surface_probability": (
            surface_probability
        ),
        "surface_matches": (
            surface_matches
        ),
    }


def combined_probability(
    row: dict[str, Any],
    min_level_matches: int,
    min_surface_matches: int,
    surface_weight: float,
) -> float | None:
    level_probability = row[
        "level_probability"
    ]
    surface_probability = row[
        "surface_probability"
    ]
    level_matches = row[
        "level_matches"
    ]
    surface_matches = row[
        "surface_matches"
    ]

    if (
        level_probability is None
        or surface_probability is None
        or level_matches is None
        or surface_matches is None
    ):
        return None

    if (
        level_matches
        < min_level_matches
    ):
        return None

    if (
        surface_matches
        < min_surface_matches
    ):
        return None

    return (
        (1.0 - surface_weight)
        * float(level_probability)
        + surface_weight
        * float(surface_probability)
    )


def brier(
    probability: float,
) -> float:
    return (
        probability - 1.0
    ) ** 2


def log_loss(
    probability: float,
) -> float:
    probability = min(
        max(
            probability,
            1e-12,
        ),
        1.0 - 1e-12,
    )
    return -math.log(probability)


def calibration(
    probabilities: list[float],
) -> list[dict[str, Any]]:
    prediction_rows = [
        {
            "confidence": max(
                probability,
                1.0 - probability,
            ),
            "correct": int(
                probability >= 0.5
            ),
        }
        for probability
        in probabilities
    ]

    buckets = []

    for lower in (
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ):
        upper = min(
            lower + 0.05,
            1.000001,
        )

        values = [
            row
            for row
            in prediction_rows
            if (
                lower
                <= row[
                    "confidence"
                ]
                < upper
            )
        ]

        if not values:
            continue

        avg_confidence = (
            sum(
                row[
                    "confidence"
                ]
                for row
                in values
            )
            / len(values)
        )
        actual_accuracy = (
            sum(
                row[
                    "correct"
                ]
                for row
                in values
            )
            / len(values)
        )

        buckets.append(
            {
                "bucket": (
                    f"{lower:.2f}-"
                    f"{min(upper, 1.0):.2f}"
                ),
                "sample_size": len(values),
                "avg_predicted_confidence": round(
                    avg_confidence,
                    4,
                ),
                "actual_accuracy": round(
                    actual_accuracy,
                    4,
                ),
                "calibration_gap": round(
                    (
                        actual_accuracy
                        - avg_confidence
                    ),
                    4,
                ),
            }
        )

    return buckets


def summarize(
    probabilities: list[float],
) -> dict[str, Any]:
    if not probabilities:
        return {
            "sample_size": 0,
            "accuracy": None,
            "brier_score": None,
            "log_loss": None,
            "avg_winner_probability": None,
            "avg_prediction_confidence": None,
            "calibration": [],
        }

    return {
        "sample_size": len(
            probabilities
        ),
        "accuracy": round(
            (
                sum(
                    probability >= 0.5
                    for probability
                    in probabilities
                )
                / len(probabilities)
            ),
            4,
        ),
        "brier_score": round(
            (
                sum(
                    brier(probability)
                    for probability
                    in probabilities
                )
                / len(probabilities)
            ),
            6,
        ),
        "log_loss": round(
            (
                sum(
                    log_loss(
                        probability
                    )
                    for probability
                    in probabilities
                )
                / len(probabilities)
            ),
            6,
        ),
        "avg_winner_probability": round(
            (
                sum(probabilities)
                / len(probabilities)
            ),
            4,
        ),
        "avg_prediction_confidence": round(
            (
                sum(
                    max(
                        probability,
                        1.0
                        - probability,
                    )
                    for probability
                    in probabilities
                )
                / len(probabilities)
            ),
            4,
        ),
        "calibration": calibration(
            probabilities
        ),
    }


def update_ratings(
    winner: dict[str, Any],
    loser: dict[str, Any],
    level: str,
    surface: str,
) -> None:
    winner_level = ensure_level(
        winner,
        level,
    )
    loser_level = ensure_level(
        loser,
        level,
    )

    (
        winner_level[
            "overall_elo"
        ],
        loser_level[
            "overall_elo"
        ],
    ) = update_pair(
        float(
            winner_level[
                "overall_elo"
            ]
        ),
        float(
            loser_level[
                "overall_elo"
            ]
        ),
        LEVEL_K,
    )

    winner_level[
        "matches"
    ] += 1
    loser_level[
        "matches"
    ] += 1

    if surface in VALID_SURFACES:
        winner_surface = ensure_surface(
            winner_level,
            surface,
        )
        loser_surface = ensure_surface(
            loser_level,
            surface,
        )

        (
            winner_surface["elo"],
            loser_surface["elo"],
        ) = update_pair(
            float(
                winner_surface[
                    "elo"
                ]
            ),
            float(
                loser_surface[
                    "elo"
                ]
            ),
            LEVEL_SURFACE_K,
        )

        winner_surface[
            "matches"
        ] += 1
        loser_surface[
            "matches"
        ] += 1


def evaluate_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results = []

    for min_level_matches in (
        LEVEL_SAMPLE_GRID
    ):
        for min_surface_matches in (
            SURFACE_SAMPLE_GRID
        ):
            common_rows = [
                row
                for row in rows
                if (
                    row[
                        "level_probability"
                    ]
                    is not None
                    and row[
                        "surface_probability"
                    ]
                    is not None
                    and row[
                        "level_matches"
                    ]
                    is not None
                    and row[
                        "surface_matches"
                    ]
                    is not None
                    and row[
                        "level_matches"
                    ]
                    >= min_level_matches
                    and row[
                        "surface_matches"
                    ]
                    >= min_surface_matches
                )
            ]

            for surface_weight in (
                SURFACE_WEIGHT_GRID
            ):
                probabilities = [
                    combined_probability(
                        row,
                        min_level_matches,
                        min_surface_matches,
                        surface_weight,
                    )
                    for row
                    in common_rows
                ]

                probabilities = [
                    probability
                    for probability
                    in probabilities
                    if probability
                    is not None
                ]

                summary = summarize(
                    probabilities
                )

                results.append(
                    {
                        "min_level_matches": (
                            min_level_matches
                        ),
                        "min_surface_matches": (
                            min_surface_matches
                        ),
                        "level_weight": round(
                            1.0
                            - surface_weight,
                            4,
                        ),
                        "surface_weight": round(
                            surface_weight,
                            4,
                        ),
                        **summary,
                    }
                )

    results.sort(
        key=lambda row: (
            row[
                "log_loss"
            ]
            if row[
                "log_loss"
            ]
            is not None
            else 999,
            row[
                "brier_score"
            ]
            if row[
                "brier_score"
            ]
            is not None
            else 999,
            -row[
                "sample_size"
            ],
        )
    )

    return results


def main() -> None:
    players = defaultdict(
        new_player_state
    )
    snapshots = []
    counters = Counter()

    for match in iter_matches():
        if not match.get(
            "ready_for_tle"
        ):
            counters[
                "not_ready_for_tle"
            ] += 1
            continue

        level = normalize_level(
            match
        )
        surface = clean(
            match.get(
                "tournament",
                {},
            ).get("surface")
        ).lower()
        gender = clean(
            match.get("gender")
        ).lower()
        date = clean(
            match.get("date")
        )

        if level not in TARGET_LEVELS:
            counters[
                "invalid_level"
            ] += 1
            continue

        winner_raw = match.get(
            "winner",
            {},
        )
        loser_raw = match.get(
            "loser",
            {},
        )

        winner = players[
            normalize_player_key(
                winner_raw,
                gender,
            )
        ]
        loser = players[
            normalize_player_key(
                loser_raw,
                gender,
            )
        ]

        pre_match = snapshot(
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
                    **pre_match,
                }
            )

        update_ratings(
            winner,
            loser,
            level,
            surface,
        )

        counters[
            "processed_matches"
        ] += 1
        counters[
            f"level_{level}"
        ] += 1
        counters[
            f"surface_{surface}"
        ] += 1

    by_level = {}
    best_by_level = {}
    by_level_surface = {}

    for level in sorted(
        TARGET_LEVELS
    ):
        level_rows = [
            row
            for row in snapshots
            if row["level"] == level
        ]

        level_results = evaluate_rows(
            level_rows
        )

        by_level[level] = (
            level_results
        )
        best_by_level[level] = (
            level_results[:20]
        )

        by_level_surface[
            level
        ] = {}

        for surface in sorted(
            VALID_SURFACES
        ):
            surface_rows = [
                row
                for row in level_rows
                if (
                    row[
                        "surface"
                    ]
                    == surface
                )
            ]

            surface_results = evaluate_rows(
                surface_rows
            )

            by_level_surface[
                level
            ][surface] = (
                surface_results[:20]
            )

    output = {
        "generated_at": now_iso(),
        "model_family": "tle",
        "model": (
            "tle_level_surface_only_v1"
        ),
        "settings": {
            "default_elo": (
                DEFAULT_ELO
            ),
            "level_k": LEVEL_K,
            "level_surface_k": (
                LEVEL_SURFACE_K
            ),
            "test_start_date": (
                TEST_START_DATE
            ),
            "level_sample_grid": (
                LEVEL_SAMPLE_GRID
            ),
            "surface_sample_grid": (
                SURFACE_SAMPLE_GRID
            ),
            "surface_weight_grid": (
                SURFACE_WEIGHT_GRID
            ),
            "grand_slam_mapped_to": (
                "main_tour"
            ),
            "global_elo_used": False,
        },
        "counts": {
            **dict(counters),
            "test_snapshots": len(
                snapshots
            ),
            "players_final": len(
                players
            ),
        },
        "best_by_level": (
            best_by_level
        ),
        "best_by_level_surface": (
            by_level_surface
        ),
        "all_results_by_level": (
            by_level
        ),
    }

    report = {
        "generated_at": output[
            "generated_at"
        ],
        "model": output["model"],
        "settings": output[
            "settings"
        ],
        "counts": output[
            "counts"
        ],
        "best_by_level": output[
            "best_by_level"
        ],
        "best_by_level_surface": (
            output[
                "best_by_level_surface"
            ]
        ),
    }

    save_json(
        OUTPUT_FILE,
        output,
    )
    save_json(
        REPORT_FILE,
        report,
    )

    print("")
    print(
        "TLE LEVEL-SURFACE "
        "BACKTEST DONE"
    )
    print(
        "COUNTS:",
        output["counts"],
    )

    for level in sorted(
        TARGET_LEVELS
    ):
        print(
            f"\nBEST "
            f"{level.upper()}:"
        )

        for row in (
            best_by_level[
                level
            ][:10]
        ):
            print(row)

        for surface in sorted(
            VALID_SURFACES
        ):
            rows = (
                by_level_surface[
                    level
                ][surface]
            )

            if not rows:
                continue

            print(
                f"\nBEST "
                f"{level.upper()} "
                f"{surface.upper()}:"
            )

            for row in rows[:5]:
                print(row)

    print(
        f"\nOutput: {OUTPUT_FILE}"
    )
    print(
        f"Report: {REPORT_FILE}"
    )
    print("")


if __name__ == "__main__":
    main()
