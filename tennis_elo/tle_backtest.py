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
    / "tle_walk_forward_backtest.json"
)

REPORT_FILE = (
    ROOT_DIR
    / "data"
    / "tle"
    / "reports"
    / "tle_walk_forward_backtest_report.json"
)

DEFAULT_ELO = float(os.getenv("TLE_DEFAULT_ELO", "1500"))
GLOBAL_K = float(os.getenv("TLE_GLOBAL_K", "24"))
GLOBAL_SURFACE_K = float(os.getenv("TLE_GLOBAL_SURFACE_K", "20"))
LEVEL_K = float(os.getenv("TLE_LEVEL_K", "24"))
LEVEL_SURFACE_K = float(os.getenv("TLE_LEVEL_SURFACE_K", "20"))

MIN_LEVEL_MATCHES = int(os.getenv("TLE_MIN_LEVEL_MATCHES", "10"))
MIN_LEVEL_SURFACE_MATCHES = int(
    os.getenv("TLE_MIN_LEVEL_SURFACE_MATCHES", "5")
)
TEST_START_DATE = os.getenv("TLE_TEST_START_DATE", "2025-01-01")

VALID_SURFACES = {"hard", "clay", "grass", "carpet"}

MODEL_NAMES = (
    "global_overall",
    "global_blended",
    "level_overall",
    "level_blended",
    "hybrid",
)


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

    year_files = manifest.get("year_files", [])

    for item in sorted(
        year_files,
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


def model_probabilities(
    winner_state: dict[str, Any],
    loser_state: dict[str, Any],
    level: str,
    surface: str,
) -> dict[str, float] | None:
    winner_global = float(
        winner_state["global_overall_elo"]
    )
    loser_global = float(
        loser_state["global_overall_elo"]
    )

    global_overall = expected_score(
        winner_global,
        loser_global,
    )

    global_blended = global_overall

    if (
        surface in VALID_SURFACES
        and surface in winner_state["global_surfaces"]
        and surface in loser_state["global_surfaces"]
    ):
        ws = winner_state["global_surfaces"][surface]
        ls = loser_state["global_surfaces"][surface]
        surface_prob = expected_score(
            float(ws["elo"]),
            float(ls["elo"]),
        )
        global_blended = 0.7 * global_overall + 0.3 * surface_prob

    winner_level = winner_state["levels"].get(level)
    loser_level = loser_state["levels"].get(level)

    if not winner_level or not loser_level:
        return {
            "global_overall": global_overall,
            "global_blended": global_blended,
            "level_overall": None,
            "level_blended": None,
            "hybrid": None,
        }

    if (
        int(winner_level["matches"]) < MIN_LEVEL_MATCHES
        or int(loser_level["matches"]) < MIN_LEVEL_MATCHES
    ):
        return {
            "global_overall": global_overall,
            "global_blended": global_blended,
            "level_overall": None,
            "level_blended": None,
            "hybrid": None,
        }

    level_overall = expected_score(
        float(winner_level["overall_elo"]),
        float(loser_level["overall_elo"]),
    )

    level_blended = level_overall

    if (
        surface in VALID_SURFACES
        and surface in winner_level["surfaces"]
        and surface in loser_level["surfaces"]
    ):
        ws = winner_level["surfaces"][surface]
        ls = loser_level["surfaces"][surface]

        if (
            int(ws["matches"]) >= MIN_LEVEL_SURFACE_MATCHES
            and int(ls["matches"]) >= MIN_LEVEL_SURFACE_MATCHES
        ):
            level_surface_prob = expected_score(
                float(ws["elo"]),
                float(ls["elo"]),
            )
            level_blended = (
                0.7 * level_overall
                + 0.3 * level_surface_prob
            )

    hybrid = (
        0.75 * level_blended
        + 0.25 * global_blended
    )

    return {
        "global_overall": global_overall,
        "global_blended": global_blended,
        "level_overall": level_overall,
        "level_blended": level_blended,
        "hybrid": hybrid,
    }


def brier(probability: float, outcome: int) -> float:
    return (probability - outcome) ** 2


def log_loss(probability: float, outcome: int) -> float:
    p = min(max(probability, 1e-12), 1 - 1e-12)
    return -(
        outcome * math.log(p)
        + (1 - outcome) * math.log(1 - p)
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}

    for model in MODEL_NAMES:
        values = [
            row
            for row in rows
            if row.get(model) is not None
        ]

        if not values:
            output[model] = {
                "sample_size": 0,
                "accuracy": None,
                "brier_score": None,
                "log_loss": None,
            }
            continue

        accuracy = sum(
            float(row[model]) >= 0.5
            for row in values
        ) / len(values)

        output[model] = {
            "sample_size": len(values),
            "accuracy": round(accuracy, 4),
            "brier_score": round(
                sum(
                    brier(float(row[model]), 1)
                    for row in values
                )
                / len(values),
                6,
            ),
            "log_loss": round(
                sum(
                    log_loss(float(row[model]), 1)
                    for row in values
                )
                / len(values),
                6,
            ),
            "avg_predicted_winner_probability": round(
                sum(float(row[model]) for row in values)
                / len(values),
                4,
            ),
        }

    return output


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
    predictions = []
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
        match_date = clean(match.get("date"))

        if level == "unknown":
            counters["unknown_level"] += 1
            continue

        winner_raw = match.get("winner", {})
        loser_raw = match.get("loser", {})

        winner_key = player_key(
            winner_raw,
            gender,
        )
        loser_key = player_key(
            loser_raw,
            gender,
        )

        winner = players[winner_key]
        loser = players[loser_key]

        probs = model_probabilities(
            winner,
            loser,
            level,
            surface,
        )

        if match_date >= TEST_START_DATE:
            predictions.append(
                {
                    "date": match_date,
                    "gender": gender,
                    "level": level,
                    "surface": surface,
                    "winner": clean(
                        winner_raw.get("name")
                    ),
                    "loser": clean(
                        loser_raw.get("name")
                    ),
                    **{
                        key: (
                            round(value, 6)
                            if value is not None
                            else None
                        )
                        for key, value in probs.items()
                    },
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
        counters[f"surface_{surface}"] += 1

    overall = summarize(predictions)

    by_level = {}
    for level in sorted(
        {row["level"] for row in predictions}
    ):
        by_level[level] = summarize(
            [
                row
                for row in predictions
                if row["level"] == level
            ]
        )

    by_surface = {}
    for surface in sorted(
        {row["surface"] for row in predictions}
    ):
        by_surface[surface] = summarize(
            [
                row
                for row in predictions
                if row["surface"] == surface
            ]
        )

    by_gender = {}
    for gender in sorted(
        {row["gender"] for row in predictions}
    ):
        by_gender[gender] = summarize(
            [
                row
                for row in predictions
                if row["gender"] == gender
            ]
        )

    model_ranking = sorted(
        MODEL_NAMES,
        key=lambda model: (
            overall[model]["log_loss"]
            if overall[model]["log_loss"] is not None
            else 999,
            overall[model]["brier_score"]
            if overall[model]["brier_score"] is not None
            else 999,
        ),
    )

    output = {
        "generated_at": now_iso(),
        "model_family": "tle",
        "model": "tle_walk_forward_v1",
        "settings": {
            "default_elo": DEFAULT_ELO,
            "global_k": GLOBAL_K,
            "global_surface_k": GLOBAL_SURFACE_K,
            "level_k": LEVEL_K,
            "level_surface_k": LEVEL_SURFACE_K,
            "min_level_matches": MIN_LEVEL_MATCHES,
            "min_level_surface_matches": (
                MIN_LEVEL_SURFACE_MATCHES
            ),
            "test_start_date": TEST_START_DATE,
            "grand_slam_mapped_to": "main_tour",
        },
        "counts": {
            **dict(counters),
            "test_predictions": len(predictions),
            "players_final": len(players),
        },
        "best_model": model_ranking[0],
        "model_ranking": model_ranking,
        "overall": overall,
        "by_level": by_level,
        "by_surface": by_surface,
        "by_gender": by_gender,
        "predictions": predictions,
    }

    report = {
        "generated_at": output["generated_at"],
        "model": output["model"],
        "settings": output["settings"],
        "counts": output["counts"],
        "best_model": output["best_model"],
        "model_ranking": output["model_ranking"],
        "overall": output["overall"],
        "by_level": output["by_level"],
        "by_surface": output["by_surface"],
        "by_gender": output["by_gender"],
    }

    save_json(OUTPUT_FILE, output)
    save_json(REPORT_FILE, report)

    print("")
    print("TLE WALK-FORWARD BACKTEST DONE")
    print("COUNTS:", output["counts"])
    print("BEST MODEL:", output["best_model"])
    print("MODEL RANKING:", output["model_ranking"])
    print("OVERALL:", output["overall"])
    print("BY LEVEL:", output["by_level"])
    print("BY SURFACE:", output["by_surface"])
    print("BY GENDER:", output["by_gender"])
    print(f"Output: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")
    print("")


if __name__ == "__main__":
    main()
