import math
import os
import re
from collections import defaultdict, deque
from datetime import date, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tennis_elo.config import ROOT_DIR
from tennis_elo.utils import clean_str, load_json, now_iso, save_json


ARCHIVE_FILE = ROOT_DIR / "data" / "totals" / "scorelines_archive.json"
OUTPUT_FILE = ROOT_DIR / "data" / "backtests" / "first_set_over_9_5_logistic_backtest.json"
REPORT_FILE = ROOT_DIR / "data" / "reports" / "first_set_over_9_5_logistic_backtest_report.json"

TEST_START = os.getenv("FIRST_SET_LOGIT_TEST_START", "2026-03-01")
MIN_PLAYER_SAMPLE = int(os.getenv("FIRST_SET_LOGIT_MIN_PLAYER_SAMPLE", "12"))
MIN_TRAIN_ROWS = int(os.getenv("FIRST_SET_LOGIT_MIN_TRAIN_ROWS", "300"))
REFIT_FREQUENCY = os.getenv("FIRST_SET_LOGIT_REFIT_FREQUENCY", "monthly")
RECENT_WINDOW = int(os.getenv("FIRST_SET_LOGIT_RECENT_WINDOW", "10"))

FEATURE_NAMES = [
    "p1_over_rate",
    "p2_over_rate",
    "p1_surface_over_rate",
    "p2_surface_over_rate",
    "p1_recent_over_rate",
    "p2_recent_over_rate",
    "p1_first_set_win_rate",
    "p2_first_set_win_rate",
    "p1_blowout_rate",
    "p2_blowout_rate",
    "p1_close_rate",
    "p2_close_rate",
    "p1_tiebreak_rate",
    "p2_tiebreak_rate",
    "abs_over_rate_diff",
    "abs_first_set_win_rate_diff",
    "sample_log_min",
    "sample_log_max",
    "surface_hard",
    "surface_clay",
    "surface_grass",
    "tour_grand_slam",
    "tour_challenger",
    "tour_itf",
    "gender_women",
]


def parse_date(raw: Any) -> date | None:
    value = clean_str(raw)
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d.%m.%Y", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def player_key(name: Any) -> str:
    value = clean_str(name).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def clip_probability(value: float) -> float:
    return min(0.999999, max(0.000001, value))


def safe_rate(successes: int, sample: int, prior: float, strength: float = 8.0) -> float:
    return (successes + prior * strength) / (sample + strength)


def accuracy(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    correct = sum((row["probability_over_9_5"] >= 0.5) == bool(row["actual_over_9_5"]) for row in rows)
    return round(correct / len(rows), 4)


def brier(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return round(mean((row["probability_over_9_5"] - row["actual_over_9_5"]) ** 2 for row in rows), 6)


def log_loss(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return round(
        mean(
            -(
                row["actual_over_9_5"] * math.log(clip_probability(row["probability_over_9_5"]))
                + (1 - row["actual_over_9_5"])
                * math.log(clip_probability(1 - row["probability_over_9_5"]))
            )
            for row in rows
        ),
        6,
    )


def calibration(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for low in [i / 10 for i in range(10)]:
        high = low + 0.1
        subset = [
            row for row in rows
            if low <= row["probability_over_9_5"] < high
            or (high >= 1 and row["probability_over_9_5"] == 1)
        ]
        if not subset:
            continue
        out.append(
            {
                "bucket": f"{low:.1f}-{high:.1f}",
                "sample_size": len(subset),
                "avg_predicted": round(mean(r["probability_over_9_5"] for r in subset), 4),
                "actual_over_rate": round(mean(r["actual_over_9_5"] for r in subset), 4),
            }
        )
    return out


class State:
    def __init__(self) -> None:
        self.matches = 0
        self.overs = 0
        self.first_set_wins = 0
        self.blowouts = 0
        self.close_sets = 0
        self.tiebreaks = 0
        self.surface_matches = defaultdict(int)
        self.surface_overs = defaultdict(int)
        self.recent = deque(maxlen=RECENT_WINDOW)

    def update(self, surface: str, actual_over: int, first_set_won: int, first_set_games: int, tiebreak: int) -> None:
        self.matches += 1
        self.overs += actual_over
        self.first_set_wins += first_set_won
        self.blowouts += int(first_set_games <= 8)
        self.close_sets += int(first_set_games >= 10)
        self.tiebreaks += tiebreak
        self.surface_matches[surface] += 1
        self.surface_overs[surface] += actual_over
        self.recent.append(actual_over)

    def features(self, surface: str, prior_over: float, prior_win: float) -> dict[str, float]:
        overall_over = safe_rate(self.overs, self.matches, prior_over)
        surface_over = safe_rate(
            self.surface_overs[surface],
            self.surface_matches[surface],
            prior_over,
        )
        recent_over = (
            safe_rate(sum(self.recent), len(self.recent), prior_over, strength=4.0)
            if self.recent else prior_over
        )
        first_set_win_rate = safe_rate(self.first_set_wins, self.matches, prior_win)
        blowout_rate = safe_rate(self.blowouts, self.matches, 0.25)
        close_rate = safe_rate(self.close_sets, self.matches, 0.50)
        tiebreak_rate = safe_rate(self.tiebreaks, self.matches, 0.10)

        return {
            "over_rate": overall_over,
            "surface_over_rate": surface_over,
            "recent_over_rate": recent_over,
            "first_set_win_rate": first_set_win_rate,
            "blowout_rate": blowout_rate,
            "close_rate": close_rate,
            "tiebreak_rate": tiebreak_rate,
        }


def valid_row(row: dict[str, Any]) -> bool:
    if row.get("retired") or row.get("walkover"):
        return False
    if not parse_date(row.get("date")):
        return False
    if not clean_str(row.get("player_1")) or not clean_str(row.get("player_2")):
        return False
    try:
        games = int(row.get("first_set_games"))
    except (TypeError, ValueError):
        return False
    return 6 <= games <= 20


def first_set_side_result(row: dict[str, Any]) -> tuple[int, int]:
    first = (row.get("set_scores") or [{}])[0]
    p1 = int(first.get("p1_games", 0))
    p2 = int(first.get("p2_games", 0))
    return int(p1 > p2), int(p2 > p1)


def global_priors(rows: list[dict[str, Any]], cutoff: date) -> tuple[float, float]:
    overs = []
    p1_wins = []
    for row in rows:
        d = parse_date(row.get("date"))
        if not d or d >= cutoff:
            continue
        overs.append(int(row["first_set_games"]) > 9.5)
        p1_win, _ = first_set_side_result(row)
        p1_wins.append(p1_win)
    return (
        sum(overs) / len(overs) if overs else 0.5,
        sum(p1_wins) / len(p1_wins) if p1_wins else 0.5,
    )


def build_feature_vector(
    row: dict[str, Any],
    state_1: State,
    state_2: State,
    prior_over: float,
    prior_win: float,
) -> list[float]:
    surface = clean_str(row.get("surface")).lower() or "unknown"
    f1 = state_1.features(surface, prior_over, prior_win)
    f2 = state_2.features(surface, prior_over, prior_win)

    tour = clean_str(row.get("tour_level")).lower()
    gender = clean_str(row.get("gender")).lower()

    values = {
        "p1_over_rate": f1["over_rate"],
        "p2_over_rate": f2["over_rate"],
        "p1_surface_over_rate": f1["surface_over_rate"],
        "p2_surface_over_rate": f2["surface_over_rate"],
        "p1_recent_over_rate": f1["recent_over_rate"],
        "p2_recent_over_rate": f2["recent_over_rate"],
        "p1_first_set_win_rate": f1["first_set_win_rate"],
        "p2_first_set_win_rate": f2["first_set_win_rate"],
        "p1_blowout_rate": f1["blowout_rate"],
        "p2_blowout_rate": f2["blowout_rate"],
        "p1_close_rate": f1["close_rate"],
        "p2_close_rate": f2["close_rate"],
        "p1_tiebreak_rate": f1["tiebreak_rate"],
        "p2_tiebreak_rate": f2["tiebreak_rate"],
        "abs_over_rate_diff": abs(f1["over_rate"] - f2["over_rate"]),
        "abs_first_set_win_rate_diff": abs(f1["first_set_win_rate"] - f2["first_set_win_rate"]),
        "sample_log_min": math.log1p(min(state_1.matches, state_2.matches)),
        "sample_log_max": math.log1p(max(state_1.matches, state_2.matches)),
        "surface_hard": float(surface == "hard"),
        "surface_clay": float(surface == "clay"),
        "surface_grass": float(surface == "grass"),
        "tour_grand_slam": float(tour == "grand_slam"),
        "tour_challenger": float(tour == "challenger"),
        "tour_itf": float(tour == "itf"),
        "gender_women": float(gender == "women"),
    }

    return [values[name] for name in FEATURE_NAMES]


def model_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    C=0.5,
                    max_iter=2000,
                    class_weight=None,
                    random_state=42,
                ),
            ),
        ]
    )


def refit_key(match_date: date) -> str:
    if REFIT_FREQUENCY == "monthly":
        return f"{match_date.year:04d}-{match_date.month:02d}"
    return match_date.isoformat()


def subset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_size": len(rows),
        "accuracy": accuracy(rows),
        "brier_score": brier(rows),
        "log_loss": log_loss(rows),
        "actual_over_rate": round(mean(r["actual_over_9_5"] for r in rows), 4) if rows else None,
        "avg_predicted_over": round(mean(r["probability_over_9_5"] for r in rows), 4) if rows else None,
    }


def main() -> None:
    payload = load_json(ARCHIVE_FILE, {})
    rows = payload.get("matches", []) if isinstance(payload, dict) else payload
    rows = [row for row in rows if isinstance(row, dict) and valid_row(row)]
    rows.sort(key=lambda r: (parse_date(r["date"]), clean_str(r.get("tournament"))))

    test_start = parse_date(TEST_START)
    if not test_start:
        raise SystemExit(f"Invalid FIRST_SET_LOGIT_TEST_START: {TEST_START}")

    prior_over, prior_win = global_priors(rows, test_start)

    states: dict[str, State] = defaultdict(State)
    train_x: dict[str, list[list[float]]] = defaultdict(list)
    train_y: dict[str, list[int]] = defaultdict(list)
    pooled_x: list[list[float]] = []
    pooled_y: list[int] = []

    models: dict[str, Pipeline] = {}
    active_refit_key = ""
    predictions = []
    counters = defaultdict(int)

    for row in rows:
        match_date = parse_date(row["date"])
        p1 = clean_str(row["player_1"])
        p2 = clean_str(row["player_2"])
        k1 = player_key(p1)
        k2 = player_key(p2)
        s1 = states[k1]
        s2 = states[k2]

        surface = clean_str(row.get("surface")).lower() or "unknown"
        gender = clean_str(row.get("gender")).lower() or "unknown"
        y = int(int(row["first_set_games"]) > 9.5)
        p1_win, p2_win = first_set_side_result(row)
        tb = int(bool(row.get("first_set_tiebreak")))

        eligible = s1.matches >= MIN_PLAYER_SAMPLE and s2.matches >= MIN_PLAYER_SAMPLE

        if eligible:
            x = build_feature_vector(row, s1, s2, prior_over, prior_win)

            if match_date < test_start:
                train_x[gender].append(x)
                train_y[gender].append(y)
                pooled_x.append(x)
                pooled_y.append(y)
            else:
                current_key = refit_key(match_date)
                if current_key != active_refit_key:
                    active_refit_key = current_key
                    models = {}

                    if len(pooled_x) >= MIN_TRAIN_ROWS and len(set(pooled_y)) > 1:
                        pooled_model = model_pipeline()
                        pooled_model.fit(pooled_x, pooled_y)
                        models["pooled"] = pooled_model

                    for group in ("men", "women"):
                        if len(train_x[group]) >= MIN_TRAIN_ROWS and len(set(train_y[group])) > 1:
                            group_model = model_pipeline()
                            group_model.fit(train_x[group], train_y[group])
                            models[group] = group_model

                model = models.get(gender) or models.get("pooled")
                if model is None:
                    counters["missing_model"] += 1
                else:
                    probability = float(model.predict_proba([x])[0][1])
                    predictions.append(
                        {
                            "date": match_date.isoformat(),
                            "tournament": clean_str(row.get("tournament")),
                            "tour_level": clean_str(row.get("tour_level")),
                            "gender": gender,
                            "surface": surface,
                            "player_1": p1,
                            "player_2": p2,
                            "first_set_score": clean_str(row.get("first_set_score")),
                            "first_set_games": int(row["first_set_games"]),
                            "actual_over_9_5": y,
                            "probability_over_9_5": round(probability, 6),
                            "prediction": "over" if probability >= 0.5 else "under",
                            "correct": (probability >= 0.5) == bool(y),
                            "model_group": gender if gender in models else "pooled",
                            "player_1_sample": s1.matches,
                            "player_2_sample": s2.matches,
                        }
                    )

                    # Add resolved test row to expanding training set after prediction.
                    train_x[gender].append(x)
                    train_y[gender].append(y)
                    pooled_x.append(x)
                    pooled_y.append(y)
        else:
            counters["insufficient_sample"] += 1

        s1.update(surface, y, p1_win, int(row["first_set_games"]), tb)
        s2.update(surface, y, p2_win, int(row["first_set_games"]), tb)

    by_surface = {
        surface: subset_summary([row for row in predictions if row["surface"] == surface])
        for surface in sorted({row["surface"] for row in predictions})
    }
    by_gender = {
        gender: subset_summary([row for row in predictions if row["gender"] == gender])
        for gender in sorted({row["gender"] for row in predictions})
    }

    summary = subset_summary(predictions)
    summary.update(
        {
            "test_start": test_start.isoformat(),
            "baseline_over_9_5": round(prior_over, 4),
            "test_matches": len(predictions),
            "train_rows_final": len(pooled_x),
        }
    )

    output = {
        "generated_at": now_iso(),
        "model": "first_set_over_9_5_walk_forward_logistic_v1",
        "source_file": str(ARCHIVE_FILE),
        "feature_names": FEATURE_NAMES,
        "settings": {
            "test_start": test_start.isoformat(),
            "min_player_sample": MIN_PLAYER_SAMPLE,
            "min_train_rows": MIN_TRAIN_ROWS,
            "recent_window": RECENT_WINDOW,
            "refit_frequency": REFIT_FREQUENCY,
        },
        "summary": summary,
        "by_surface": by_surface,
        "by_gender": by_gender,
        "calibration": calibration(predictions),
        "counters": dict(counters),
        "predictions": predictions,
    }

    report = {
        "generated_at": now_iso(),
        "model": output["model"],
        "summary": summary,
        "by_surface": by_surface,
        "by_gender": by_gender,
        "calibration": output["calibration"],
        "counters": dict(counters),
    }

    save_json(OUTPUT_FILE, output)
    save_json(REPORT_FILE, report)

    print("")
    print("FIRST SET OVER 9.5 LOGISTIC BACKTEST DONE")
    print("SUMMARY:", summary)
    print("BY SURFACE:", by_surface)
    print("BY GENDER:", by_gender)
    print("CALIBRATION:", output["calibration"])
    print("COUNTERS:", dict(counters))
    print(f"Output: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")
    print("")


if __name__ == "__main__":
    main()
