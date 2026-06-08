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
OUTPUT_FILE = ROOT_DIR / "data" / "backtests" / "first_set_over_9_5_logistic_v2_backtest.json"
REPORT_FILE = ROOT_DIR / "data" / "reports" / "first_set_over_9_5_logistic_v2_backtest_report.json"

TEST_START = os.getenv("FIRST_SET_LOGIT_V2_TEST_START", "2026-03-01")
MIN_PLAYER_SAMPLE = int(os.getenv("FIRST_SET_LOGIT_V2_MIN_PLAYER_SAMPLE", "10"))
MIN_TRAIN_ROWS = int(os.getenv("FIRST_SET_LOGIT_V2_MIN_TRAIN_ROWS", "300"))
RECENT_WINDOW = int(os.getenv("FIRST_SET_LOGIT_V2_RECENT_WINDOW", "10"))
REFIT_FREQUENCY = os.getenv("FIRST_SET_LOGIT_V2_REFIT_FREQUENCY", "monthly")

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
    "rank_gap_log",
    "rank_missing",
    "favorite_probability",
    "odds_balance",
    "odds_missing",
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
    values = []
    for row in rows:
        p = clip_probability(row["probability_over_9_5"])
        y = row["actual_over_9_5"]
        values.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))
    return round(mean(values), 6)


def calibration(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for low in [i / 10 for i in range(10)]:
        high = low + 0.1
        subset = [
            row for row in rows
            if low <= row["probability_over_9_5"] < high
            or (high >= 1.0 and row["probability_over_9_5"] == 1.0)
        ]
        if not subset:
            continue
        output.append(
            {
                "bucket": f"{low:.1f}-{high:.1f}",
                "sample_size": len(subset),
                "avg_predicted": round(mean(r["probability_over_9_5"] for r in subset), 4),
                "actual_over_rate": round(mean(r["actual_over_9_5"] for r in subset), 4),
            }
        )
    return output


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
        surface_over = safe_rate(self.surface_overs[surface], self.surface_matches[surface], prior_over)
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
        match_date = parse_date(row.get("date"))
        if not match_date or match_date >= cutoff:
            continue
        overs.append(int(row["first_set_games"]) > 9.5)
        p1_win, _ = first_set_side_result(row)
        p1_wins.append(p1_win)
    return (
        sum(overs) / len(overs) if overs else 0.5,
        sum(p1_wins) / len(p1_wins) if p1_wins else 0.5,
    )


def numeric(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def ranking_features(row: dict[str, Any]) -> tuple[float, float]:
    rank_1 = numeric(row.get("rank_1"))
    rank_2 = numeric(row.get("rank_2"))

    if not rank_1 or not rank_2 or rank_1 <= 0 or rank_2 <= 0:
        return 0.0, 1.0

    return math.log1p(abs(rank_1 - rank_2)), 0.0


def odds_features(row: dict[str, Any]) -> tuple[float, float, float]:
    bookmaker = row.get("bookmaker") or {}
    odd_1 = numeric(bookmaker.get("player_1_odds"))
    odd_2 = numeric(bookmaker.get("player_2_odds"))

    if not odd_1 or not odd_2 or odd_1 <= 1.0 or odd_2 <= 1.0:
        return 0.5, 0.0, 1.0

    raw_p1 = 1.0 / odd_1
    raw_p2 = 1.0 / odd_2
    total = raw_p1 + raw_p2

    if total <= 0:
        return 0.5, 0.0, 1.0

    p1 = raw_p1 / total
    p2 = raw_p2 / total

    favorite_probability = max(p1, p2)
    odds_balance = 1.0 - abs(p1 - p2)

    return favorite_probability, odds_balance, 0.0


def build_feature_vector(
    row: dict[str, Any],
    state_1: State,
    state_2: State,
    prior_over: float,
    prior_win: float,
) -> list[float]:
    surface = clean_str(row.get("surface")).lower() or "unknown"
    tour = clean_str(row.get("tour_level")).lower()
    gender = clean_str(row.get("gender")).lower()

    f1 = state_1.features(surface, prior_over, prior_win)
    f2 = state_2.features(surface, prior_over, prior_win)

    rank_gap_log, rank_missing = ranking_features(row)
    favorite_probability, odds_balance, odds_missing = odds_features(row)

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
        "rank_gap_log": rank_gap_log,
        "rank_missing": rank_missing,
        "favorite_probability": favorite_probability,
        "odds_balance": odds_balance,
        "odds_missing": odds_missing,
    }

    return [values[name] for name in FEATURE_NAMES]


def model_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    C=0.2,
                    max_iter=2500,
                    random_state=42,
                ),
            ),
        ]
    )


def refit_key(match_date: date) -> str:
    if REFIT_FREQUENCY == "monthly":
        return f"{match_date.year:04d}-{match_date.month:02d}"
    return match_date.isoformat()


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
    rows.sort(key=lambda row: (parse_date(row["date"]), clean_str(row.get("tournament"))))

    test_start = parse_date(TEST_START)
    if not test_start:
        raise SystemExit(f"Invalid FIRST_SET_LOGIT_V2_TEST_START: {TEST_START}")

    prior_over, prior_win = global_priors(rows, test_start)

    states: dict[str, State] = defaultdict(State)
    train_x: dict[str, list[list[float]]] = defaultdict(list)
    train_y: dict[str, list[int]] = defaultdict(list)
    pooled_x: list[list[float]] = []
    pooled_y: list[int] = []

    models: dict[str, Pipeline] = {}
    current_refit_key = ""
    predictions: list[dict[str, Any]] = []
    counters = defaultdict(int)

    for row in rows:
        match_date = parse_date(row["date"])
        player_1 = clean_str(row["player_1"])
        player_2 = clean_str(row["player_2"])

        key_1 = player_key(player_1)
        key_2 = player_key(player_2)
        state_1 = states[key_1]
        state_2 = states[key_2]

        surface = clean_str(row.get("surface")).lower() or "unknown"
        gender = clean_str(row.get("gender")).lower() or "unknown"
        actual = int(int(row["first_set_games"]) > 9.5)
        p1_win, p2_win = first_set_side_result(row)
        tiebreak = int(bool(row.get("first_set_tiebreak")))

        eligible = (
            state_1.matches >= MIN_PLAYER_SAMPLE
            and state_2.matches >= MIN_PLAYER_SAMPLE
        )

        if eligible:
            features = build_feature_vector(
                row,
                state_1,
                state_2,
                prior_over,
                prior_win,
            )

            if match_date < test_start:
                train_x[gender].append(features)
                train_y[gender].append(actual)
                pooled_x.append(features)
                pooled_y.append(actual)
            else:
                key = refit_key(match_date)

                if key != current_refit_key:
                    current_refit_key = key
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
                    probability = float(model.predict_proba([features])[0][1])
                    favorite_probability, odds_balance, odds_missing = odds_features(row)
                    rank_gap_log, rank_missing = ranking_features(row)

                    predictions.append(
                        {
                            "date": match_date.isoformat(),
                            "tournament": clean_str(row.get("tournament")),
                            "tour_level": clean_str(row.get("tour_level")),
                            "gender": gender,
                            "surface": surface,
                            "player_1": player_1,
                            "player_2": player_2,
                            "first_set_score": clean_str(row.get("first_set_score")),
                            "first_set_games": int(row["first_set_games"]),
                            "actual_over_9_5": actual,
                            "probability_over_9_5": round(probability, 6),
                            "prediction": "over" if probability >= 0.5 else "under",
                            "correct": (probability >= 0.5) == bool(actual),
                            "model_group": gender if gender in models else "pooled",
                            "player_1_sample": state_1.matches,
                            "player_2_sample": state_2.matches,
                            "favorite_probability": round(favorite_probability, 6),
                            "odds_balance": round(odds_balance, 6),
                            "odds_missing": bool(odds_missing),
                            "rank_gap_log": round(rank_gap_log, 6),
                            "rank_missing": bool(rank_missing),
                        }
                    )

                    train_x[gender].append(features)
                    train_y[gender].append(actual)
                    pooled_x.append(features)
                    pooled_y.append(actual)
        else:
            counters["insufficient_sample"] += 1

        state_1.update(surface, actual, p1_win, int(row["first_set_games"]), tiebreak)
        state_2.update(surface, actual, p2_win, int(row["first_set_games"]), tiebreak)

    by_surface = {
        surface: summary([row for row in predictions if row["surface"] == surface])
        for surface in sorted({row["surface"] for row in predictions})
    }

    by_gender = {
        gender: summary([row for row in predictions if row["gender"] == gender])
        for gender in sorted({row["gender"] for row in predictions})
    }

    balanced_matches = [
        row for row in predictions
        if not row["odds_missing"] and row["favorite_probability"] <= 0.60
    ]
    strong_favorites = [
        row for row in predictions
        if not row["odds_missing"] and row["favorite_probability"] >= 0.75
    ]

    overall_summary = summary(predictions)
    overall_summary.update(
        {
            "test_start": test_start.isoformat(),
            "baseline_over_9_5": round(prior_over, 4),
            "test_matches": len(predictions),
            "train_rows_final": len(pooled_x),
        }
    )

    output = {
        "generated_at": now_iso(),
        "model": "first_set_over_9_5_walk_forward_logistic_v2_odds_rank",
        "source_file": str(ARCHIVE_FILE),
        "feature_names": FEATURE_NAMES,
        "settings": {
            "test_start": test_start.isoformat(),
            "min_player_sample": MIN_PLAYER_SAMPLE,
            "min_train_rows": MIN_TRAIN_ROWS,
            "recent_window": RECENT_WINDOW,
            "refit_frequency": REFIT_FREQUENCY,
        },
        "summary": overall_summary,
        "by_surface": by_surface,
        "by_gender": by_gender,
        "by_match_balance": {
            "balanced_favorite_prob_le_0_60": summary(balanced_matches),
            "strong_favorite_prob_ge_0_75": summary(strong_favorites),
        },
        "calibration": calibration(predictions),
        "counters": dict(counters),
        "predictions": predictions,
    }

    report = {
        "generated_at": now_iso(),
        "model": output["model"],
        "summary": overall_summary,
        "by_surface": by_surface,
        "by_gender": by_gender,
        "by_match_balance": output["by_match_balance"],
        "calibration": output["calibration"],
        "counters": dict(counters),
    }

    save_json(OUTPUT_FILE, output)
    save_json(REPORT_FILE, report)

    print("")
    print("FIRST SET OVER 9.5 LOGISTIC V2 BACKTEST DONE")
    print("SUMMARY:", overall_summary)
    print("BY SURFACE:", by_surface)
    print("BY GENDER:", by_gender)
    print("BY MATCH BALANCE:", output["by_match_balance"])
    print("CALIBRATION:", output["calibration"])
    print("COUNTERS:", dict(counters))
    print(f"Output: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")
    print("")


if __name__ == "__main__":
    main()
