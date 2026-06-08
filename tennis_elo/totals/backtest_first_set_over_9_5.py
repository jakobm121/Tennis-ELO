import json
import math
import os
import re
from collections import defaultdict, deque
from datetime import date, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from tennis_elo.config import ROOT_DIR
from tennis_elo.utils import clean_str, load_json, now_iso, save_json


ARCHIVE_FILE = ROOT_DIR / "data" / "totals" / "scorelines_archive.json"
OUTPUT_FILE = ROOT_DIR / "data" / "backtests" / "first_set_over_9_5_backtest.json"
REPORT_FILE = ROOT_DIR / "data" / "reports" / "first_set_over_9_5_backtest_report.json"

TEST_START = os.getenv("FIRST_SET_BACKTEST_START", "2026-03-01")
MIN_PLAYER_SAMPLE = int(os.getenv("FIRST_SET_MIN_PLAYER_SAMPLE", "15"))
MIN_SURFACE_SAMPLE = int(os.getenv("FIRST_SET_MIN_SURFACE_SAMPLE", "8"))
RECENT_WINDOW = int(os.getenv("FIRST_SET_RECENT_WINDOW", "10"))

# Blend weights.
WEIGHT_OVERALL = float(os.getenv("FIRST_SET_WEIGHT_OVERALL", "0.45"))
WEIGHT_SURFACE = float(os.getenv("FIRST_SET_WEIGHT_SURFACE", "0.30"))
WEIGHT_RECENT = float(os.getenv("FIRST_SET_WEIGHT_RECENT", "0.15"))
WEIGHT_BALANCE = float(os.getenv("FIRST_SET_WEIGHT_BALANCE", "0.10"))

BASELINE_OVER_9_5 = float(os.getenv("FIRST_SET_BASELINE_OVER_9_5", "0.50"))


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
    return min(0.95, max(0.05, value))


def safe_rate(successes: int, sample: int, prior: float = BASELINE_OVER_9_5, strength: float = 8.0) -> float:
    # Bayesian shrinkage toward a league-wide baseline.
    return (successes + prior * strength) / (sample + strength)


def brier_score(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return round(mean((row["predicted_over_9_5"] - row["actual_over_9_5"]) ** 2 for row in rows), 6)


def log_loss(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None

    values = []
    for row in rows:
        p = min(0.999999, max(0.000001, row["predicted_over_9_5"]))
        y = row["actual_over_9_5"]
        values.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))

    return round(mean(values), 6)


def accuracy(rows: list[dict[str, Any]], threshold: float = 0.5) -> float | None:
    if not rows:
        return None

    correct = sum(
        (row["predicted_over_9_5"] >= threshold) == bool(row["actual_over_9_5"])
        for row in rows
    )
    return round(correct / len(rows), 4)


def calibration_buckets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets = []

    for low in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        high = low + 0.1
        bucket_rows = [
            row for row in rows
            if low <= row["predicted_over_9_5"] < high
            or (high >= 1.0 and row["predicted_over_9_5"] == 1.0)
        ]

        if not bucket_rows:
            continue

        buckets.append(
            {
                "bucket": f"{low:.1f}-{high:.1f}",
                "sample_size": len(bucket_rows),
                "avg_predicted": round(mean(r["predicted_over_9_5"] for r in bucket_rows), 4),
                "actual_over_rate": round(mean(r["actual_over_9_5"] for r in bucket_rows), 4),
            }
        )

    return buckets


class PlayerState:
    def __init__(self) -> None:
        self.total_matches = 0
        self.total_overs = 0
        self.surface_matches = defaultdict(int)
        self.surface_overs = defaultdict(int)
        self.recent = deque(maxlen=RECENT_WINDOW)

    def add(self, surface: str, actual_over: int) -> None:
        self.total_matches += 1
        self.total_overs += actual_over
        self.surface_matches[surface] += 1
        self.surface_overs[surface] += actual_over
        self.recent.append(actual_over)

    def overall_rate(self, prior: float) -> float:
        return safe_rate(self.total_overs, self.total_matches, prior=prior)

    def surface_rate(self, surface: str, prior: float) -> float:
        return safe_rate(
            self.surface_overs[surface],
            self.surface_matches[surface],
            prior=prior,
        )

    def recent_rate(self, prior: float) -> float:
        if not self.recent:
            return prior
        return safe_rate(sum(self.recent), len(self.recent), prior=prior, strength=4.0)


def valid_row(row: dict[str, Any]) -> tuple[bool, str]:
    if row.get("retired") or row.get("walkover"):
        return False, "retired_or_walkover"

    match_date = parse_date(row.get("date"))
    if not match_date:
        return False, "invalid_date"

    if not clean_str(row.get("player_1")) or not clean_str(row.get("player_2")):
        return False, "missing_players"

    try:
        first_games = int(row.get("first_set_games"))
    except (TypeError, ValueError):
        return False, "missing_first_set_games"

    if first_games < 6 or first_games > 20:
        return False, "invalid_first_set_games"

    return True, "ok"


def global_baseline(rows: list[dict[str, Any]], cutoff: date) -> float:
    training = []
    for row in rows:
        d = parse_date(row.get("date"))
        if not d or d >= cutoff:
            continue
        try:
            training.append(int(row.get("first_set_games")) > 9.5)
        except (TypeError, ValueError):
            continue

    if not training:
        return BASELINE_OVER_9_5

    return sum(training) / len(training)


def matchup_balance_signal(state_a: PlayerState, state_b: PlayerState, baseline: float) -> float:
    """
    Totals-only proxy:
    If both players have similar historical first-set over rates, matchup is treated
    as more balanced. Extreme disagreement is shrunk toward baseline.
    """
    rate_a = state_a.overall_rate(baseline)
    rate_b = state_b.overall_rate(baseline)
    disagreement = abs(rate_a - rate_b)

    return clip_probability(((rate_a + rate_b) / 2) - 0.25 * disagreement)


def predict_match(
    state_a: PlayerState,
    state_b: PlayerState,
    surface: str,
    baseline: float,
) -> tuple[float, dict[str, Any]] | tuple[None, dict[str, Any]]:
    if state_a.total_matches < MIN_PLAYER_SAMPLE or state_b.total_matches < MIN_PLAYER_SAMPLE:
        return None, {
            "reason": "insufficient_player_sample",
            "player_1_sample": state_a.total_matches,
            "player_2_sample": state_b.total_matches,
        }

    overall = (
        state_a.overall_rate(baseline) + state_b.overall_rate(baseline)
    ) / 2

    surface_usable = (
        state_a.surface_matches[surface] >= MIN_SURFACE_SAMPLE
        and state_b.surface_matches[surface] >= MIN_SURFACE_SAMPLE
    )

    if surface_usable:
        surface_rate = (
            state_a.surface_rate(surface, baseline)
            + state_b.surface_rate(surface, baseline)
        ) / 2
    else:
        surface_rate = overall

    recent = (
        state_a.recent_rate(baseline) + state_b.recent_rate(baseline)
    ) / 2

    balance = matchup_balance_signal(state_a, state_b, baseline)

    predicted = (
        WEIGHT_OVERALL * overall
        + WEIGHT_SURFACE * surface_rate
        + WEIGHT_RECENT * recent
        + WEIGHT_BALANCE * balance
    )

    return clip_probability(predicted), {
        "player_1_sample": state_a.total_matches,
        "player_2_sample": state_b.total_matches,
        "player_1_surface_sample": state_a.surface_matches[surface],
        "player_2_surface_sample": state_b.surface_matches[surface],
        "surface_used": surface_usable,
        "components": {
            "overall": round(overall, 4),
            "surface": round(surface_rate, 4),
            "recent": round(recent, 4),
            "balance": round(balance, 4),
        },
    }


def main() -> None:
    payload = load_json(ARCHIVE_FILE, {})
    rows = payload.get("matches", []) if isinstance(payload, dict) else payload

    if not isinstance(rows, list):
        rows = []

    counters = defaultdict(int)
    valid_rows = []

    for row in rows:
        ok, reason = valid_row(row)
        counters[reason] += 1
        if ok:
            valid_rows.append(row)

    valid_rows.sort(key=lambda row: (parse_date(row.get("date")), clean_str(row.get("tournament"))))

    test_start = parse_date(TEST_START)
    if not test_start:
        raise SystemExit(f"Invalid FIRST_SET_BACKTEST_START: {TEST_START}")

    baseline = global_baseline(valid_rows, test_start)

    player_states: dict[str, PlayerState] = defaultdict(PlayerState)
    predictions = []

    for row in valid_rows:
        match_date = parse_date(row.get("date"))
        if not match_date:
            continue

        p1 = clean_str(row.get("player_1"))
        p2 = clean_str(row.get("player_2"))
        key_1 = player_key(p1)
        key_2 = player_key(p2)
        surface = clean_str(row.get("surface")).lower() or "unknown"
        actual = 1 if int(row.get("first_set_games")) > 9.5 else 0

        state_1 = player_states[key_1]
        state_2 = player_states[key_2]

        # Strict walk-forward: predict before updating with this match.
        if match_date >= test_start:
            predicted, details = predict_match(state_1, state_2, surface, baseline)

            if predicted is None:
                counters[details["reason"]] += 1
            else:
                predictions.append(
                    {
                        "date": match_date.isoformat(),
                        "tournament": clean_str(row.get("tournament")),
                        "tour_level": clean_str(row.get("tour_level")),
                        "gender": clean_str(row.get("gender")),
                        "surface": surface,
                        "player_1": p1,
                        "player_2": p2,
                        "first_set_score": clean_str(row.get("first_set_score")),
                        "first_set_games": int(row.get("first_set_games")),
                        "actual_over_9_5": actual,
                        "predicted_over_9_5": round(predicted, 6),
                        "predicted_under_9_5": round(1 - predicted, 6),
                        "prediction": "over" if predicted >= 0.5 else "under",
                        "correct": (predicted >= 0.5) == bool(actual),
                        "details": details,
                    }
                )

        state_1.add(surface, actual)
        state_2.add(surface, actual)

    by_surface = {}
    for surface in sorted({row["surface"] for row in predictions}):
        subset = [row for row in predictions if row["surface"] == surface]
        by_surface[surface] = {
            "sample_size": len(subset),
            "accuracy": accuracy(subset),
            "brier_score": brier_score(subset),
            "log_loss": log_loss(subset),
            "actual_over_rate": round(mean(r["actual_over_9_5"] for r in subset), 4),
            "avg_predicted_over": round(mean(r["predicted_over_9_5"] for r in subset), 4),
        }

    by_gender = {}
    for gender in sorted({row["gender"] for row in predictions}):
        subset = [row for row in predictions if row["gender"] == gender]
        by_gender[gender or "unknown"] = {
            "sample_size": len(subset),
            "accuracy": accuracy(subset),
            "brier_score": brier_score(subset),
            "log_loss": log_loss(subset),
        }

    summary = {
        "test_start": test_start.isoformat(),
        "baseline_over_9_5": round(baseline, 4),
        "test_matches": len(predictions),
        "accuracy": accuracy(predictions),
        "brier_score": brier_score(predictions),
        "log_loss": log_loss(predictions),
        "actual_over_rate": (
            round(mean(row["actual_over_9_5"] for row in predictions), 4)
            if predictions else None
        ),
        "avg_predicted_over": (
            round(mean(row["predicted_over_9_5"] for row in predictions), 4)
            if predictions else None
        ),
    }

    output = {
        "generated_at": now_iso(),
        "model": "first_set_over_9_5_walk_forward_v1",
        "source_file": str(ARCHIVE_FILE),
        "settings": {
            "test_start": test_start.isoformat(),
            "min_player_sample": MIN_PLAYER_SAMPLE,
            "min_surface_sample": MIN_SURFACE_SAMPLE,
            "recent_window": RECENT_WINDOW,
            "weights": {
                "overall": WEIGHT_OVERALL,
                "surface": WEIGHT_SURFACE,
                "recent": WEIGHT_RECENT,
                "balance": WEIGHT_BALANCE,
            },
        },
        "summary": summary,
        "by_surface": by_surface,
        "by_gender": by_gender,
        "calibration": calibration_buckets(predictions),
        "predictions": predictions,
    }

    report = {
        "generated_at": now_iso(),
        "model": output["model"],
        "source_file": str(ARCHIVE_FILE),
        "output_file": str(OUTPUT_FILE),
        "summary": summary,
        "parse_counts": dict(counters),
        "by_surface": by_surface,
        "by_gender": by_gender,
        "calibration": output["calibration"],
    }

    save_json(OUTPUT_FILE, output)
    save_json(REPORT_FILE, report)

    print("")
    print("FIRST SET OVER 9.5 BACKTEST DONE")
    print(summary)
    print("Parse:", dict(counters))
    print(f"Output: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")
    print("")


if __name__ == "__main__":
    main()
