import math
import os
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from tennis_elo.config import ROOT_DIR
from tennis_elo.utils import clean_str, load_json, now_iso, save_json


SOURCE_FILE = ROOT_DIR / "data" / "backtests" / "first_set_over_9_5_logistic_v2_backtest.json"
OUTPUT_FILE = ROOT_DIR / "data" / "backtests" / "first_set_under_9_5_strong_favorites.json"
REPORT_FILE = ROOT_DIR / "data" / "reports" / "first_set_under_9_5_strong_favorites_report.json"

THRESHOLDS = [
    float(x.strip())
    for x in os.getenv(
        "STRONG_FAVORITE_THRESHOLDS",
        "0.70,0.75,0.80,0.85",
    ).split(",")
    if x.strip()
]

MIN_SAMPLE = int(os.getenv("STRONG_FAVORITE_MIN_SAMPLE", "30"))


def parse_date(raw: Any):
    value = clean_str(raw)
    if not value:
        return None

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def fair_odds(probability: float | None) -> float | None:
    if probability is None or probability <= 0:
        return None
    return round(1.0 / probability, 3)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample = len(rows)

    if sample == 0:
        return {
            "sample_size": 0,
            "under_hit_rate": None,
            "over_hit_rate": None,
            "fair_odds_under": None,
            "avg_favorite_probability": None,
            "avg_model_over_probability": None,
            "avg_model_under_probability": None,
        }

    under_rate = mean(1 - int(row["actual_over_9_5"]) for row in rows)
    over_rate = 1.0 - under_rate

    return {
        "sample_size": sample,
        "under_hits": sum(1 - int(row["actual_over_9_5"]) for row in rows),
        "over_hits": sum(int(row["actual_over_9_5"]) for row in rows),
        "under_hit_rate": round(under_rate, 4),
        "over_hit_rate": round(over_rate, 4),
        "fair_odds_under": fair_odds(under_rate),
        "avg_favorite_probability": round(
            mean(float(row["favorite_probability"]) for row in rows),
            4,
        ),
        "avg_model_over_probability": round(
            mean(float(row["probability_over_9_5"]) for row in rows),
            4,
        ),
        "avg_model_under_probability": round(
            mean(1.0 - float(row["probability_over_9_5"]) for row in rows),
            4,
        ),
    }


def breakdown(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    values = sorted({clean_str(row.get(key)) or "unknown" for row in rows})

    return {
        value: summarize(
            [
                row for row in rows
                if (clean_str(row.get(key)) or "unknown") == value
            ]
        )
        for value in values
    }


def rank_gap_bucket(row: dict[str, Any]) -> str:
    gap = row.get("rank_gap_log")

    try:
        gap = float(gap)
    except (TypeError, ValueError):
        return "missing"

    if row.get("rank_missing"):
        return "missing"

    # rank_gap_log = log1p(abs(rank_1-rank_2))
    raw_gap = math.expm1(gap)

    if raw_gap < 25:
        return "0-24"
    if raw_gap < 50:
        return "25-49"
    if raw_gap < 100:
        return "50-99"
    return "100+"


def threshold_report(
    predictions: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    selected = [
        row for row in predictions
        if not row.get("odds_missing")
        and float(row.get("favorite_probability") or 0) >= threshold
    ]

    by_rank_gap: dict[str, dict[str, Any]] = {}
    for bucket in ("0-24", "25-49", "50-99", "100+", "missing"):
        subset = [row for row in selected if rank_gap_bucket(row) == bucket]
        if subset:
            by_rank_gap[bucket] = summarize(subset)

    return {
        "threshold": threshold,
        "eligible": len(selected) >= MIN_SAMPLE,
        "overall": summarize(selected),
        "by_gender": breakdown(selected, "gender"),
        "by_surface": breakdown(selected, "surface"),
        "by_tour_level": breakdown(selected, "tour_level"),
        "by_rank_gap": by_rank_gap,
        "matches": selected,
    }


def main() -> None:
    payload = load_json(SOURCE_FILE, {})
    predictions = payload.get("predictions", []) if isinstance(payload, dict) else []

    if not isinstance(predictions, list) or not predictions:
        raise SystemExit(
            "No predictions found in logistic V2 backtest. "
            "Run backtest_first_set_over_9_5_logistic_v2 first."
        )

    results = [
        threshold_report(predictions, threshold)
        for threshold in sorted(THRESHOLDS)
    ]

    eligible = [
        item for item in results
        if item["overall"]["sample_size"] >= MIN_SAMPLE
        and item["overall"]["under_hit_rate"] is not None
    ]

    best = None
    if eligible:
        best = max(
            eligible,
            key=lambda item: (
                item["overall"]["under_hit_rate"],
                item["overall"]["sample_size"],
            ),
        )

    output = {
        "generated_at": now_iso(),
        "source_file": str(SOURCE_FILE),
        "market": "first_set_under_9_5",
        "settings": {
            "thresholds": THRESHOLDS,
            "minimum_sample": MIN_SAMPLE,
        },
        "best_threshold": (
            {
                "threshold": best["threshold"],
                **best["overall"],
            }
            if best
            else None
        ),
        "threshold_results": results,
    }

    report = {
        "generated_at": now_iso(),
        "source_file": str(SOURCE_FILE),
        "market": "first_set_under_9_5",
        "settings": output["settings"],
        "best_threshold": output["best_threshold"],
        "threshold_summary": [
            {
                "threshold": item["threshold"],
                "eligible": item["eligible"],
                **item["overall"],
            }
            for item in results
        ],
    }

    save_json(OUTPUT_FILE, output)
    save_json(REPORT_FILE, report)

    print("")
    print("FIRST SET UNDER 9.5 STRONG FAVORITES BACKTEST DONE")
    print("BEST:", output["best_threshold"])

    for item in report["threshold_summary"]:
        print(
            "THRESHOLD:",
            item["threshold"],
            "sample=", item["sample_size"],
            "under_rate=", item["under_hit_rate"],
            "fair_odds=", item["fair_odds_under"],
        )

    print(f"Output: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")
    print("")


if __name__ == "__main__":
    main()
