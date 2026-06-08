import math
import os
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from tennis_elo.config import ROOT_DIR
from tennis_elo.utils import clean_str, load_json, now_iso, save_json


ARCHIVE_FILE = ROOT_DIR / "data" / "totals" / "scorelines_archive.json"
OUTPUT_FILE = ROOT_DIR / "data" / "backtests" / "all_first_set_totals_backtest.json"
REPORT_FILE = ROOT_DIR / "data" / "reports" / "all_first_set_totals_backtest_report.json"

VALIDATION_START = os.getenv(
    "FIRST_SET_ALL_TOTALS_VALIDATION_START",
    "2026-03-01",
)
LINES = [
    float(value.strip())
    for value in os.getenv(
        "FIRST_SET_ALL_TOTALS_LINES",
        "8.5,9.5,10.5,11.5",
    ).split(",")
    if value.strip()
]
MIN_DISCOVERY_SAMPLE = int(
    os.getenv("FIRST_SET_ALL_TOTALS_MIN_DISCOVERY_SAMPLE", "100")
)
MIN_VALIDATION_SAMPLE = int(
    os.getenv("FIRST_SET_ALL_TOTALS_MIN_VALIDATION_SAMPLE", "40")
)
MAX_STABILITY_GAP = float(
    os.getenv("FIRST_SET_ALL_TOTALS_MAX_STABILITY_GAP", "0.10")
)
TOP_SEGMENTS = int(
    os.getenv("FIRST_SET_ALL_TOTALS_TOP_SEGMENTS", "100")
)


def parse_date(raw: Any) -> date | None:
    value = clean_str(raw)

    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%d.%m.%Y",
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


def numeric(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def valid_row(row: dict[str, Any]) -> bool:
    if row.get("retired") or row.get("walkover"):
        return False

    if not parse_date(row.get("date")):
        return False

    try:
        games = int(row.get("first_set_games"))
    except (TypeError, ValueError):
        return False

    return 6 <= games <= 20


def favorite_probability(row: dict[str, Any]) -> float | None:
    bookmaker = row.get("bookmaker") or {}
    odd_1 = numeric(bookmaker.get("player_1_odds"))
    odd_2 = numeric(bookmaker.get("player_2_odds"))

    if not odd_1 or not odd_2 or odd_1 <= 1 or odd_2 <= 1:
        return None

    raw_1 = 1 / odd_1
    raw_2 = 1 / odd_2
    total = raw_1 + raw_2

    if total <= 0:
        return None

    return max(raw_1 / total, raw_2 / total)


def wilson_interval(
    successes: int,
    sample: int,
    z: float = 1.96,
) -> tuple[float | None, float | None]:
    if sample <= 0:
        return None, None

    p = successes / sample
    denominator = 1 + (z * z / sample)
    center = (
        p + z * z / (2 * sample)
    ) / denominator
    margin = (
        z
        * math.sqrt(
            (p * (1 - p) / sample)
            + (z * z / (4 * sample * sample))
        )
        / denominator
    )

    return max(0.0, center - margin), min(1.0, center + margin)


def fair_odds(hit_rate: float | None) -> float | None:
    if hit_rate is None or hit_rate <= 0:
        return None
    return round(1 / hit_rate, 3)


def hit(first_set_games: int, side: str, line: float) -> int:
    if side == "over":
        return int(first_set_games > line)
    return int(first_set_games < line)


def base_segment_definitions() -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = [
        {
            "segment_id": "all",
            "label": "All matches",
            "filters": {},
        }
    ]

    for gender in ("men", "women"):
        definitions.append(
            {
                "segment_id": f"gender:{gender}",
                "label": gender,
                "filters": {"gender": gender},
            }
        )

    for surface in ("hard", "clay", "grass"):
        definitions.append(
            {
                "segment_id": f"surface:{surface}",
                "label": surface,
                "filters": {"surface": surface},
            }
        )

    for threshold in (0.70, 0.75, 0.80, 0.85):
        definitions.append(
            {
                "segment_id": f"favorite_ge:{threshold:.2f}",
                "label": f"Favorite >= {threshold:.2f}",
                "filters": {"favorite_probability_ge": threshold},
            }
        )

    for threshold in (0.55, 0.60, 0.65):
        definitions.append(
            {
                "segment_id": f"favorite_le:{threshold:.2f}",
                "label": f"Favorite <= {threshold:.2f}",
                "filters": {"favorite_probability_le": threshold},
            }
        )

    for gender in ("men", "women"):
        for threshold in (0.75, 0.80, 0.85):
            definitions.append(
                {
                    "segment_id": (
                        f"gender:{gender}|favorite_ge:{threshold:.2f}"
                    ),
                    "label": (
                        f"{gender}, favorite >= {threshold:.2f}"
                    ),
                    "filters": {
                        "gender": gender,
                        "favorite_probability_ge": threshold,
                    },
                }
            )

    for surface in ("hard", "clay", "grass"):
        for threshold in (0.75, 0.80, 0.85):
            definitions.append(
                {
                    "segment_id": (
                        f"surface:{surface}|favorite_ge:{threshold:.2f}"
                    ),
                    "label": (
                        f"{surface}, favorite >= {threshold:.2f}"
                    ),
                    "filters": {
                        "surface": surface,
                        "favorite_probability_ge": threshold,
                    },
                }
            )

    for gender in ("men", "women"):
        for surface in ("hard", "clay", "grass"):
            definitions.append(
                {
                    "segment_id": (
                        f"gender:{gender}|surface:{surface}"
                    ),
                    "label": f"{gender}, {surface}",
                    "filters": {
                        "gender": gender,
                        "surface": surface,
                    },
                }
            )

    return definitions


def row_matches_segment(
    row: dict[str, Any],
    segment: dict[str, Any],
) -> bool:
    filters = segment.get("filters") or {}

    gender = clean_str(row.get("gender")).lower()
    surface = clean_str(row.get("surface")).lower()
    favorite = row.get("_favorite_probability")

    if filters.get("gender") and gender != filters["gender"]:
        return False

    if filters.get("surface") and surface != filters["surface"]:
        return False

    if "favorite_probability_ge" in filters:
        if favorite is None:
            return False
        if favorite < filters["favorite_probability_ge"]:
            return False

    if "favorite_probability_le" in filters:
        if favorite is None:
            return False
        if favorite > filters["favorite_probability_le"]:
            return False

    return True


def summarize_market(
    rows: list[dict[str, Any]],
    side: str,
    line: float,
) -> dict[str, Any]:
    sample = len(rows)

    if sample == 0:
        return {
            "sample_size": 0,
            "hits": 0,
            "hit_rate": None,
            "fair_odds": None,
            "wilson_low": None,
            "wilson_high": None,
        }

    hits = sum(
        hit(int(row["first_set_games"]), side, line)
        for row in rows
    )
    hit_rate = hits / sample
    low, high = wilson_interval(hits, sample)

    return {
        "sample_size": sample,
        "hits": hits,
        "hit_rate": round(hit_rate, 4),
        "fair_odds": fair_odds(hit_rate),
        "wilson_low": round(low, 4) if low is not None else None,
        "wilson_high": round(high, 4) if high is not None else None,
        "avg_first_set_games": round(
            mean(int(row["first_set_games"]) for row in rows),
            3,
        ),
    }


def global_baselines(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}

    for line in LINES:
        line_key = f"{line:.1f}"
        output[line_key] = {}

        for side in ("over", "under"):
            summary = summarize_market(rows, side, line)
            output[line_key][side] = (
                summary["hit_rate"]
                if summary["hit_rate"] is not None
                else 0.0
            )

    return output


def main() -> None:
    payload = load_json(ARCHIVE_FILE, {})
    rows = payload.get("matches", []) if isinstance(payload, dict) else payload

    if not isinstance(rows, list):
        rows = []

    rows = [
        row for row in rows
        if isinstance(row, dict) and valid_row(row)
    ]

    validation_start = parse_date(VALIDATION_START)

    if not validation_start:
        raise SystemExit(
            f"Invalid validation start: {VALIDATION_START}"
        )

    prepared = []

    for row in rows:
        item = dict(row)
        item["_favorite_probability"] = favorite_probability(row)
        prepared.append(item)

    discovery_rows = [
        row for row in prepared
        if parse_date(row["date"]) < validation_start
    ]
    validation_rows = [
        row for row in prepared
        if parse_date(row["date"]) >= validation_start
    ]

    discovery_baseline = global_baselines(discovery_rows)
    validation_baseline = global_baselines(validation_rows)
    segments = base_segment_definitions()

    results: list[dict[str, Any]] = []

    for line in LINES:
        line_key = f"{line:.1f}"

        for side in ("over", "under"):
            for segment in segments:
                discovery_subset = [
                    row for row in discovery_rows
                    if row_matches_segment(row, segment)
                ]
                validation_subset = [
                    row for row in validation_rows
                    if row_matches_segment(row, segment)
                ]

                discovery = summarize_market(
                    discovery_subset,
                    side,
                    line,
                )
                validation = summarize_market(
                    validation_subset,
                    side,
                    line,
                )

                discovery_rate = discovery["hit_rate"]
                validation_rate = validation["hit_rate"]

                stability_gap = None
                if (
                    discovery_rate is not None
                    and validation_rate is not None
                ):
                    stability_gap = abs(
                        discovery_rate - validation_rate
                    )

                validation_base = validation_baseline[line_key][side]
                lift = None

                if validation_rate is not None:
                    lift = validation_rate - validation_base

                eligible = (
                    discovery["sample_size"]
                    >= MIN_DISCOVERY_SAMPLE
                    and validation["sample_size"]
                    >= MIN_VALIDATION_SAMPLE
                    and stability_gap is not None
                    and stability_gap <= MAX_STABILITY_GAP
                )

                results.append(
                    {
                        "market": (
                            f"first_set_{side}_{line_key}"
                        ),
                        "side": side,
                        "line": line,
                        "segment_id": segment["segment_id"],
                        "segment_label": segment["label"],
                        "segment_filters": segment["filters"],
                        "eligible": eligible,
                        "discovery": discovery,
                        "validation": validation,
                        "stability_gap": (
                            round(stability_gap, 4)
                            if stability_gap is not None
                            else None
                        ),
                        "validation_baseline_hit_rate": round(
                            validation_base,
                            4,
                        ),
                        "validation_lift": (
                            round(lift, 4)
                            if lift is not None
                            else None
                        ),
                    }
                )

    eligible_results = [
        row for row in results if row["eligible"]
    ]

    eligible_results.sort(
        key=lambda row: (
            row["validation"]["wilson_low"] or 0,
            row["validation_lift"] or -999,
            row["validation"]["sample_size"],
        ),
        reverse=True,
    )

    best_by_market: dict[str, dict[str, Any]] = {}

    for row in eligible_results:
        if row["market"] not in best_by_market:
            best_by_market[row["market"]] = row

    output = {
        "generated_at": now_iso(),
        "model": "all_first_set_totals_segment_backtest_v1",
        "source_file": str(ARCHIVE_FILE),
        "settings": {
            "validation_start": validation_start.isoformat(),
            "lines": LINES,
            "min_discovery_sample": MIN_DISCOVERY_SAMPLE,
            "min_validation_sample": MIN_VALIDATION_SAMPLE,
            "max_stability_gap": MAX_STABILITY_GAP,
        },
        "counts": {
            "all_valid_matches": len(prepared),
            "discovery_matches": len(discovery_rows),
            "validation_matches": len(validation_rows),
            "segments_tested": len(segments),
            "market_segment_tests": len(results),
            "eligible_segments": len(eligible_results),
        },
        "baselines": {
            "discovery": discovery_baseline,
            "validation": validation_baseline,
        },
        "best_by_market": best_by_market,
        "best_segments": eligible_results[:TOP_SEGMENTS],
        "all_results": results,
    }

    report = {
        "generated_at": now_iso(),
        "model": output["model"],
        "settings": output["settings"],
        "counts": output["counts"],
        "best_by_market": best_by_market,
        "top_25_segments": eligible_results[:25],
    }

    save_json(OUTPUT_FILE, output)
    save_json(REPORT_FILE, report)

    print("")
    print("ALL FIRST SET TOTALS BACKTEST DONE")
    print("COUNTS:", output["counts"])

    print("\nBEST BY MARKET:")

    for market, row in best_by_market.items():
        validation = row["validation"]

        print(
            market,
            "| segment=", row["segment_label"],
            "| discovery_n=", row["discovery"]["sample_size"],
            "| discovery_rate=", row["discovery"]["hit_rate"],
            "| validation_n=", validation["sample_size"],
            "| validation_rate=", validation["hit_rate"],
            "| fair_odds=", validation["fair_odds"],
            "| wilson_low=", validation["wilson_low"],
            "| lift=", row["validation_lift"],
            "| stability_gap=", row["stability_gap"],
        )

    print("\nTOP 20 SEGMENTS:")

    for row in eligible_results[:20]:
        validation = row["validation"]

        print(
            row["market"],
            "|", row["segment_label"],
            "| n=", validation["sample_size"],
            "| hit=", validation["hit_rate"],
            "| fair=", validation["fair_odds"],
            "| wilson_low=", validation["wilson_low"],
            "| lift=", row["validation_lift"],
            "| gap=", row["stability_gap"],
        )

    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")
    print("")


if __name__ == "__main__":
    main()
