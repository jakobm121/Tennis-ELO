import math
import os
from collections import defaultdict
from datetime import datetime
from statistics import mean
from typing import Any

from tennis_elo.config import CANONICAL_MATCHES_FILE, DEFAULT_ELO, K_FACTOR, ROOT_DIR, SURFACE_K_FACTOR
from tennis_elo.utils import canonical_player_name, clean_str, load_json, now_iso, save_json

OUTPUT_FILE = ROOT_DIR / "data" / "backtests" / "match_winner_elo_backtest.json"
REPORT_FILE = ROOT_DIR / "data" / "reports" / "match_winner_elo_backtest_report.json"
TEST_START = os.getenv("MATCH_WINNER_ELO_TEST_START", "2026-03-01")
MIN_PLAYER_MATCHES = int(os.getenv("MATCH_WINNER_ELO_MIN_PLAYER_MATCHES", "10"))
BLEND_OVERALL_WEIGHT = float(os.getenv("MATCH_WINNER_ELO_BLEND_OVERALL_WEIGHT", "0.60"))
BLEND_SURFACE_WEIGHT = float(os.getenv("MATCH_WINNER_ELO_BLEND_SURFACE_WEIGHT", "0.40"))
MODEL_NAMES = ("overall_elo", "surface_elo", "blended_elo")


def parse_date(raw: Any):
    value = clean_str(raw)
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d.%m.%Y", "%d.%m.%y", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def expected_score(a: float, b: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (b - a) / 400.0))


def update_pair(winner: float, loser: float, k: float):
    p = expected_score(winner, loser)
    return winner + k * (1.0 - p), loser - k * (1.0 - p)


def summarize(rows, model):
    key = f"{model}_p1_probability"
    probs = [float(r[key]) for r in rows]
    ys = [int(r["player_1_won"]) for r in rows]
    if not probs:
        return {"sample_size": 0, "accuracy": None, "brier_score": None, "log_loss": None, "calibration": []}
    accuracy = sum((p >= 0.5) == bool(y) for p, y in zip(probs, ys)) / len(probs)
    brier = mean((p - y) ** 2 for p, y in zip(probs, ys))
    losses = []
    for p, y in zip(probs, ys):
        p = min(0.999999, max(0.000001, p))
        losses.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))
    calibration = []
    for i in range(10):
        low, high = i / 10, (i + 1) / 10
        bucket = [(p, y) for p, y in zip(probs, ys) if low <= p < high or (high == 1.0 and p == 1.0)]
        if bucket:
            calibration.append({
                "bucket": f"{low:.1f}-{high:.1f}",
                "sample_size": len(bucket),
                "avg_predicted": round(mean(p for p, _ in bucket), 4),
                "actual_win_rate": round(mean(y for _, y in bucket), 4),
            })
    return {
        "sample_size": len(rows),
        "accuracy": round(accuracy, 4),
        "brier_score": round(brier, 6),
        "log_loss": round(mean(losses), 6),
        "avg_predicted_p1": round(mean(probs), 4),
        "actual_p1_win_rate": round(mean(ys), 4),
        "calibration": calibration,
    }


def player_state():
    return {"overall_elo": float(DEFAULT_ELO), "surface_elo": {}, "matches_total": 0, "surface_matches": defaultdict(int)}


def surface_rating(player, surface):
    return float(player["surface_elo"].get(surface, DEFAULT_ELO))


def blended_probability(p1o, p2o, p1s, p2s, p1n, p2n):
    po = expected_score(p1o, p2o)
    ps = expected_score(p1s, p2s)
    confidence = min(1.0, min(p1n, p2n) / 20.0)
    sw = BLEND_SURFACE_WEIGHT * confidence
    ow = BLEND_OVERALL_WEIGHT + BLEND_SURFACE_WEIGHT * (1.0 - confidence)
    return (po * ow + ps * sw) / (ow + sw)


def valid_match(row):
    winner, loser = clean_str(row.get("winner")), clean_str(row.get("loser"))
    return bool(parse_date(row.get("date")) and winner and loser and canonical_player_name(winner) != canonical_player_name(loser) and row.get("ready_for_elo") is not False)


def main():
    payload = load_json(CANONICAL_MATCHES_FILE, {})
    matches = payload.get("matches", []) if isinstance(payload, dict) else []
    matches = [r for r in matches if isinstance(r, dict) and valid_match(r)]
    matches.sort(key=lambda r: (parse_date(r.get("date")), clean_str(r.get("tournament") or r.get("tournament_code")), clean_str(r.get("canonical_match_id"))))
    test_start = parse_date(TEST_START)
    if not test_start:
        raise SystemExit(f"Invalid MATCH_WINNER_ELO_TEST_START: {TEST_START}")

    players = defaultdict(player_state)
    predictions = []
    counters = defaultdict(int)

    for row in matches:
        match_date = parse_date(row.get("date"))
        p1_name = clean_str(row.get("player_1")) or clean_str(row.get("winner"))
        p2_name = clean_str(row.get("player_2")) or clean_str(row.get("loser"))
        p1_key, p2_key = canonical_player_name(p1_name), canonical_player_name(p2_name)
        winner_key = canonical_player_name(row.get("winner"))
        if winner_key not in {p1_key, p2_key}:
            counters["winner_not_one_of_players"] += 1
            continue
        surface = clean_str(row.get("surface")).lower() or "unknown"
        p1, p2 = players[p1_key], players[p2_key]
        p1o, p2o = float(p1["overall_elo"]), float(p2["overall_elo"])
        p1s, p2s = surface_rating(p1, surface), surface_rating(p2, surface)
        p1sn, p2sn = int(p1["surface_matches"][surface]), int(p2["surface_matches"][surface])
        overall_p = expected_score(p1o, p2o)
        surface_p = expected_score(p1s, p2s)
        blended_p = blended_probability(p1o, p2o, p1s, p2s, p1sn, p2sn)

        if match_date >= test_start and p1["matches_total"] >= MIN_PLAYER_MATCHES and p2["matches_total"] >= MIN_PLAYER_MATCHES:
            predictions.append({
                "date": match_date.isoformat(), "tournament": row.get("tournament") or row.get("tournament_code") or "", "surface": surface,
                "player_1": p1_name, "player_2": p2_name, "winner": row.get("winner"), "player_1_won": int(winner_key == p1_key),
                "player_1_matches_before": p1["matches_total"], "player_2_matches_before": p2["matches_total"],
                "player_1_surface_matches_before": p1sn, "player_2_surface_matches_before": p2sn,
                "player_1_overall_elo_before": round(p1o, 3), "player_2_overall_elo_before": round(p2o, 3),
                "player_1_surface_elo_before": round(p1s, 3), "player_2_surface_elo_before": round(p2s, 3),
                "overall_elo_p1_probability": round(overall_p, 6), "surface_elo_p1_probability": round(surface_p, 6),
                "blended_elo_p1_probability": round(blended_p, 6), "canonical_match_id": row.get("canonical_match_id"),
            })
        elif match_date >= test_start:
            counters["insufficient_player_history"] += 1

        if winner_key == p1_key:
            winner, loser, ws, ls = p1, p2, p1s, p2s
        else:
            winner, loser, ws, ls = p2, p1, p2s, p1s
        winner["overall_elo"], loser["overall_elo"] = update_pair(float(winner["overall_elo"]), float(loser["overall_elo"]), K_FACTOR)
        winner["surface_elo"][surface], loser["surface_elo"][surface] = update_pair(ws, ls, SURFACE_K_FACTOR)
        p1["matches_total"] += 1; p2["matches_total"] += 1
        p1["surface_matches"][surface] += 1; p2["surface_matches"][surface] += 1

    summaries = {m: summarize(predictions, m) for m in MODEL_NAMES}
    by_surface = {}
    for surface in sorted({r["surface"] for r in predictions}):
        subset = [r for r in predictions if r["surface"] == surface]
        by_surface[surface] = {m: summarize(subset, m) for m in MODEL_NAMES}
    ranking = sorted(MODEL_NAMES, key=lambda m: (summaries[m]["brier_score"] if summaries[m]["brier_score"] is not None else 999, summaries[m]["log_loss"] if summaries[m]["log_loss"] is not None else 999, -(summaries[m]["accuracy"] or 0)))

    output = {
        "generated_at": now_iso(), "model": "match_winner_elo_walk_forward_v1", "source_file": str(CANONICAL_MATCHES_FILE),
        "settings": {"test_start": test_start.isoformat(), "default_elo": DEFAULT_ELO, "overall_k_factor": K_FACTOR, "surface_k_factor": SURFACE_K_FACTOR, "min_player_matches": MIN_PLAYER_MATCHES, "blend_overall_weight": BLEND_OVERALL_WEIGHT, "blend_surface_weight": BLEND_SURFACE_WEIGHT},
        "counts": {"input_matches": len(matches), "test_predictions": len(predictions), "players_final": len(players), **dict(counters)},
        "best_model": ranking[0] if ranking else None, "model_ranking": ranking, "summary": summaries, "by_surface": by_surface, "predictions": predictions,
    }
    report = {k: output[k] for k in ("generated_at", "model", "settings", "counts", "best_model", "model_ranking", "summary", "by_surface")}
    save_json(OUTPUT_FILE, output); save_json(REPORT_FILE, report)
    print("\nMATCH WINNER ELO BACKTEST DONE")
    print("COUNTS:", output["counts"]); print("BEST MODEL:", output["best_model"]); print("MODEL RANKING:", ranking)
    for model in MODEL_NAMES: print(model.upper() + ":", summaries[model])
    print("BY SURFACE:", by_surface); print(f"Output: {OUTPUT_FILE}"); print(f"Report: {REPORT_FILE}\n")


if __name__ == "__main__":
    main()
