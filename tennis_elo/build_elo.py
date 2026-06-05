import math
from collections import deque

from tennis_elo.config import (
    CANONICAL_MATCHES_FILE,
    PLAYER_RATINGS_FILE,
    ELO_REPORT_FILE,
    DEFAULT_ELO,
    K_FACTOR,
    SURFACE_K_FACTOR,
)
from tennis_elo.utils import load_json, save_json, save_text, now_iso, clean_str, canonical_player_name


RECENT_WINDOW = 10


def expected_score(rating_a, rating_b):
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def update_elo(winner_rating, loser_rating, k):
    winner_expected = expected_score(winner_rating, loser_rating)
    loser_expected = expected_score(loser_rating, winner_rating)

    new_winner = winner_rating + k * (1.0 - winner_expected)
    new_loser = loser_rating + k * (0.0 - loser_expected)

    return new_winner, new_loser


def confidence(matches_total):
    """
    Simple confidence score from 0 to 1.

    This protects us from overtrusting players with only a few matches.
    Later we can improve this with age of data, source coverage, and tour level.
    """
    try:
        matches_total = int(matches_total or 0)
    except Exception:
        matches_total = 0

    return round(min(1.0, matches_total / 50.0), 3)


def surface_confidence(surface_matches):
    out = {}

    for surface, count in (surface_matches or {}).items():
        try:
            count = int(count or 0)
        except Exception:
            count = 0

        out[surface] = round(min(1.0, count / 25.0), 3)

    return out


def empty_player(name):
    return {
        "player_name": name,
        "player_key": canonical_player_name(name),

        # Ratings
        "overall_elo": DEFAULT_ELO,
        "surface_elo": {},

        # Basic record
        "matches_total": 0,
        "wins": 0,
        "losses": 0,
        "surface_matches": {},
        "surface_wins": {},
        "surface_losses": {},

        # Activity
        "last_match_date": "",
        "last_surface": "",

        # Strength of schedule / quality
        "wins_vs_higher_elo": 0,
        "losses_vs_lower_elo": 0,
        "wins_vs_higher_surface_elo": 0,
        "losses_vs_lower_surface_elo": 0,

        "best_win": None,
        "worst_loss": None,

        # Recent form; stored internally as rolling list during build.
        "_recent_matches": deque(maxlen=RECENT_WINDOW),
    }


def get_player(players, name):
    key = canonical_player_name(name)

    if key not in players:
        players[key] = empty_player(name)

    return players[key]


def get_surface_rating(player, surface):
    surface = clean_str(surface) or "unknown"
    return float(player.get("surface_elo", {}).get(surface, DEFAULT_ELO))


def set_surface_rating(player, surface, value):
    surface = clean_str(surface) or "unknown"
    player.setdefault("surface_elo", {})[surface] = round(float(value), 3)


def add_surface_counter(player, surface, key):
    surface = clean_str(surface) or "unknown"
    player.setdefault(key, {})
    player[key][surface] = int(player[key].get(surface, 0)) + 1


def build_match_snapshot(row, opponent_name, result, player_pre_elo, opponent_pre_elo, player_pre_surface_elo, opponent_pre_surface_elo):
    return {
        "date": row.get("date") or "",
        "tournament": row.get("tournament") or row.get("tournament_code") or "",
        "surface": row.get("surface") or "unknown",
        "opponent": opponent_name,
        "result": result,
        "score": row.get("score") or "",
        "player_pre_match_elo": round(float(player_pre_elo), 3),
        "opponent_pre_match_elo": round(float(opponent_pre_elo), 3),
        "player_pre_surface_elo": round(float(player_pre_surface_elo), 3),
        "opponent_pre_surface_elo": round(float(opponent_pre_surface_elo), 3),
        "source_url": row.get("source_url") or "",
        "canonical_match_id": row.get("canonical_match_id") or "",
    }


def update_best_win(player, snapshot):
    opponent_elo = snapshot.get("opponent_pre_match_elo")

    if opponent_elo is None:
        return

    current = player.get("best_win")

    if current is None or float(opponent_elo) > float(current.get("opponent_pre_match_elo", -9999)):
        player["best_win"] = snapshot


def update_worst_loss(player, snapshot):
    opponent_elo = snapshot.get("opponent_pre_match_elo")

    if opponent_elo is None:
        return

    current = player.get("worst_loss")

    if current is None or float(opponent_elo) < float(current.get("opponent_pre_match_elo", 9999)):
        player["worst_loss"] = snapshot


def add_recent_match(player, snapshot):
    player["_recent_matches"].append(snapshot)


def finalize_recent_form(player):
    recent = list(player.get("_recent_matches", []))

    wins = sum(1 for m in recent if m.get("result") == "win")
    losses = sum(1 for m in recent if m.get("result") == "loss")
    sample_size = len(recent)

    opponent_elos = [
        float(m.get("opponent_pre_match_elo"))
        for m in recent
        if m.get("opponent_pre_match_elo") is not None
    ]

    opponent_surface_elos = [
        float(m.get("opponent_pre_surface_elo"))
        for m in recent
        if m.get("opponent_pre_surface_elo") is not None
    ]

    avg_opponent_elo = None
    avg_opponent_surface_elo = None

    if opponent_elos:
        avg_opponent_elo = round(sum(opponent_elos) / len(opponent_elos), 3)

    if opponent_surface_elos:
        avg_opponent_surface_elo = round(sum(opponent_surface_elos) / len(opponent_surface_elos), 3)

    player["recent_10_sample_size"] = sample_size
    player["recent_10_wins"] = wins
    player["recent_10_losses"] = losses
    player["recent_10_record"] = f"{wins}-{losses}"
    player["recent_10_win_rate"] = round(wins / sample_size, 3) if sample_size else None
    player["recent_10_avg_opponent_elo"] = avg_opponent_elo
    player["recent_10_avg_opponent_surface_elo"] = avg_opponent_surface_elo
    player["recent_10_matches"] = recent

    player.pop("_recent_matches", None)


def process_match(players, row):
    winner_name = clean_str(row.get("winner"))
    loser_name = clean_str(row.get("loser"))
    surface = clean_str(row.get("surface")) or "unknown"

    if not winner_name or not loser_name or winner_name == loser_name:
        return False

    winner = get_player(players, winner_name)
    loser = get_player(players, loser_name)

    # Pre-match ratings are what we use for strength-of-schedule metrics.
    winner_pre_elo = float(winner.get("overall_elo", DEFAULT_ELO))
    loser_pre_elo = float(loser.get("overall_elo", DEFAULT_ELO))

    winner_pre_surface_elo = get_surface_rating(winner, surface)
    loser_pre_surface_elo = get_surface_rating(loser, surface)

    # Quality counters based on pre-match ratings.
    if loser_pre_elo > winner_pre_elo:
        winner["wins_vs_higher_elo"] += 1

    if winner_pre_elo < loser_pre_elo:
        loser["losses_vs_lower_elo"] += 1

    if loser_pre_surface_elo > winner_pre_surface_elo:
        winner["wins_vs_higher_surface_elo"] += 1

    if winner_pre_surface_elo < loser_pre_surface_elo:
        loser["losses_vs_lower_surface_elo"] += 1

    # Recent snapshots before ELO update.
    winner_snapshot = build_match_snapshot(
        row=row,
        opponent_name=loser_name,
        result="win",
        player_pre_elo=winner_pre_elo,
        opponent_pre_elo=loser_pre_elo,
        player_pre_surface_elo=winner_pre_surface_elo,
        opponent_pre_surface_elo=loser_pre_surface_elo,
    )

    loser_snapshot = build_match_snapshot(
        row=row,
        opponent_name=winner_name,
        result="loss",
        player_pre_elo=loser_pre_elo,
        opponent_pre_elo=winner_pre_elo,
        player_pre_surface_elo=loser_pre_surface_elo,
        opponent_pre_surface_elo=winner_pre_surface_elo,
    )

    update_best_win(winner, winner_snapshot)
    update_worst_loss(loser, loser_snapshot)

    add_recent_match(winner, winner_snapshot)
    add_recent_match(loser, loser_snapshot)

    # Overall ELO update.
    new_winner_elo, new_loser_elo = update_elo(
        winner_pre_elo,
        loser_pre_elo,
        K_FACTOR,
    )

    winner["overall_elo"] = round(new_winner_elo, 3)
    loser["overall_elo"] = round(new_loser_elo, 3)

    # Surface ELO update.
    new_winner_surface_elo, new_loser_surface_elo = update_elo(
        winner_pre_surface_elo,
        loser_pre_surface_elo,
        SURFACE_K_FACTOR,
    )

    set_surface_rating(winner, surface, new_winner_surface_elo)
    set_surface_rating(loser, surface, new_loser_surface_elo)

    # Basic counters.
    winner["matches_total"] += 1
    winner["wins"] += 1
    winner["last_match_date"] = row.get("date") or winner.get("last_match_date", "")
    winner["last_surface"] = surface

    loser["matches_total"] += 1
    loser["losses"] += 1
    loser["last_match_date"] = row.get("date") or loser.get("last_match_date", "")
    loser["last_surface"] = surface

    add_surface_counter(winner, surface, "surface_matches")
    add_surface_counter(loser, surface, "surface_matches")

    add_surface_counter(winner, surface, "surface_wins")
    add_surface_counter(loser, surface, "surface_losses")

    return True


def normalize_sort_date(value):
    """
    Flashscore/Rezulati currently gives dates like 05.06.26.
    For sorting inside one dataset, convert to 2026-06-05 style when possible.
    """
    value = clean_str(value)

    parts = value.split(".")
    if len(parts) >= 3:
        dd, mm, yy = parts[0], parts[1], parts[2]
        if len(yy) == 2:
            yy = "20" + yy
        return f"{yy}-{mm.zfill(2)}-{dd.zfill(2)}"

    return value


def build_report(output):
    lines = []
    counts = output.get("counts", {})

    lines.append("# Tennis Elo Machine Report")
    lines.append("")
    lines.append(f"Generated: `{output.get('generated_at')}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Input matches: **{counts.get('input_matches', 0)}**")
    lines.append(f"- Used matches: **{counts.get('used_matches', 0)}**")
    lines.append(f"- Skipped matches: **{counts.get('skipped_matches', 0)}**")
    lines.append(f"- Players: **{counts.get('players', 0)}**")
    lines.append("")
    lines.append("## Top ratings")
    lines.append("")
    lines.append("| Rank | Player | Overall ELO | Matches | W-L | Recent 10 | Recent Opp Avg ELO | Wins vs higher ELO | Losses vs lower ELO | Confidence | Last match | Last surface |")
    lines.append("|---:|---|---:|---:|---|---|---:|---:|---:|---:|---|---|")

    for i, row in enumerate(output.get("ratings", [])[:100], start=1):
        wl = f"{row.get('wins', 0)}-{row.get('losses', 0)}"
        lines.append(
            f"| {i} "
            f"| {row.get('player_name', '')} "
            f"| {row.get('overall_elo', '')} "
            f"| {row.get('matches_total', 0)} "
            f"| {wl} "
            f"| {row.get('recent_10_record', '')} "
            f"| {row.get('recent_10_avg_opponent_elo', '')} "
            f"| {row.get('wins_vs_higher_elo', 0)} "
            f"| {row.get('losses_vs_lower_elo', 0)} "
            f"| {row.get('confidence', 0)} "
            f"| {row.get('last_match_date', '')} "
            f"| {row.get('last_surface', '')} |"
        )

    lines.append("")
    lines.append("## Best wins")
    lines.append("")
    lines.append("| Player | Best win opponent | Opponent pre-match ELO | Date | Surface | Score |")
    lines.append("|---|---|---:|---|---|---|")

    best_win_rows = []
    for row in output.get("ratings", []):
        best = row.get("best_win")
        if best:
            best_win_rows.append((float(best.get("opponent_pre_match_elo", 0)), row, best))

    best_win_rows.sort(key=lambda x: x[0], reverse=True)

    for _, row, best in best_win_rows[:50]:
        lines.append(
            f"| {row.get('player_name', '')} "
            f"| {best.get('opponent', '')} "
            f"| {best.get('opponent_pre_match_elo', '')} "
            f"| {best.get('date', '')} "
            f"| {best.get('surface', '')} "
            f"| {best.get('score', '')} |"
        )

    return "\n".join(lines)


def main():
    payload = load_json(CANONICAL_MATCHES_FILE, {})
    matches = payload.get("matches", []) if isinstance(payload, dict) else []

    players = {}
    used = 0
    skipped = 0

    # Chronological build is important for ELO.
    matches = sorted(
        matches,
        key=lambda r: (
            normalize_sort_date(r.get("date")),
            str(r.get("tournament") or ""),
            str(r.get("canonical_match_id") or ""),
        ),
    )

    for row in matches:
        if process_match(players, row):
            used += 1
        else:
            skipped += 1

    ratings = list(players.values())

    for row in ratings:
        finalize_recent_form(row)
        row["confidence"] = confidence(row.get("matches_total", 0))
        row["surface_confidence"] = surface_confidence(row.get("surface_matches", {}))

    ratings.sort(
        key=lambda r: (
            -float(r.get("overall_elo", 0)),
            -int(r.get("matches_total", 0)),
            str(r.get("player_name") or ""),
        )
    )

    output = {
        "generated_at": now_iso(),
        "model": "flashscore_poc_elo_v2_strength_of_schedule",
        "source_file": str(CANONICAL_MATCHES_FILE),
        "settings": {
            "default_elo": DEFAULT_ELO,
            "k_factor": K_FACTOR,
            "surface_k_factor": SURFACE_K_FACTOR,
            "recent_window": RECENT_WINDOW,
            "note": "This is still a cold-start POC. Confidence must be used before betting decisions.",
        },
        "counts": {
            "input_matches": len(matches),
            "used_matches": used,
            "skipped_matches": skipped,
            "players": len(ratings),
        },
        "ratings": ratings,
    }

    save_json(PLAYER_RATINGS_FILE, output)
    save_text(ELO_REPORT_FILE, build_report(output))

    print("")
    print("BUILD ELO DONE")
    print(output["counts"])
    print(f"Output: {PLAYER_RATINGS_FILE}")
    print(f"Report: {ELO_REPORT_FILE}")
    print("")


if __name__ == "__main__":
    main()
