import math

from tennis_elo.config import CANONICAL_MATCHES_FILE, PLAYER_RATINGS_FILE, ELO_REPORT_FILE, DEFAULT_ELO, K_FACTOR, SURFACE_K_FACTOR
from tennis_elo.utils import load_json, save_json, save_text, now_iso, clean_str, canonical_player_name


def expected_score(rating_a, rating_b):
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def update_elo(winner_rating, loser_rating, k):
    ew = expected_score(winner_rating, loser_rating)
    el = expected_score(loser_rating, winner_rating)
    return winner_rating + k * (1.0 - ew), loser_rating + k * (0.0 - el)


def empty_player(name):
    return {
        "player_name": name,
        "player_key": canonical_player_name(name),
        "overall_elo": DEFAULT_ELO,
        "surface_elo": {},
        "matches_total": 0,
        "wins": 0,
        "losses": 0,
        "surface_matches": {},
        "last_match_date": "",
        "last_surface": "",
    }


def confidence(matches_total):
    return round(min(1.0, matches_total / 50.0), 3)


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
    lines.append("| Rank | Player | Overall ELO | Matches | W-L | Confidence | Last match | Last surface |")
    lines.append("|---:|---|---:|---:|---|---:|---|---|")
    for i, row in enumerate(output.get("ratings", [])[:100], start=1):
        wl = f"{row.get('wins', 0)}-{row.get('losses', 0)}"
        lines.append(f"| {i} | {row.get('player_name', '')} | {row.get('overall_elo', '')} | {row.get('matches_total', 0)} | {wl} | {row.get('confidence', 0)} | {row.get('last_match_date', '')} | {row.get('last_surface', '')} |")
    return "\n".join(lines)


def main():
    payload = load_json(CANONICAL_MATCHES_FILE, {})
    matches = payload.get("matches", []) if isinstance(payload, dict) else []
    players = {}

    def get_player(name):
        key = canonical_player_name(name)
        if key not in players:
            players[key] = empty_player(name)
        return players[key]

    used = 0
    skipped = 0
    matches = sorted(matches, key=lambda r: (str(r.get("date") or ""), str(r.get("tournament") or ""), str(r.get("winner") or "")))
    for row in matches:
        winner_name = clean_str(row.get("winner"))
        loser_name = clean_str(row.get("loser"))
        surface = clean_str(row.get("surface")) or "unknown"
        if not winner_name or not loser_name or winner_name == loser_name:
            skipped += 1
            continue
        winner = get_player(winner_name)
        loser = get_player(loser_name)
        new_w, new_l = update_elo(float(winner["overall_elo"]), float(loser["overall_elo"]), K_FACTOR)
        winner["overall_elo"] = round(new_w, 3)
        loser["overall_elo"] = round(new_l, 3)
        winner_surface_rating = float(winner["surface_elo"].get(surface, DEFAULT_ELO))
        loser_surface_rating = float(loser["surface_elo"].get(surface, DEFAULT_ELO))
        new_sw, new_sl = update_elo(winner_surface_rating, loser_surface_rating, SURFACE_K_FACTOR)
        winner["surface_elo"][surface] = round(new_sw, 3)
        loser["surface_elo"][surface] = round(new_sl, 3)
        winner["matches_total"] += 1
        winner["wins"] += 1
        winner["last_match_date"] = row.get("date") or winner["last_match_date"]
        winner["last_surface"] = surface
        loser["matches_total"] += 1
        loser["losses"] += 1
        loser["last_match_date"] = row.get("date") or loser["last_match_date"]
        loser["last_surface"] = surface
        winner["surface_matches"][surface] = winner["surface_matches"].get(surface, 0) + 1
        loser["surface_matches"][surface] = loser["surface_matches"].get(surface, 0) + 1
        used += 1
    ratings = list(players.values())
    for row in ratings:
        row["confidence"] = confidence(row.get("matches_total", 0))
    ratings.sort(key=lambda r: (-float(r.get("overall_elo", 0)), r.get("player_name", "")))
    output = {
        "generated_at": now_iso(),
        "model": "flashscore_poc_elo_v1",
        "source_file": str(CANONICAL_MATCHES_FILE),
        "settings": {"default_elo": DEFAULT_ELO, "k_factor": K_FACTOR, "surface_k_factor": SURFACE_K_FACTOR},
        "counts": {"input_matches": len(matches), "used_matches": used, "skipped_matches": skipped, "players": len(ratings)},
        "ratings": ratings,
    }
    save_json(PLAYER_RATINGS_FILE, output)
    save_text(ELO_REPORT_FILE, build_report(output))
    print("\nBUILD ELO DONE")
    print(output["counts"])
    print(f"Output: {PLAYER_RATINGS_FILE}")
    print(f"Report: {ELO_REPORT_FILE}\n")


if __name__ == "__main__":
    main()
