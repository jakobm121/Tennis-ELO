from tennis_elo.config import FLASH_HISTORY_NORMALIZED_FILE, CANONICAL_MATCHES_FILE
from tennis_elo.utils import load_json, save_json, now_iso, canonical_key_part, clean_str


def canonical_match_key(row):
    date = canonical_key_part(row.get("date"))
    winner = canonical_key_part(row.get("winner"))
    loser = canonical_key_part(row.get("loser"))
    tournament = canonical_key_part(row.get("tournament"))
    score = canonical_key_part(row.get("score"))
    players = sorted([winner, loser])
    return "|".join([date, players[0], players[1], tournament, score])


def is_elo_usable(row):
    return bool(
        clean_str(row.get("date"))
        and clean_str(row.get("winner"))
        and clean_str(row.get("loser"))
        and clean_str(row.get("winner")) != clean_str(row.get("loser"))
    )


def main():
    payload = load_json(FLASH_HISTORY_NORMALIZED_FILE, {})
    rows = payload.get("matches", []) if isinstance(payload, dict) else []
    canonical = {}
    skipped = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not is_elo_usable(row):
            skipped.append({"reason": "not_elo_usable", "raw": row.get("raw"), "row": row})
            continue
        key = canonical_match_key(row)
        if key not in canonical:
            new_row = dict(row)
            new_row["canonical_match_id"] = key
            new_row["sources"] = [row.get("source") or "unknown"]
            canonical[key] = new_row
            continue
        old = canonical[key]
        old_sources = set(old.get("sources", []))
        old_sources.add(row.get("source") or "unknown")
        old["sources"] = sorted(old_sources)
        for k, v in row.items():
            if not old.get(k) and v:
                old[k] = v
    matches = list(canonical.values())
    matches.sort(key=lambda r: (str(r.get("date") or ""), str(r.get("tournament") or ""), str(r.get("winner") or "")))
    output = {
        "generated_at": now_iso(),
        "source_file": str(FLASH_HISTORY_NORMALIZED_FILE),
        "counts": {"input_rows": len(rows), "canonical_matches": len(matches), "skipped": len(skipped)},
        "matches": matches,
        "skipped": skipped[:200],
    }
    save_json(CANONICAL_MATCHES_FILE, output)
    print("\nDEDUPE MATCHES DONE")
    print(output["counts"])
    print(f"Output: {CANONICAL_MATCHES_FILE}\n")


if __name__ == "__main__":
    main()
