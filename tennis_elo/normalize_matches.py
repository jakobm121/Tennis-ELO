from tennis_elo.config import FLASH_HISTORY_RAW_FILE, FLASH_HISTORY_NORMALIZED_FILE
from tennis_elo.utils import load_json, save_json, now_iso, clean_str


def normalize_date(value):
    """
    Keeps Flashscore/Rezulati date as-is for now.
    Example: 05.06.26

    Later we can convert to ISO when we confirm century/year rules.
    """
    return clean_str(value)


def normalize_surface(row, page):
    surface = clean_str(row.get("surface"))
    if surface:
        return surface

    surface_filter = clean_str(row.get("surface_filter") or page.get("surface_filter"))
    if surface_filter and surface_filter != "all":
        return surface_filter

    return ""


def normalize_row(row, page):
    """
    New parser already gives structured rows:
    - date
    - tournament_code
    - player_1
    - player_2
    - winner
    - loser
    - score

    This normalizer should not re-parse raw text anymore.
    It should trust only rows with parse_status == ok and winner/loser present.
    """
    player_1 = clean_str(row.get("player_1"))
    player_2 = clean_str(row.get("player_2"))
    winner = clean_str(row.get("winner"))
    loser = clean_str(row.get("loser"))

    parse_status = clean_str(row.get("parse_status")) or "unknown"

    ready_for_elo = (
        parse_status == "ok"
        and bool(player_1)
        and bool(player_2)
        and bool(winner)
        and bool(loser)
        and winner != loser
    )

    return {
        "date": normalize_date(row.get("date")),
        "tournament": clean_str(row.get("tournament") or row.get("tournament_code")),
        "tournament_code": clean_str(row.get("tournament_code")),
        "surface": normalize_surface(row, page),
        "surface_filter": clean_str(row.get("surface_filter") or page.get("surface_filter")),
        "focus_player": clean_str(row.get("focus_player")),
        "player_1": player_1,
        "player_2": player_2,
        "sets_1": row.get("sets_1"),
        "sets_2": row.get("sets_2"),
        "winner": winner,
        "loser": loser,
        "score": clean_str(row.get("score")),
        "result_marker": clean_str(row.get("result_marker")),
        "source": clean_str(row.get("source")) or "flashscore",
        "source_url": clean_str(row.get("source_url") or page.get("url")),
        "raw": clean_str(row.get("raw")),
        "raw_players": clean_str(row.get("raw_players")),
        "parser": clean_str(row.get("parser")),
        "split_method": clean_str(row.get("split_method")),
        "parse_status": "ok" if ready_for_elo else "needs_parser_improvement",
        "ready_for_elo": ready_for_elo,
    }


def main():
    payload = load_json(FLASH_HISTORY_RAW_FILE, {})
    pages = payload.get("pages", []) if isinstance(payload, dict) else []

    normalized = []

    for page in pages:
        for row in page.get("rows", []):
            if not isinstance(row, dict):
                continue

            normalized.append(normalize_row(row, page))

    ready = [r for r in normalized if r.get("ready_for_elo")]
    needs = [r for r in normalized if not r.get("ready_for_elo")]

    output = {
        "generated_at": now_iso(),
        "source_file": str(FLASH_HISTORY_RAW_FILE),
        "counts": {
            "raw_pages": len(pages),
            "normalized_rows": len(normalized),
            "ready_for_elo": len(ready),
            "needs_parser_improvement": len(needs),
        },
        "matches": normalized,
    }

    save_json(FLASH_HISTORY_NORMALIZED_FILE, output)

    print("")
    print("NORMALIZE MATCHES DONE")
    print(output["counts"])
    print(f"Output: {FLASH_HISTORY_NORMALIZED_FILE}")
    print("")


if __name__ == "__main__":
    main()
