import re

from tennis_elo.config import FLASH_HISTORY_RAW_FILE, FLASH_HISTORY_NORMALIZED_FILE
from tennis_elo.utils import load_json, save_json, now_iso, clean_str

DATE_RE = re.compile(r"(?P<date>\d{1,2}\.\d{1,2}\.|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2})")
SCORE_RE = re.compile(r"(?P<a>[0-7])\s*[- ]\s*(?P<b>[0-7])")


def guess_result_from_raw(raw):
    raw = clean_str(raw)
    date = ""
    m = DATE_RE.search(raw)
    if m:
        date = m.group("date")
    scores = SCORE_RE.findall(raw)
    return {
        "date": date,
        "tournament": "",
        "surface": "",
        "player_1": "",
        "player_2": "",
        "winner": "",
        "loser": "",
        "score": " ".join([f"{a}-{b}" for a, b in scores]),
        "raw": raw,
        "parse_status": "needs_parser_improvement",
    }


def main():
    payload = load_json(FLASH_HISTORY_RAW_FILE, {})
    pages = payload.get("pages", []) if isinstance(payload, dict) else []
    normalized = []
    for page in pages:
        url = page.get("url")
        surface_hint = page.get("surface_hint") or ""
        for row in page.get("rows", []):
            item = guess_result_from_raw(row.get("raw", ""))
            item["source"] = "flashscore"
            item["source_url"] = url
            if surface_hint and not item.get("surface"):
                item["surface"] = surface_hint
            normalized.append(item)
    output = {
        "generated_at": now_iso(),
        "source_file": str(FLASH_HISTORY_RAW_FILE),
        "counts": {
            "raw_pages": len(pages),
            "normalized_rows": len(normalized),
            "ready_for_elo": len([r for r in normalized if r.get("winner") and r.get("loser")]),
            "needs_parser_improvement": len([r for r in normalized if not r.get("winner") or not r.get("loser")]),
        },
        "matches": normalized,
    }
    save_json(FLASH_HISTORY_NORMALIZED_FILE, output)
    print("\nNORMALIZE MATCHES DONE")
    print(output["counts"])
    print(f"Output: {FLASH_HISTORY_NORMALIZED_FILE}\n")


if __name__ == "__main__":
    main()
