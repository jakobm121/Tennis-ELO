import json
import os
from copy import deepcopy

from tennis_elo.config import ROOT_DIR
from tennis_elo.utils import now_iso, save_json, clean_str


CURRENT_RAW_FILE = ROOT_DIR / "data" / "raw" / "flashscore_history_raw.json"
ARCHIVE_FILE = ROOT_DIR / "data" / "raw" / "flashscore_history_archive.json"
REPORT_FILE = ROOT_DIR / "data" / "reports" / "merge_history_archive_report.json"

# Keep this true so existing normalize_matches.py can keep reading the same raw file path.
# After merge, flashscore_history_raw.json becomes the full archive payload.
OVERWRITE_CURRENT_WITH_ARCHIVE = os.getenv("OVERWRITE_CURRENT_WITH_ARCHIVE", "1") == "1"


PAGE_LIST_KEYS = [
    "pages",
    "raw_pages",
    "results",
    "data",
]

ROW_LIST_KEYS = [
    "rows",
    "matches",
    "raw_rows",
    "history_rows",
    "table_rows",
    "items",
]


def load_json(path, default):
    if not os.path.exists(path):
        return default

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def stable_json(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(value)


def first_list_for_keys(payload, keys):
    if not isinstance(payload, dict):
        return None, ""

    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value, key

    return None, ""


def get_pages(payload):
    """
    Raw parser versions can store page objects under different keys.
    Return list of page-like objects and the key where they came from.
    """
    if isinstance(payload, list):
        return payload, "__root_list__"

    pages, key = first_list_for_keys(payload, PAGE_LIST_KEYS)
    if pages is not None:
        return pages, key

    # Some simple raw payloads may directly contain matches/rows.
    rows, row_key = first_list_for_keys(payload, ROW_LIST_KEYS)
    if rows is not None:
        return [
            {
                "source_url": payload.get("source_url") or payload.get("url") or "",
                "surface_filter": payload.get("surface_filter") or "",
                "focus_player": payload.get("focus_player") or "",
                "matches": rows,
            }
        ], f"wrapped_{row_key}"

    return [], ""


def get_rows_from_page(page):
    if not isinstance(page, dict):
        return []

    rows, key = first_list_for_keys(page, ROW_LIST_KEYS)
    if rows is not None:
        return rows

    return []


def row_key(row):
    if not isinstance(row, dict):
        return stable_json(row)

    # Prefer fields that define one historical match row.
    parts = [
        clean_str(row.get("source_url")),
        clean_str(row.get("surface_filter")),
        clean_str(row.get("focus_player")),
        clean_str(row.get("date")),
        clean_str(row.get("tournament") or row.get("tournament_code")),
        clean_str(row.get("raw")),
        clean_str(row.get("player_1")),
        clean_str(row.get("player_2")),
        clean_str(row.get("score")),
    ]

    key = "|".join(parts).strip("|")

    if key:
        return key

    return stable_json(row)


def page_key(page):
    if not isinstance(page, dict):
        return stable_json(page)

    source_url = clean_str(page.get("source_url") or page.get("url") or page.get("history_url"))
    surface_filter = clean_str(page.get("surface_filter") or page.get("surface") or page.get("filter"))
    focus_player = clean_str(page.get("focus_player") or page.get("player") or page.get("player_name"))

    if source_url:
        return f"{source_url}|{surface_filter}|{focus_player}"

    rows = get_rows_from_page(page)
    if rows:
        # If no URL, use first/last row signatures.
        return f"{row_key(rows[0])}::{row_key(rows[-1])}"

    return stable_json(page)


def merge_page_rows(old_page, new_page):
    """
    If the same page URL was scraped multiple times, merge its row list instead of
    duplicating the page.
    """
    old = deepcopy(old_page) if isinstance(old_page, dict) else old_page
    new = deepcopy(new_page) if isinstance(new_page, dict) else new_page

    if not isinstance(old, dict) or not isinstance(new, dict):
        return new

    old_rows, old_key = first_list_for_keys(old, ROW_LIST_KEYS)
    new_rows, new_key = first_list_for_keys(new, ROW_LIST_KEYS)

    if old_rows is None or new_rows is None:
        # Prefer newer metadata if we cannot identify rows.
        merged = deepcopy(old)
        for k, v in new.items():
            if v not in [None, "", [], {}]:
                merged[k] = v
        return merged

    seen = {}
    merged_rows = []

    for row in old_rows + new_rows:
        key = row_key(row)
        if key not in seen:
            seen[key] = row
            merged_rows.append(row)
        else:
            # Keep row with more fields if duplicate.
            existing = seen[key]
            if isinstance(row, dict) and isinstance(existing, dict) and len(row.keys()) > len(existing.keys()):
                idx = merged_rows.index(existing)
                merged_rows[idx] = row
                seen[key] = row

    merged = deepcopy(old)

    # Overlay non-empty metadata from newer page.
    for k, v in new.items():
        if k == new_key:
            continue
        if v not in [None, "", [], {}]:
            merged[k] = v

    merged[old_key or new_key or "matches"] = merged_rows
    return merged


def count_rows(pages):
    total = 0

    for page in pages:
        rows = get_rows_from_page(page)

        if rows:
            total += len(rows)
        else:
            total += 1

    return total


def count_unique_row_keys(pages):
    keys = set()

    for page in pages:
        rows = get_rows_from_page(page)
        if not rows:
            keys.add(page_key(page))
            continue

        for row in rows:
            keys.add(row_key(row))

    return len(keys)


def merge_pages(archive_pages, current_pages):
    by_key = {}
    order = []

    for page in archive_pages + current_pages:
        key = page_key(page)

        if key not in by_key:
            by_key[key] = deepcopy(page)
            order.append(key)
        else:
            by_key[key] = merge_page_rows(by_key[key], page)

    return [by_key[key] for key in order]


def main():
    current_payload = load_json(CURRENT_RAW_FILE, {})
    archive_payload = load_json(ARCHIVE_FILE, {})

    current_pages, current_key = get_pages(current_payload)
    archive_pages, archive_key = get_pages(archive_payload)

    merged_pages = merge_pages(archive_pages, current_pages)

    archive_output = {
        "generated_at": now_iso(),
        "source": "merge_history_archive",
        "source_files": {
            "current_raw": str(CURRENT_RAW_FILE),
            "archive": str(ARCHIVE_FILE),
        },
        "counts": {
            "archive_pages_before": len(archive_pages),
            "current_pages": len(current_pages),
            "merged_pages": len(merged_pages),
            "archive_rows_before": count_rows(archive_pages),
            "current_rows": count_rows(current_pages),
            "merged_rows": count_rows(merged_pages),
            "unique_row_keys": count_unique_row_keys(merged_pages),
        },
        "pages": merged_pages,
    }

    save_json(ARCHIVE_FILE, archive_output)

    if OVERWRITE_CURRENT_WITH_ARCHIVE:
        save_json(CURRENT_RAW_FILE, archive_output)

    report = {
        "generated_at": now_iso(),
        "current_raw_file": str(CURRENT_RAW_FILE),
        "archive_file": str(ARCHIVE_FILE),
        "overwrite_current_with_archive": OVERWRITE_CURRENT_WITH_ARCHIVE,
        "input_keys": {
            "current_pages_key": current_key,
            "archive_pages_key": archive_key,
        },
        "counts": archive_output["counts"],
    }

    save_json(REPORT_FILE, report)

    print("")
    print("MERGE HISTORY ARCHIVE DONE")
    print(report["counts"])
    print(f"Archive: {ARCHIVE_FILE}")
    print(f"Current raw: {CURRENT_RAW_FILE}")
    print(f"Report: {REPORT_FILE}")
    print("")


if __name__ == "__main__":
    main()
