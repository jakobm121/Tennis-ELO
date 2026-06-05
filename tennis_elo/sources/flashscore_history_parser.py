import re
from playwright.sync_api import sync_playwright

from tennis_elo.config import FLASH_HISTORY_RAW_FILE, WATCHLIST_FILE, RAW_DIR
from tennis_elo.utils import now_iso, save_json, save_text, clean_str

DEBUG_DIR = RAW_DIR / "debug"

BLOCK_TEXTS = [
    "The requested page can't be displayed",
    "Please try again later",
    "Access denied",
]


def read_urls():
    if not WATCHLIST_FILE.exists():
        return []
    urls = []
    for line in WATCHLIST_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def click_show_more(page, max_clicks=8):
    clicked = 0
    button_texts = [
        "Show more matches",
        "Show more",
        "Prikaži još mečeva",
        "Prikaži više",
        "Prikaži več",
        "Load more",
    ]
    for _ in range(max_clicks):
        did_click = False
        for text in button_texts:
            try:
                locator = page.get_by_text(text, exact=False).first
                if locator.count() > 0 and locator.is_visible(timeout=1000):
                    locator.click(timeout=2000)
                    page.wait_for_timeout(1200)
                    clicked += 1
                    did_click = True
                    break
            except Exception:
                continue
        if not did_click:
            break
    return clicked


def extract_candidate_rows_from_text(text):
    lines = [x.strip() for x in str(text or "").splitlines()]
    lines = [x for x in lines if x]
    rows = []
    buffer = []
    date_like = re.compile(r"(\d{1,2}\.\d{1,2}\.|\d{4}|\d{1,2}/\d{1,2})")
    score_like = re.compile(r"\b[0-7]\s+[0-7]\b|\b[0-7]-[0-7]\b")
    for line in lines:
        buffer.append(line)
        if len(buffer) > 12:
            buffer.pop(0)
        joined = " | ".join(buffer)
        if date_like.search(joined) and score_like.search(joined):
            rows.append({"raw": joined, "parser": "text_candidate"})
            buffer = []
    deduped = []
    seen = set()
    for row in rows:
        raw = row["raw"]
        if raw in seen:
            continue
        seen.add(raw)
        deduped.append(row)
    return deduped


def detect_surface_from_text(text):
    lower = str(text or "").lower()
    if "clay" in lower or "zemlja" in lower:
        return "clay"
    if "hard" in lower or "tvrda" in lower:
        return "hard"
    if "grass" in lower or "trava" in lower:
        return "grass"
    if "indoor" in lower:
        return "indoor"
    return ""


def parse_one_url(page, url, index):
    result = {
        "url": clean_str(url),
        "parsed_at": now_iso(),
        "blocked": False,
        "error": "",
        "show_more_clicked": 0,
        "surface_hint": "",
        "rows": [],
        "debug_files": {},
    }
    try:
        page.goto(result["url"], wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(5000)
        result["show_more_clicked"] = click_show_more(page, max_clicks=8)
        html = page.content()
        text = page.locator("body").inner_text(timeout=8000)
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        html_path = DEBUG_DIR / f"flashscore_history_{index}.html"
        text_path = DEBUG_DIR / f"flashscore_history_{index}.txt"
        screenshot_path = DEBUG_DIR / f"flashscore_history_{index}.png"
        save_text(html_path, html)
        save_text(text_path, text)
        try:
            page.screenshot(path=str(screenshot_path), full_page=True)
        except Exception:
            pass
        result["debug_files"] = {
            "html": str(html_path),
            "text": str(text_path),
            "screenshot": str(screenshot_path),
        }
        if any(block_text in text for block_text in BLOCK_TEXTS):
            result["blocked"] = True
            result["error"] = "flashscore error page / blocked"
            return result
        result["surface_hint"] = detect_surface_from_text(text)
        result["rows"] = extract_candidate_rows_from_text(text)
        if not result["rows"]:
            result["error"] = "no candidate history rows found"
        return result
    except Exception as e:
        result["error"] = str(e)
        return result


def main():
    urls = read_urls()
    output = {
        "generated_at": now_iso(),
        "source_file": str(WATCHLIST_FILE),
        "counts": {"urls": len(urls), "blocked": 0, "errors": 0, "raw_rows": 0},
        "pages": [],
    }
    if not urls:
        save_json(FLASH_HISTORY_RAW_FILE, output)
        print("No URLs found in config/flashscore_urls.txt")
        return
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 1200}, locale="en-US", timezone_id="Europe/Ljubljana")
        page = context.new_page()
        for i, url in enumerate(urls, start=1):
            print(f"Parsing {i}/{len(urls)}: {url}")
            page_result = parse_one_url(page, url, i)
            output["pages"].append(page_result)
            if page_result.get("blocked"):
                output["counts"]["blocked"] += 1
            if page_result.get("error"):
                output["counts"]["errors"] += 1
            output["counts"]["raw_rows"] += len(page_result.get("rows", []))
        context.close()
        browser.close()
    save_json(FLASH_HISTORY_RAW_FILE, output)
    print("\nFLASHSCORE HISTORY PARSER DONE")
    print(output["counts"])
    print(f"Output: {FLASH_HISTORY_RAW_FILE}\n")


if __name__ == "__main__":
    main()
