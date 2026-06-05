import json
import os
import re
from urllib.parse import urlparse, parse_qs

from playwright.sync_api import sync_playwright

from tennis_elo.config import ROOT_DIR
from tennis_elo.utils import now_iso, save_json, save_text, clean_str


DAILY_URL = os.getenv("REZULTATI_DAILY_URL", "https://www.rezultati.com/tenis/")
OUTPUT_FILE = ROOT_DIR / "data" / "input" / "matches_today.json"
REPORT_FILE = ROOT_DIR / "data" / "reports" / "rezultati_daily_matches_report.json"
DEBUG_TEXT_FILE = ROOT_DIR / "data" / "reports" / "debug_rezultati_daily_text.txt"
DEBUG_HTML_FILE = ROOT_DIR / "data" / "reports" / "debug_rezultati_daily_page.html"
DEBUG_SCREENSHOT_FILE = ROOT_DIR / "data" / "reports" / "debug_rezultati_daily_screenshot.png"

MAX_MATCHES = int(os.getenv("MAX_DAILY_MATCHES", "20"))
WAIT_MS = int(os.getenv("DAILY_WAIT_MS", "7000"))

INCLUDE_STATUSES = {
    x.strip().lower()
    for x in os.getenv("DAILY_INCLUDE_STATUSES", "scheduled,live").split(",")
    if x.strip()
}

BLOCK_TEXTS = [
    "The requested page can't be displayed",
    "Please try again later",
    "Access denied",
]


def ensure_parent(path):
    folder = os.path.dirname(str(path))
    if folder:
        os.makedirs(folder, exist_ok=True)


def normalize_url(url):
    url = clean_str(url)

    if not url:
        return ""

    if url.startswith("/"):
        url = "https://www.rezultati.com" + url

    url = url.split("#", 1)[0]

    if "rezultati.com/utakmica/tenis/" not in url:
        return ""

    if "?mid=" not in url:
        return ""

    return url.rstrip("/")


def extract_mid(url):
    url = clean_str(url)
    if not url:
        return ""

    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        if "mid" in qs and qs["mid"]:
            return qs["mid"][0]
    except Exception:
        pass

    m = re.search(r"[?&]mid=([^&#/]+)", url)
    if m:
        return m.group(1)

    return ""


def slug_to_name(slug):
    """
    Converts rough URL slug to a readable fallback player name.
    Example:
      mensik-jakub-UFj257G8 -> Mensik Jakub
      yunchaokete-bu-Qabjt3Y5 -> Yunchaokete Bu
    """
    slug = clean_str(slug)
    if not slug:
        return ""

    parts = [p for p in slug.split("-") if p]

    # Last part is usually Flashscore player id. Remove if it looks like mixed id.
    if parts and re.search(r"[A-Z0-9]", parts[-1]) and len(parts[-1]) >= 6:
        parts = parts[:-1]

    name = " ".join(parts).strip()
    if not name:
        return ""

    return " ".join(part.capitalize() for part in name.split())


def infer_players_from_url(url):
    """
    URL pattern:
    https://www.rezultati.com/utakmica/tenis/player-one-id/player-two-id/?mid=...
    """
    url = normalize_url(url)
    if not url:
        return "", ""

    path = urlparse(url).path.strip("/")
    parts = path.split("/")

    # Expected: utakmica / tenis / player1 / player2
    try:
        idx = parts.index("tenis")
        p1_slug = parts[idx + 1]
        p2_slug = parts[idx + 2]
    except Exception:
        return "", ""

    return slug_to_name(p1_slug), slug_to_name(p2_slug)


def get_status_from_row_text(text):
    lower = clean_str(text).lower()

    live_markers = [
        "set ",
        "live",
        "uÅ¾ivo",
        "uzivo",
        "break",
    ]

    finished_markers = [
        "finished",
        "kraj",
        "nakon predaje",
        "retired",
        "walkover",
    ]

    if any(x in lower for x in finished_markers):
        return "finished"

    if any(x in lower for x in live_markers):
        return "live"

    # If row contains time pattern, treat as scheduled.
    if re.search(r"\b\d{1,2}:\d{2}\b", lower):
        return "scheduled"

    return "scheduled"


def collect_match_links(page):
    page.goto(DAILY_URL, wait_until="domcontentloaded", timeout=45000)
    page.wait_for_timeout(WAIT_MS)

    # Try click cookie buttons if visible.
    for text in ["I accept", "Accept", "PrihvaÄam", "SlaÅ¾em se", "OK"]:
        try:
            locator = page.get_by_text(text, exact=False).first
            if locator.count() > 0 and locator.is_visible(timeout=1000):
                locator.click(timeout=2000)
                page.wait_for_timeout(1000)
                break
        except Exception:
            pass

    html = page.content()
    body_text = page.locator("body").inner_text(timeout=10000)

    ensure_parent(DEBUG_TEXT_FILE)
    save_text(DEBUG_TEXT_FILE, body_text)
    save_text(DEBUG_HTML_FILE, html)

    try:
        page.screenshot(path=str(DEBUG_SCREENSHOT_FILE), full_page=True)
    except Exception:
        pass

    if any(block_text in body_text for block_text in BLOCK_TEXTS):
        return [], True, "rezultati error page / blocked"

    links_payload = page.locator("a").evaluate_all(
        """
        els => els.map(a => {
          const href = a.href || "";
          let text = "";
          try {
            text = a.innerText || "";
          } catch (e) {
            text = "";
          }
          let parentText = "";
          try {
            let p = a.closest('[class*="event__match"], [class*="event__participant"], [class*="eventRow"], [id^="g_"]');
            parentText = p ? p.innerText : "";
          } catch (e) {
            parentText = "";
          }
          return {href, text, parentText};
        }).filter(x => x.href)
        """
    )

    by_url = {}

    for item in links_payload:
        url = normalize_url(item.get("href"))
        if not url:
            continue

        p1, p2 = infer_players_from_url(url)
        status = get_status_from_row_text(item.get("parentText") or item.get("text"))

        if INCLUDE_STATUSES and status.lower() not in INCLUDE_STATUSES:
            continue

        by_url[url] = {
            "match": f"{p1} - {p2}" if p1 and p2 else "",
            "match_url": url,
            "date": "",
            "status": status,
            "tour_level": "",
            "gender": "",
            "tournament": "",
            "country": "",
            "surface": "",
            "player_1": p1,
            "player_2": p2,
            "source": "rezultati_daily",
            "source_match_id": extract_mid(url),
            "raw_link_text": clean_str(item.get("text")),
            "raw_parent_text": clean_str(item.get("parentText")),
        }

    matches = list(by_url.values())

    # Prefer rows with player names.
    matches.sort(
        key=lambda r: (
            0 if r.get("player_1") and r.get("player_2") else 1,
            r.get("match") or "",
        )
    )

    return matches, False, ""


def save_outputs(matches, blocked=False, error=""):
    selected = matches[:MAX_MATCHES]

    output = {
        "generated_at": now_iso(),
        "source_url": DAILY_URL,
        "max_matches": MAX_MATCHES,
        "matches": selected,
    }

    report = {
        "generated_at": now_iso(),
        "source_url": DAILY_URL,
        "output_file": str(OUTPUT_FILE),
        "settings": {
            "max_matches": MAX_MATCHES,
            "wait_ms": WAIT_MS,
            "include_statuses": sorted(INCLUDE_STATUSES),
        },
        "counts": {
            "found_matches": len(matches),
            "selected": len(selected),
            "blocked": 1 if blocked else 0,
            "errors": 1 if error else 0,
        },
        "error": error,
        "selected": selected,
        "debug_files": {
            "text": str(DEBUG_TEXT_FILE),
            "html": str(DEBUG_HTML_FILE),
            "screenshot": str(DEBUG_SCREENSHOT_FILE),
        },
    }

    save_json(OUTPUT_FILE, output)
    save_json(REPORT_FILE, report)

    return report


def main():
    matches = []
    blocked = False
    error = ""

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )

            context = browser.new_context(
                viewport={"width": 1400, "height": 1200},
                locale="hr-HR",
                timezone_id="Europe/Ljubljana",
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
            )

            page = context.new_page()
            matches, blocked, error = collect_match_links(page)

            context.close()
            browser.close()

    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    report = save_outputs(matches, blocked=blocked, error=error)

    print("")
    print("REZULTATI DAILY MATCHES DONE")
    print(report["counts"])
    print(f"Output: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")

    if error:
        print(f"WARNING: {error}")

    print("")


if __name__ == "__main__":
    main()
