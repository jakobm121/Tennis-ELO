import os
import re
from urllib.parse import urlparse, parse_qs

from playwright.sync_api import sync_playwright

from tennis_elo.config import ROOT_DIR
from tennis_elo.utils import now_iso, save_json, save_text, clean_str, load_json


BASE_URL = os.getenv("REZULTATI_BASE_URL", "https://www.rezultati.com/")
DAILY_URL = os.getenv("REZULTATI_DAILY_URL", "https://www.rezultati.com/tenis/")

OUTPUT_FILE = ROOT_DIR / "data" / "input" / "matches_today.json"
SEEN_FILE = ROOT_DIR / "data" / "state" / "seen_match_urls.json"
REPORT_FILE = ROOT_DIR / "data" / "reports" / "rezultati_daily_matches_report.json"

DEBUG_TEXT_FILE = ROOT_DIR / "data" / "reports" / "debug_rezultati_daily_text.txt"
DEBUG_HTML_FILE = ROOT_DIR / "data" / "reports" / "debug_rezultati_daily_page.html"
DEBUG_SCREENSHOT_FILE = ROOT_DIR / "data" / "reports" / "debug_rezultati_daily_screenshot.png"

MAX_MATCHES = int(os.getenv("MAX_DAILY_MATCHES", "20"))
WAIT_MS = int(os.getenv("DAILY_WAIT_MS", "8000"))

# If 1, already processed input match URLs are skipped.
SKIP_SEEN_MATCHES = os.getenv("SKIP_SEEN_MATCHES", "1") == "1"

# If 1, clear seen cache before selecting matches.
# Useful only for manual testing.
RESET_SEEN_MATCHES = os.getenv("RESET_SEEN_MATCHES", "0") == "1"

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


def load_seen():
    if RESET_SEEN_MATCHES:
        return {
            "generated_at": now_iso(),
            "seen_urls": [],
            "seen": {},
        }

    payload = load_json(SEEN_FILE, {})
    if not isinstance(payload, dict):
        return {
            "generated_at": now_iso(),
            "seen_urls": [],
            "seen": {},
        }

    if "seen" not in payload or not isinstance(payload.get("seen"), dict):
        seen = {}
        for url in payload.get("seen_urls", []) or []:
            seen[clean_str(url)] = {"first_seen_at": "", "last_seen_at": ""}
        payload["seen"] = seen

    if "seen_urls" not in payload or not isinstance(payload.get("seen_urls"), list):
        payload["seen_urls"] = sorted(payload.get("seen", {}).keys())

    return payload


def save_seen(seen_payload):
    seen_payload["generated_at"] = now_iso()
    seen_payload["seen_urls"] = sorted(seen_payload.get("seen", {}).keys())
    save_json(SEEN_FILE, seen_payload)


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
    slug = clean_str(slug)
    if not slug:
        return ""

    parts = [p for p in slug.split("-") if p]

    # Last token is usually player id, for example UFj257G8.
    if parts and re.search(r"[A-Z0-9]", parts[-1]) and len(parts[-1]) >= 6:
        parts = parts[:-1]

    name = " ".join(parts).strip()
    if not name:
        return ""

    return " ".join(part.capitalize() for part in name.split())


def infer_players_from_url(url):
    url = normalize_url(url)
    if not url:
        return "", ""

    path = urlparse(url).path.strip("/")
    parts = path.split("/")

    try:
        idx = parts.index("tenis")
        p1_slug = parts[idx + 1]
        p2_slug = parts[idx + 2]
    except Exception:
        return "", ""

    return slug_to_name(p1_slug), slug_to_name(p2_slug)


def get_status_from_text(text):
    lower = clean_str(text).lower()

    if any(x in lower for x in ["finished", "kraj", "retired", "walkover", "predaja"]):
        return "finished"

    if any(x in lower for x in ["set ", "uÅ¾ivo", "uzivo", "live", "break"]):
        return "live"

    if re.search(r"\b\d{1,2}:\d{2}\b", lower):
        return "scheduled"

    return "scheduled"


def accept_cookies(page):
    for text in [
        "PrihvaÄam",
        "Prihvati",
        "SlaÅ¾em se",
        "Accept all",
        "I accept",
        "Accept",
        "OK",
    ]:
        try:
            locator = page.get_by_text(text, exact=False).first
            if locator.count() > 0 and locator.is_visible(timeout=1000):
                locator.click(timeout=2500)
                page.wait_for_timeout(1200)
                return True
        except Exception:
            pass

    return False


def save_debug(page):
    html = page.content()
    text = page.locator("body").inner_text(timeout=10000)

    ensure_parent(DEBUG_TEXT_FILE)
    save_text(DEBUG_TEXT_FILE, text)
    save_text(DEBUG_HTML_FILE, html)

    try:
        page.screenshot(path=str(DEBUG_SCREENSHOT_FILE), full_page=True)
    except Exception:
        pass

    return text


def page_looks_like_tennis(text):
    lower = clean_str(text).lower()

    if "hokej livescore" in lower or "hokej rezultati" in lower:
        return False

    tennis_signals = [
        "tenis",
        "atp",
        "wta",
        "challenger",
        "itf",
        "french open",
        "roland garros",
    ]

    return any(x in lower for x in tennis_signals)


def go_to_tennis_page(page):
    page.goto(BASE_URL, wait_until="domcontentloaded", timeout=45000)
    page.wait_for_timeout(3500)
    accept_cookies(page)
    page.wait_for_timeout(1500)

    clicked_tennis = False
    for text in ["TENIS", "Tenis", "Tennis"]:
        try:
            locator = page.get_by_text(text, exact=True).first
            if locator.count() > 0 and locator.is_visible(timeout=2000):
                locator.click(timeout=5000)
                page.wait_for_timeout(WAIT_MS)
                clicked_tennis = True
                break
        except Exception:
            pass

    try:
        body_text = page.locator("body").inner_text(timeout=8000)
    except Exception:
        body_text = ""

    if not clicked_tennis or not page_looks_like_tennis(body_text):
        page.goto(DAILY_URL, wait_until="domcontentloaded", timeout=45000)
        page.wait_for_timeout(WAIT_MS)
        accept_cookies(page)
        page.wait_for_timeout(1500)

        try:
            body_text = page.locator("body").inner_text(timeout=8000)
        except Exception:
            body_text = ""

    if not page_looks_like_tennis(body_text):
        for text in ["TENIS", "Tenis", "Tennis"]:
            try:
                locator = page.get_by_text(text, exact=True).first
                if locator.count() > 0 and locator.is_visible(timeout=2000):
                    locator.click(timeout=5000)
                    page.wait_for_timeout(WAIT_MS)
                    break
            except Exception:
                pass


def row_to_match_from_url(url, row_text="", source="rezultati_daily"):
    url = normalize_url(url)
    if not url:
        return None

    p1, p2 = infer_players_from_url(url)
    status = get_status_from_text(row_text)

    if INCLUDE_STATUSES and status.lower() not in INCLUDE_STATUSES:
        return None

    return {
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
        "source": source,
        "source_match_id": extract_mid(url),
        "raw_parent_text": clean_str(row_text),
    }


def collect_anchor_links(page):
    links_payload = page.locator("a").evaluate_all(
        """
        els => els.map(a => {
          const href = a.href || "";
          let text = "";
          try { text = a.innerText || ""; } catch (e) {}
          let parentText = "";
          try {
            const p = a.closest('[id^="g_"], [class*="event__match"], [class*="eventRow"], [class*="sportName"]');
            parentText = p ? p.innerText : "";
          } catch (e) {}
          return {href, text, parentText};
        }).filter(x => x.href)
        """
    )

    out = {}

    for item in links_payload:
        match = row_to_match_from_url(
            item.get("href"),
            row_text=item.get("parentText") or item.get("text"),
            source="rezultati_daily_anchor",
        )
        if not match:
            continue
        out[match["match_url"]] = match

    return out


def collect_clickable_rows(page):
    rows = page.locator('[id^="g_2_"]')

    try:
        count = rows.count()
    except Exception:
        count = 0

    collected = {}
    max_clicks = min(count, MAX_MATCHES * 8)

    for idx in range(max_clicks):
        # Do not stop here on collected count, because seen filtering happens later.
        try:
            rows = page.locator('[id^="g_2_"]')
            row = rows.nth(idx)

            row_text = ""
            try:
                row_text = row.inner_text(timeout=2000)
            except Exception:
                row_text = ""

            status = get_status_from_text(row_text)
            if INCLUDE_STATUSES and status.lower() not in INCLUDE_STATUSES:
                continue

            before_url = page.url

            try:
                row.scroll_into_view_if_needed(timeout=3000)
            except Exception:
                pass

            row.click(timeout=5000)
            page.wait_for_timeout(2500)

            after_url = normalize_url(page.url)

            if after_url:
                match = row_to_match_from_url(
                    after_url,
                    row_text=row_text,
                    source="rezultati_daily_click",
                )
                if match:
                    collected[match["match_url"]] = match

            try:
                page.go_back(wait_until="domcontentloaded", timeout=15000)
                page.wait_for_timeout(1800)
            except Exception:
                page.goto(before_url or DAILY_URL, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(2500)

        except Exception:
            try:
                go_to_tennis_page(page)
            except Exception:
                pass
            continue

    return collected


def collect_match_links(page):
    go_to_tennis_page(page)
    body_text = save_debug(page)

    if any(block_text in body_text for block_text in BLOCK_TEXTS):
        return [], True, "rezultati error page / blocked"

    if not page_looks_like_tennis(body_text):
        return [], False, "page did not switch to tennis; debug text is probably another sport"

    by_url = {}
    by_url.update(collect_anchor_links(page))

    # Always try clickable rows too; direct anchors can miss many matches.
    by_url.update(collect_clickable_rows(page))

    matches = list(by_url.values())

    matches.sort(
        key=lambda r: (
            0 if r.get("player_1") and r.get("player_2") else 1,
            r.get("match") or "",
        )
    )

    return matches, False, ""


def select_unseen_matches(matches, seen_payload):
    seen = seen_payload.setdefault("seen", {})
    now = now_iso()

    selected = []
    skipped_seen = []
    available_unseen = []

    for match in matches:
        url = clean_str(match.get("match_url"))
        if not url:
            continue

        if SKIP_SEEN_MATCHES and url in seen:
            skipped_seen.append(match)
            # Update last_seen_at because it appeared again in daily list.
            if isinstance(seen.get(url), dict):
                seen[url]["last_seen_at"] = now
            continue

        available_unseen.append(match)

    for match in available_unseen:
        if len(selected) >= MAX_MATCHES:
            break

        url = clean_str(match.get("match_url"))
        selected.append(match)

        seen[url] = {
            "first_seen_at": seen.get(url, {}).get("first_seen_at") if isinstance(seen.get(url), dict) else now,
            "last_seen_at": now,
            "match": match.get("match", ""),
            "player_1": match.get("player_1", ""),
            "player_2": match.get("player_2", ""),
            "status": match.get("status", ""),
            "source_match_id": match.get("source_match_id", ""),
        }

    return selected, skipped_seen, available_unseen


def save_outputs(found_matches, selected, skipped_seen, available_unseen, seen_payload, blocked=False, error=""):
    output = {
        "generated_at": now_iso(),
        "source_url": DAILY_URL,
        "max_matches": MAX_MATCHES,
        "skip_seen_matches": SKIP_SEEN_MATCHES,
        "matches": selected,
    }

    report = {
        "generated_at": now_iso(),
        "source_url": DAILY_URL,
        "output_file": str(OUTPUT_FILE),
        "seen_file": str(SEEN_FILE),
        "settings": {
            "max_matches": MAX_MATCHES,
            "wait_ms": WAIT_MS,
            "include_statuses": sorted(INCLUDE_STATUSES),
            "skip_seen_matches": SKIP_SEEN_MATCHES,
            "reset_seen_matches": RESET_SEEN_MATCHES,
        },
        "counts": {
            "found_matches": len(found_matches),
            "available_unseen": len(available_unseen),
            "selected": len(selected),
            "skipped_seen": len(skipped_seen),
            "seen_total": len(seen_payload.get("seen", {})),
            "blocked": 1 if blocked else 0,
            "errors": 1 if error else 0,
        },
        "error": error,
        "selected": selected,
        "skipped_seen_examples": skipped_seen[:50],
        "debug_files": {
            "text": str(DEBUG_TEXT_FILE),
            "html": str(DEBUG_HTML_FILE),
            "screenshot": str(DEBUG_SCREENSHOT_FILE),
        },
    }

    save_json(OUTPUT_FILE, output)
    save_seen(seen_payload)
    save_json(REPORT_FILE, report)

    return report


def main():
    found_matches = []
    blocked = False
    error = ""

    seen_payload = load_seen()

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
            found_matches, blocked, error = collect_match_links(page)

            context.close()
            browser.close()

    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    selected, skipped_seen, available_unseen = select_unseen_matches(found_matches, seen_payload)

    report = save_outputs(
        found_matches=found_matches,
        selected=selected,
        skipped_seen=skipped_seen,
        available_unseen=available_unseen,
        seen_payload=seen_payload,
        blocked=blocked,
        error=error,
    )

    print("")
    print("REZULTATI DAILY MATCHES DONE")
    print(report["counts"])
    print(f"Output: {OUTPUT_FILE}")
    print(f"Seen:   {SEEN_FILE}")
    print(f"Report: {REPORT_FILE}")

    if error:
        print(f"WARNING: {error}")

    print("")


if __name__ == "__main__":
    main()
