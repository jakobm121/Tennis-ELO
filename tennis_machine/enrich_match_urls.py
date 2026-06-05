import json
import os
import re
from urllib.parse import urlparse, parse_qs

from playwright.sync_api import sync_playwright


INPUT_FILE = "tennis_machine/data/processed/flashscore_matches.json"
OUTPUT_FILE = "tennis_machine/data/processed/flashscore_matches.json"
REPORT_FILE = "tennis_machine/reports/enrich_match_urls_report.md"

FLASHSCORE_TENNIS_URL = os.getenv("FLASHSCORE_TENNIS_URL", "https://www.flashscore.com/tennis/")

WAIT_MS = int(os.getenv("ENRICH_URLS_WAIT_MS", "7000"))


def ensure_parent(path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, payload):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def save_text(path, text):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def clean(value):
    return str(value or "").strip()


def mid_from_match_id(match_id):
    match_id = clean(match_id)
    if match_id.startswith("g_2_"):
        return match_id.replace("g_2_", "", 1)
    return match_id


def extract_mid_from_url(url):
    url = clean(url)
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


def normalize_flashscore_match_url(url):
    """
    Returns canonical Flashscore match-summary URL.
    """
    url = clean(url)
    if not url:
        return ""

    if url.startswith("/"):
        url = "https://www.flashscore.com" + url

    if "flashscore.com/match/tennis/" not in url:
        return ""

    url_no_hash = url.split("#", 1)[0].rstrip("/")

    # Keep mid query.
    if "?mid=" not in url_no_hash:
        return ""

    return url_no_hash + "/#/match-summary"


def make_odds_url(match_url):
    match_url = clean(match_url)
    if not match_url:
        return ""

    base = match_url.split("#", 1)[0].rstrip("/")
    return base + "/#/odds-comparison/1x2-odds/full-time"


def make_rezultati_match_url(match_url):
    """
    Converts:
      https://www.flashscore.com/match/tennis/mensik.../?mid=abc/#/match-summary
    into:
      https://www.rezultati.com/utakmica/tenis/mensik.../?mid=abc

    This is what Tennis-ELO history URL generator needs.
    """
    match_url = clean(match_url)
    if not match_url:
        return ""

    base = match_url.split("#", 1)[0].rstrip("/")

    if "flashscore.com/match/tennis/" not in base:
        return ""

    base = base.replace(
        "https://www.flashscore.com/match/tennis/",
        "https://www.rezultati.com/utakmica/tenis/",
    )
    base = base.replace(
        "http://www.flashscore.com/match/tennis/",
        "https://www.rezultati.com/utakmica/tenis/",
    )

    return base


def collect_match_links_from_page(page):
    page.goto(FLASHSCORE_TENNIS_URL, wait_until="domcontentloaded", timeout=45000)
    page.wait_for_timeout(WAIT_MS)

    hrefs = page.locator("a").evaluate_all("els => els.map(a => a.href).filter(Boolean)")

    by_mid = {}
    all_links = []

    for href in hrefs:
        match_url = normalize_flashscore_match_url(href)
        if not match_url:
            continue

        mid = extract_mid_from_url(match_url)
        if not mid:
            continue

        if mid not in by_mid:
            by_mid[mid] = match_url

        all_links.append(match_url)

    return by_mid, sorted(set(all_links))


def enrich_payload(payload, url_by_mid):
    if isinstance(payload, dict):
        matches = payload.get("matches", [])
    elif isinstance(payload, list):
        matches = payload
    else:
        matches = []

    updated = 0
    already_had_url = 0
    missing = 0

    examples_updated = []
    examples_missing = []

    for row in matches:
        if not isinstance(row, dict):
            continue

        existing_url = clean(row.get("match_url"))
        if existing_url:
            already_had_url += 1
            continue

        mid = mid_from_match_id(row.get("match_id"))
        match_url = url_by_mid.get(mid)

        if not match_url:
            missing += 1
            if len(examples_missing) < 20:
                examples_missing.append({
                    "match_id": row.get("match_id"),
                    "mid": mid,
                    "match": row.get("match"),
                    "status": row.get("status"),
                    "tournament": row.get("tournament"),
                })
            continue

        row["match_url"] = match_url
        row["odds_url"] = make_odds_url(match_url)
        row["rezultati_match_url"] = make_rezultati_match_url(match_url)

        updated += 1

        if len(examples_updated) < 20:
            examples_updated.append({
                "match_id": row.get("match_id"),
                "match": row.get("match"),
                "match_url": row.get("match_url"),
                "rezultati_match_url": row.get("rezultati_match_url"),
            })

    return {
        "updated": updated,
        "already_had_url": already_had_url,
        "missing": missing,
        "examples_updated": examples_updated,
        "examples_missing": examples_missing,
        "total_matches": len(matches),
    }


def build_report(stats, links_found):
    lines = []
    lines.append("# Tennis Machine - Enrich Match URLs Report")
    lines.append("")
    lines.append(f"Source page: `{FLASHSCORE_TENNIS_URL}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Match links found on page: **{links_found}**")
    lines.append(f"- Total matches in JSON: **{stats.get('total_matches', 0)}**")
    lines.append(f"- Updated with URLs: **{stats.get('updated', 0)}**")
    lines.append(f"- Already had URL: **{stats.get('already_had_url', 0)}**")
    lines.append(f"- Missing URL: **{stats.get('missing', 0)}**")
    lines.append("")
    lines.append("## Updated examples")
    lines.append("")
    for item in stats.get("examples_updated", []):
        lines.append(f"- `{item.get('match_id')}` {item.get('match')}: {item.get('rezultati_match_url')}")
    lines.append("")
    lines.append("## Missing examples")
    lines.append("")
    for item in stats.get("examples_missing", []):
        lines.append(f"- `{item.get('match_id')}` {item.get('match')} / {item.get('status')} / {item.get('tournament')}")
    lines.append("")
    return "\n".join(lines)


def main():
    payload = load_json(INPUT_FILE)

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
            locale="en-US",
            timezone_id="Europe/Ljubljana",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()
        url_by_mid, all_links = collect_match_links_from_page(page)
        context.close()
        browser.close()

    stats = enrich_payload(payload, url_by_mid)

    save_json(OUTPUT_FILE, payload)
    save_text(REPORT_FILE, build_report(stats, links_found=len(all_links)))

    print("")
    print("TENNIS MACHINE ENRICH MATCH URLS DONE")
    print({
        "links_found": len(all_links),
        "total_matches": stats.get("total_matches", 0),
        "updated": stats.get("updated", 0),
        "already_had_url": stats.get("already_had_url", 0),
        "missing": stats.get("missing", 0),
    })
    print(f"Output: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")
    print("")


if __name__ == "__main__":
    main()
