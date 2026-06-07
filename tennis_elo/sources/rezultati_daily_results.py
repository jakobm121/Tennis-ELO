import os
import re
from urllib.parse import urlparse, parse_qs

from playwright.sync_api import sync_playwright

from tennis_elo.config import ROOT_DIR
from tennis_elo.utils import now_iso, save_json, save_text, clean_str, load_json


BASE_URL = os.getenv("REZULTATI_BASE_URL", "https://www.rezultati.com/")
DAILY_URL = os.getenv("REZULTATI_DAILY_URL", "https://www.rezultati.com/tenis/")

DAILY_OUTPUT_FILE = ROOT_DIR / "data" / "totals" / "daily_scorelines.json"
ARCHIVE_FILE = ROOT_DIR / "data" / "totals" / "scorelines_archive.json"
REPORT_FILE = ROOT_DIR / "data" / "reports" / "rezultati_daily_results_report.json"

DEBUG_TEXT_FILE = ROOT_DIR / "data" / "reports" / "debug_rezultati_daily_results_text.txt"
DEBUG_HTML_FILE = ROOT_DIR / "data" / "reports" / "debug_rezultati_daily_results_page.html"
DEBUG_SCREENSHOT_FILE = ROOT_DIR / "data" / "reports" / "debug_rezultati_daily_results_screenshot.png"

WAIT_MS = int(os.getenv("DAILY_RESULTS_WAIT_MS", "8000"))
MAX_RESULT_ROWS = int(os.getenv("MAX_DAILY_RESULT_ROWS", "200"))
CLICK_FINISHED_TAB = os.getenv("CLICK_FINISHED_TAB", "1") == "1"

FINISHED_WORDS = ["kraj", "finished", "gotovo"]
BAD_RESULT_WORDS = ["walkover", "w.o.", "predaja", "retired", "ret."]


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
    slug = clean_str(slug)
    if not slug:
        return ""

    parts = [p for p in slug.split("-") if p]

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


def accept_cookies(page):
    for text in ["PrihvaÄam", "Prihvati", "SlaÅ¾em se", "Accept all", "I accept", "Accept", "OK"]:
        try:
            locator = page.get_by_text(text, exact=False).first
            if locator.count() > 0 and locator.is_visible(timeout=1000):
                locator.click(timeout=2500)
                page.wait_for_timeout(1200)
                return True
        except Exception:
            pass
    return False


def page_looks_like_tennis(text):
    lower = clean_str(text).lower()
    if "hokej livescore" in lower or "hokej rezultati" in lower:
        return False
    return any(x in lower for x in ["tenis", "atp", "wta", "challenger", "itf", "french open", "roland garros"])


def click_text_first(page, labels):
    for exact in [True, False]:
        for text in labels:
            try:
                locator = page.get_by_text(text, exact=exact).first
                if locator.count() > 0 and locator.is_visible(timeout=2000):
                    locator.click(timeout=5000)
                    page.wait_for_timeout(2500)
                    return True
            except Exception:
                pass
    return False


def go_to_tennis_page(page):
    page.goto(BASE_URL, wait_until="domcontentloaded", timeout=45000)
    page.wait_for_timeout(3500)
    accept_cookies(page)
    page.wait_for_timeout(1500)

    clicked = click_text_first(page, ["TENIS", "Tenis", "Tennis"])

    try:
        body_text = page.locator("body").inner_text(timeout=8000)
    except Exception:
        body_text = ""

    if not clicked or not page_looks_like_tennis(body_text):
        page.goto(DAILY_URL, wait_until="domcontentloaded", timeout=45000)
        page.wait_for_timeout(WAIT_MS)
        accept_cookies(page)
        page.wait_for_timeout(1500)

    if CLICK_FINISHED_TAB:
        click_text_first(page, ["GOTOVO", "Gotovo", "Finished", "Kraj"])


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


def is_int_text(value):
    return bool(re.fullmatch(r"\d{1,2}", clean_str(value)))


def parse_int(value):
    try:
        return int(clean_str(value))
    except Exception:
        return None


def clean_lines(text):
    lines = []
    for line in str(text or "").splitlines():
        line = clean_str(line)
        if not line:
            continue
        if line in ["-", "â"]:
            continue
        lines.append(line)
    return lines


def is_finished_text(text):
    lower = clean_str(text).lower()
    if any(x in lower for x in BAD_RESULT_WORDS):
        return True
    return any(x in lower for x in FINISHED_WORDS)


def is_retired_or_walkover(text):
    lower = clean_str(text).lower()
    return any(x in lower for x in BAD_RESULT_WORDS)


def parse_score_numbers(lines):
    numeric_positions = [(idx, parse_int(line)) for idx, line in enumerate(lines) if is_int_text(line)]

    if len(numeric_positions) < 4:
        return None

    first_num_idx = numeric_positions[0][0]
    nums = [n for _, n in numeric_positions]

    sets_1, sets_2 = nums[0], nums[1]

    if sets_1 is None or sets_2 is None:
        return None

    if sets_1 < 0 or sets_2 < 0 or sets_1 > 5 or sets_2 > 5:
        return None

    total_sets = sets_1 + sets_2

    if total_sets <= 0 or total_sets > 5:
        return None

    remaining = nums[2:]
    set_scores = []
    pos = 0

    for set_number in range(1, total_sets + 1):
        if pos + 1 >= len(remaining):
            break

        # Tie-break encoded as p1_games, p1_tb, p2_games, p2_tb.
        if pos + 3 < len(remaining):
            a_games = remaining[pos]
            a_tb = remaining[pos + 1]
            b_games = remaining[pos + 2]
            b_tb = remaining[pos + 3]

            if (
                ((a_games == 7 and b_games == 6) or (a_games == 6 and b_games == 7))
                and a_tb is not None
                and b_tb is not None
                and 0 <= a_tb <= 30
                and 0 <= b_tb <= 30
            ):
                set_scores.append({
                    "set_number": set_number,
                    "p1_games": a_games,
                    "p2_games": b_games,
                    "tiebreak": True,
                    "tiebreak_p1": a_tb,
                    "tiebreak_p2": b_tb,
                })
                pos += 4
                continue

        a_games = remaining[pos]
        b_games = remaining[pos + 1]
        set_scores.append({
            "set_number": set_number,
            "p1_games": a_games,
            "p2_games": b_games,
            "tiebreak": bool((a_games == 7 and b_games == 6) or (a_games == 6 and b_games == 7)),
        })
        pos += 2

    if len(set_scores) != total_sets:
        return None

    return {
        "first_num_idx": first_num_idx,
        "sets_1": sets_1,
        "sets_2": sets_2,
        "set_scores": set_scores,
    }


def derive_from_set_scores(set_scores, sets_1, sets_2):
    total_games = 0
    p1_total_games = 0
    p2_total_games = 0
    tiebreak_count = 0

    for s in set_scores:
        p1 = int(s.get("p1_games") or 0)
        p2 = int(s.get("p2_games") or 0)
        p1_total_games += p1
        p2_total_games += p2
        total_games += p1 + p2

        if s.get("tiebreak"):
            tiebreak_count += 1

    first = set_scores[0] if set_scores else {}
    first_set_games = int(first.get("p1_games") or 0) + int(first.get("p2_games") or 0)

    total_sets = int(sets_1 or 0) + int(sets_2 or 0)

    return {
        "first_set_score": f"{first.get('p1_games')}-{first.get('p2_games')}" if first else "",
        "first_set_games": first_set_games if first else None,
        "first_set_tiebreak": bool(first.get("tiebreak")) if first else False,
        "total_games": total_games,
        "p1_total_games_won": p1_total_games,
        "p2_total_games_won": p2_total_games,
        "game_margin_p1": p1_total_games - p2_total_games,
        "total_sets": total_sets,
        "straight_sets": bool(sets_1 == 0 or sets_2 == 0),
        "deciding_set": bool(total_sets in [3, 5]),
        "had_tiebreak": tiebreak_count > 0,
        "tiebreak_count": tiebreak_count,
    }


def estimate_best_of(sets_1, sets_2, tournament="", gender=""):
    total_sets = int(sets_1 or 0) + int(sets_2 or 0)
    if total_sets > 3:
        return 5

    tournament_lower = clean_str(tournament).lower()
    gender_lower = clean_str(gender).lower()

    if gender_lower == "men" and any(x in tournament_lower for x in ["french open", "australian open", "wimbledon", "us open", "fo"]):
        return 5

    return 3


def row_text_to_match(row_text, match_url=""):
    lines = clean_lines(row_text)

    if not lines or not is_finished_text(row_text):
        return None

    retired = is_retired_or_walkover(row_text)
    parsed = parse_score_numbers(lines)

    if not parsed:
        return None

    first_num_idx = parsed["first_num_idx"]
    sets_1 = parsed["sets_1"]
    sets_2 = parsed["sets_2"]
    set_scores = parsed["set_scores"]

    before_nums = [line for line in lines[:first_num_idx] if not is_int_text(line)]
    before_nums = [x for x in before_nums if clean_str(x).lower() not in ["kraj", "finished", "gotovo"]]

    p1 = ""
    p2 = ""

    if len(before_nums) >= 2:
        p1, p2 = before_nums[-2], before_nums[-1]

    url_p1, url_p2 = infer_players_from_url(match_url)

    if url_p1:
        p1 = url_p1
    if url_p2:
        p2 = url_p2

    if not p1 or not p2:
        return None

    winner = p1 if sets_1 > sets_2 else p2
    loser = p2 if sets_1 > sets_2 else p1

    derived = derive_from_set_scores(set_scores, sets_1, sets_2)
    match_id = extract_mid(match_url)

    out = {
        "match_id": match_id,
        "match_url": match_url,
        "source": "rezultati_daily_results",
        "date": "",
        "tournament": "",
        "tour_level": "",
        "gender": "",
        "surface": "",
        "indoor": None,
        "best_of": estimate_best_of(sets_1, sets_2),
        "is_grand_slam": None,
        "player_1": p1,
        "player_2": p2,
        "match": f"{p1} - {p2}",
        "winner": winner,
        "loser": loser,
        "sets_1": sets_1,
        "sets_2": sets_2,
        "final_score": f"{sets_1}-{sets_2}",
        "set_scores": set_scores,
        "completed": not retired,
        "retired": retired,
        "walkover": retired,
        "raw_parent_text": clean_str(row_text),
    }

    out.update(derived)
    return out


def collect_rows(page):
    rows = page.locator('[id^="g_2_"]')

    try:
        count = rows.count()
    except Exception:
        count = 0

    results = {}
    max_rows = min(count, MAX_RESULT_ROWS)

    for idx in range(max_rows):
        try:
            rows = page.locator('[id^="g_2_"]')
            row = rows.nth(idx)
            row_text = row.inner_text(timeout=2500)

            if not is_finished_text(row_text):
                continue

            before_url = page.url
            match_url = ""

            try:
                row.scroll_into_view_if_needed(timeout=3000)
            except Exception:
                pass

            try:
                row.click(timeout=5000)
                page.wait_for_timeout(1800)
                match_url = normalize_url(page.url)
            except Exception:
                match_url = ""

            if match_url:
                try:
                    page.go_back(wait_until="domcontentloaded", timeout=15000)
                    page.wait_for_timeout(1200)
                except Exception:
                    page.goto(before_url or DAILY_URL, wait_until="domcontentloaded", timeout=30000)
                    page.wait_for_timeout(2500)

            parsed = row_text_to_match(row_text, match_url=match_url)

            if not parsed:
                continue

            key = parsed.get("match_id") or parsed.get("match_url") or f"{parsed.get('player_1')}|{parsed.get('player_2')}|{parsed.get('final_score')}|{parsed.get('raw_parent_text')}"
            results[key] = parsed

        except Exception:
            continue

    return list(results.values())


def archive_key(row):
    return clean_str(row.get("match_id")) or clean_str(row.get("match_url")) or "|".join([
        clean_str(row.get("date")),
        clean_str(row.get("player_1")),
        clean_str(row.get("player_2")),
        clean_str(row.get("final_score")),
        clean_str(row.get("first_set_score")),
    ])


def merge_archive(existing_rows, new_rows):
    by_key = {}
    order = []

    for row in existing_rows + new_rows:
        key = archive_key(row)
        if not key:
            continue

        if key not in by_key:
            by_key[key] = row
            order.append(key)
        else:
            old = by_key[key]
            if isinstance(row, dict) and isinstance(old, dict) and len(row.keys()) > len(old.keys()):
                by_key[key] = row

    return [by_key[k] for k in order]


def load_archive_rows():
    payload = load_json(ARCHIVE_FILE, {})
    if isinstance(payload, dict):
        rows = payload.get("matches") or payload.get("scorelines") or []
        if isinstance(rows, list):
            return rows
    if isinstance(payload, list):
        return payload
    return []


def save_outputs(daily_rows):
    archive_before = load_archive_rows()
    archive_after = merge_archive(archive_before, daily_rows)

    daily_payload = {
        "generated_at": now_iso(),
        "source_url": DAILY_URL,
        "counts": {
            "daily_scorelines": len(daily_rows),
        },
        "matches": daily_rows,
    }

    archive_payload = {
        "generated_at": now_iso(),
        "source": "rezultati_daily_results",
        "counts": {
            "archive_before": len(archive_before),
            "daily_scorelines": len(daily_rows),
            "archive_after": len(archive_after),
            "added": max(0, len(archive_after) - len(archive_before)),
        },
        "matches": archive_after,
    }

    report = {
        "generated_at": now_iso(),
        "source_url": DAILY_URL,
        "output_file": str(DAILY_OUTPUT_FILE),
        "archive_file": str(ARCHIVE_FILE),
        "counts": {
            "archive_before": len(archive_before),
            "daily_scorelines": len(daily_rows),
            "archive_after": len(archive_after),
            "added": max(0, len(archive_after) - len(archive_before)),
        },
        "sample": daily_rows[:20],
        "debug_files": {
            "text": str(DEBUG_TEXT_FILE),
            "html": str(DEBUG_HTML_FILE),
            "screenshot": str(DEBUG_SCREENSHOT_FILE),
        },
    }

    save_json(DAILY_OUTPUT_FILE, daily_payload)
    save_json(ARCHIVE_FILE, archive_payload)
    save_json(REPORT_FILE, report)

    return report


def main():
    daily_rows = []
    error = ""

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
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
            go_to_tennis_page(page)
            save_debug(page)
            daily_rows = collect_rows(page)

            context.close()
            browser.close()

    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    report = save_outputs(daily_rows)

    if error:
        report["error"] = error
        save_json(REPORT_FILE, report)

    print("")
    print("REZULTATI DAILY RESULTS DONE")
    print(report["counts"])
    print(f"Daily:   {DAILY_OUTPUT_FILE}")
    print(f"Archive: {ARCHIVE_FILE}")
    print(f"Report:  {REPORT_FILE}")

    if error:
        print(f"WARNING: {error}")

    print("")


if __name__ == "__main__":
    main()
