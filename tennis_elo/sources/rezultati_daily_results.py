import os
import re
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from zoneinfo import ZoneInfo

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

TIMEZONE = os.getenv("TIMEZONE", "Europe/Ljubljana")
WAIT_MS = int(os.getenv("DAILY_RESULTS_WAIT_MS", "8000"))
CLICK_FINISHED_TAB = os.getenv("CLICK_FINISHED_TAB", "1") == "1"

FINISHED_WORDS = {"kraj", "finished", "gotovo"}
TOUR_MARKERS = ("ATP", "WTA", "CHALLENGER", "ITF", "JUNIOR", "JUNIORK")
SURFACE_MAP = {
    "zemlja": "clay",
    "Å¡ljaka": "clay",
    "sljaka": "clay",
    "trava": "grass",
    "tvrda": "hard",
    "hard": "hard",
    "tepih": "carpet",
    "carpet": "carpet",
}
GRAND_SLAM_MARKERS = (
    "french open", "roland garros", "australian open", "wimbledon", "us open", "u.s. open"
)


def ensure_parent(path):
    folder = os.path.dirname(str(path))
    if folder:
        os.makedirs(folder, exist_ok=True)


def today_iso():
    return datetime.now(ZoneInfo(TIMEZONE)).date().isoformat()


def accept_cookies(page):
    for text in ["PrihvaÄam", "Prihvati", "SlaÅ¾em se", "Accept all", "I accept", "Accept", "OK"]:
        try:
            locator = page.get_by_text(text, exact=False).first
            if locator.count() and locator.is_visible(timeout=1000):
                locator.click(timeout=2500)
                page.wait_for_timeout(1000)
                return
        except Exception:
            pass


def click_text_first(page, labels):
    for exact in (True, False):
        for text in labels:
            try:
                locator = page.get_by_text(text, exact=exact).first
                if locator.count() and locator.is_visible(timeout=2000):
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
    page.wait_for_timeout(1000)

    click_text_first(page, ["TENIS", "Tenis", "Tennis"])
    page.wait_for_timeout(WAIT_MS)

    body = ""
    try:
        body = page.locator("body").inner_text(timeout=8000)
    except Exception:
        pass

    if "Tenis rezultati" not in body and "ATP" not in body and "WTA" not in body:
        page.goto(DAILY_URL, wait_until="domcontentloaded", timeout=45000)
        page.wait_for_timeout(WAIT_MS)
        accept_cookies(page)

    if CLICK_FINISHED_TAB:
        click_text_first(page, ["GOTOVO", "Gotovo", "Finished"])
        page.wait_for_timeout(2500)


def save_debug(page):
    text = page.locator("body").inner_text(timeout=10000)
    html = page.content()

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


def derive_from_sets(set_scores, sets_1, sets_2):
    p1_total = sum(int(s["p1_games"]) for s in set_scores)
    p2_total = sum(int(s["p2_games"]) for s in set_scores)
    tb_count = sum(1 for s in set_scores if s.get("tiebreak"))

    first = set_scores[0] if set_scores else None
    total_sets = sets_1 + sets_2

    return {
        "first_set_score": f"{first['p1_games']}-{first['p2_games']}" if first else "",
        "first_set_games": (first["p1_games"] + first["p2_games"]) if first else None,
        "first_set_tiebreak": bool(first and first.get("tiebreak")),
        "total_games": p1_total + p2_total,
        "p1_total_games_won": p1_total,
        "p2_total_games_won": p2_total,
        "game_margin_p1": p1_total - p2_total,
        "total_sets": total_sets,
        "straight_sets": sets_1 == 0 or sets_2 == 0,
        "deciding_set": total_sets in (3, 5),
        "had_tiebreak": tb_count > 0,
        "tiebreak_count": tb_count,
    }


def parse_metadata(tournament_line, tour_line, sets_1, sets_2):
    lower_tournament = clean_str(tournament_line).lower()
    lower_tour = clean_str(tour_line).lower()

    surface = ""
    for marker, value in SURFACE_MAP.items():
        if marker in lower_tournament:
            surface = value
            break

    gender = ""
    if "Å¾enski" in lower_tour or "zenski" in lower_tour or "wta" in lower_tour:
        gender = "women"
    elif "muÅ¡ki" in lower_tour or "muski" in lower_tour or "atp" in lower_tour:
        gender = "men"

    if "challenger" in lower_tour:
        tour_level = "challenger"
    elif "itf" in lower_tour:
        tour_level = "itf"
    elif "wta" in lower_tour:
        tour_level = "wta"
    elif "atp" in lower_tour:
        tour_level = "atp"
    elif "junior" in lower_tour:
        tour_level = "junior"
    else:
        tour_level = ""

    is_grand_slam = any(marker in lower_tournament for marker in GRAND_SLAM_MARKERS)
    if is_grand_slam:
        tour_level = "grand_slam"

    best_of = 5 if (sets_1 + sets_2 > 3 or (is_grand_slam and gender == "men")) else 3

    tournament = re.sub(
        r",\s*(zemlja|trava|tvrda|hard|tepih)\s*$",
        "",
        clean_str(tournament_line),
        flags=re.IGNORECASE,
    )

    return {
        "raw_header_text": f"{tournament_line}\n{tour_line}".strip(),
        "tournament": tournament,
        "tour_level": tour_level,
        "gender": gender,
        "surface": surface,
        "indoor": True if "dvorana" in lower_tournament else None,
        "best_of": best_of,
        "is_grand_slam": is_grand_slam,
    }


def parse_set_scores(number_tokens, total_sets):
    out = []
    pos = 0

    for set_number in range(1, total_sets + 1):
        if pos + 1 >= len(number_tokens):
            return None

        if pos + 3 < len(number_tokens):
            a_games, a_tb, b_games, b_tb = number_tokens[pos:pos + 4]

            if (a_games, b_games) in ((7, 6), (6, 7)):
                out.append({
                    "set_number": set_number,
                    "p1_games": a_games,
                    "p2_games": b_games,
                    "tiebreak": True,
                    "tiebreak_p1": a_tb,
                    "tiebreak_p2": b_tb,
                })
                pos += 4
                continue

        a_games, b_games = number_tokens[pos:pos + 2]
        out.append({
            "set_number": set_number,
            "p1_games": a_games,
            "p2_games": b_games,
            "tiebreak": (a_games, b_games) in ((7, 6), (6, 7)),
        })
        pos += 2

    return out


def parse_body_text(body_text):
    lines = [clean_str(line) for line in str(body_text or "").splitlines() if clean_str(line)]

    matches = []
    tournament_line = ""
    tour_line = ""
    i = 0

    while i < len(lines):
        line = lines[i]
        upper = line.upper()

        if any(marker in upper for marker in TOUR_MARKERS) and "SINGL" in upper:
            tour_line = line

            for back in range(i - 1, max(-1, i - 5), -1):
                candidate = lines[back]
                if candidate.lower() in {"Å¾drijeb", "zdrijeb", "tablica", "draw"}:
                    continue
                if any(word in candidate.lower() for word in ["reklama", "raspored", "gotovo", "uÅ¾ivo", "uzivo"]):
                    continue
                tournament_line = candidate
                break

            i += 1
            continue

        if line.lower() not in FINISHED_WORDS:
            i += 1
            continue

        if i + 6 >= len(lines):
            break

        p1 = lines[i + 1]
        p2 = lines[i + 2]

        if not is_int_text(lines[i + 3]) or not is_int_text(lines[i + 4]):
            i += 1
            continue

        sets_1 = int(lines[i + 3])
        sets_2 = int(lines[i + 4])
        total_sets = sets_1 + sets_2

        if total_sets <= 0 or total_sets > 5:
            i += 1
            continue

        nums = []
        j = i + 5

        while j < len(lines) and is_int_text(lines[j]) and len(nums) < 20:
            nums.append(int(lines[j]))
            j += 1

        set_scores = parse_set_scores(nums, total_sets)

        if not set_scores or len(set_scores) != total_sets:
            i += 1
            continue

        metadata = parse_metadata(tournament_line, tour_line, sets_1, sets_2)
        winner = p1 if sets_1 > sets_2 else p2
        loser = p2 if sets_1 > sets_2 else p1

        row = {
            "match_id": "",
            "match_url": "",
            "source": "rezultati_daily_results_text_fallback",
            "date": today_iso(),
            "player_1": p1,
            "player_2": p2,
            "match": f"{p1} - {p2}",
            "winner": winner,
            "loser": loser,
            "sets_1": sets_1,
            "sets_2": sets_2,
            "final_score": f"{sets_1}-{sets_2}",
            "set_scores": set_scores,
            "completed": True,
            "retired": False,
            "walkover": False,
            "raw_parent_text": "\n".join(lines[i:j]),
        }

        row.update(metadata)
        row.update(derive_from_sets(set_scores, sets_1, sets_2))
        matches.append(row)

        i = j

    return matches


def archive_key(row):
    return clean_str(row.get("match_id")) or clean_str(row.get("match_url")) or "|".join([
        clean_str(row.get("date")),
        clean_str(row.get("tournament")),
        clean_str(row.get("player_1")),
        clean_str(row.get("player_2")),
        clean_str(row.get("final_score")),
        clean_str(row.get("first_set_score")),
    ])


def merge_rows(old, new):
    merged = dict(old)

    for key, value in new.items():
        if merged.get(key) in (None, "", [], {}) and value not in (None, "", [], {}):
            merged[key] = value

    return merged


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
            by_key[key] = merge_rows(by_key[key], row)

    return [by_key[key] for key in order]


def load_archive_rows():
    payload = load_json(ARCHIVE_FILE, {})

    if isinstance(payload, dict):
        rows = payload.get("matches") or payload.get("scorelines") or []
        return rows if isinstance(rows, list) else []

    return payload if isinstance(payload, list) else []


def save_outputs(daily_rows):
    archive_before = load_archive_rows()
    archive_after = merge_archive(archive_before, daily_rows)

    daily_payload = {
        "generated_at": now_iso(),
        "source_url": DAILY_URL,
        "counts": {"daily_scorelines": len(daily_rows)},
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

    metadata_counts = {
        "with_surface": sum(1 for row in archive_after if row.get("surface")),
        "with_gender": sum(1 for row in archive_after if row.get("gender")),
        "with_tour_level": sum(1 for row in archive_after if row.get("tour_level")),
        "grand_slam": sum(1 for row in archive_after if row.get("is_grand_slam") is True),
        "best_of_5": sum(1 for row in archive_after if row.get("best_of") == 5),
    }

    report = {
        "generated_at": now_iso(),
        "source_url": DAILY_URL,
        "counts": archive_payload["counts"],
        "metadata_counts": metadata_counts,
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
                timezone_id=TIMEZONE,
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
            )

            page = context.new_page()
            go_to_tennis_page(page)
            body_text = save_debug(page)
            daily_rows = parse_body_text(body_text)

            context.close()
            browser.close()

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    report = save_outputs(daily_rows)

    if error:
        report["error"] = error
        save_json(REPORT_FILE, report)

    print("")
    print("REZULTATI DAILY RESULTS DONE")
    print(report["counts"])
    print("Metadata:", report["metadata_counts"])
    print(f"Daily:   {DAILY_OUTPUT_FILE}")
    print(f"Archive: {ARCHIVE_FILE}")
    print(f"Report:  {REPORT_FILE}")

    if error:
        print(f"WARNING: {error}")

    print("")


if __name__ == "__main__":
    main()
