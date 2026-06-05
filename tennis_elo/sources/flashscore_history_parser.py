import re
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

from tennis_elo.config import (
    FLASH_HISTORY_RAW_FILE,
    WATCHLIST_FILE,
    RAW_DIR,
)
from tennis_elo.utils import now_iso, save_json, save_text, clean_str, normalize_text


DEBUG_DIR = RAW_DIR / "debug"

BLOCK_TEXTS = [
    "The requested page can't be displayed",
    "Please try again later",
    "Access denied",
]

DATE_RE = re.compile(r"\b\d{2}\.\d{2}\.\d{2}\b")
SECTION_RE = re.compile(
    r"POSLJEDNJI\s+ME[ČC]EVI:\s*(?P<player>[^\n\r]+)",
    re.IGNORECASE,
)

# Example collapsed rows from rezultati.com / flashscore H2H:
# 05.06.26 TYL Walton A. Martin A. 2 0 P
# 04.06.26 TYL Ardila L. Legout T. 0 2 P
ROW_RE = re.compile(
    r"(?P<date>\d{2}\.\d{2}\.\d{2})\s+"
    r"(?P<tournament>[A-ZČŠŽĐĆ]{2,8})\s+"
    r"(?P<players>.+?)\s+"
    r"(?P<sets_1>[0-7])\s+"
    r"(?P<sets_2>[0-7])\s+"
    r"(?P<result>[PIWL])"
    r"(?=\s+\d{2}\.\d{2}\.\d{2}|\s+Prika|\s+Show|\s+POSLJEDNJI|\s*$)",
    re.IGNORECASE,
)


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


def normalize_flashscore_url(url):
    url = clean_str(url)
    if not url:
        return ""

    return url


def click_show_more(page, max_clicks=12):
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
                    locator.click(timeout=2500)
                    page.wait_for_timeout(1300)
                    clicked += 1
                    did_click = True
                    break
            except Exception:
                continue

        if not did_click:
            break

    return clicked


def detect_surface_filter_from_url(url):
    lower = clean_str(url).lower()

    if "sve-podloge" in lower or "overall" in lower:
        return "all"
    if "tvrda-podloga" in lower or "hard" in lower:
        return "hard"
    if "zemlja" in lower or "clay" in lower:
        return "clay"
    if "trava" in lower or "grass" in lower:
        return "grass"

    return ""


def detect_surface_from_text(text):
    lower = str(text or "").lower()

    if "zemlja" in lower or "clay" in lower:
        return "clay"
    if "tvrda podloga" in lower or "hard" in lower:
        return "hard"
    if "trava" in lower or "grass" in lower:
        return "grass"
    if "indoor" in lower:
        return "indoor"

    return ""


def clean_player_name(value):
    value = clean_str(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def names_equalish(a, b):
    return normalize_text(a) == normalize_text(b)


def find_focus_in_player_text(player_text, focus_player):
    """
    Splits 'Walton A. Martin A.' with focus Walton A. into:
      player_1=Walton A., player_2=Martin A.

    Splits 'Ardila L. Legout T.' with focus Legout T. into:
      player_1=Ardila L., player_2=Legout T.

    This is intentionally strict around the known section player.
    """
    player_text = clean_player_name(player_text)
    focus_player = clean_player_name(focus_player)

    if not player_text or not focus_player:
        return "", "", ""

    # Direct exact edge match.
    if player_text == focus_player:
        return focus_player, "", "focus_only"

    if player_text.startswith(focus_player + " "):
        opponent = clean_player_name(player_text[len(focus_player):])
        return focus_player, opponent, "focus_first_exact"

    if player_text.endswith(" " + focus_player):
        opponent = clean_player_name(player_text[: -len(focus_player)])
        return opponent, focus_player, "focus_second_exact"

    # Sometimes punctuation differs. Use normalized text and map by token length.
    p_tokens = player_text.split()
    focus_tokens = focus_player.split()

    norm_focus_tokens = [normalize_text(x) for x in focus_tokens]
    norm_p_tokens = [normalize_text(x) for x in p_tokens]

    for i in range(0, len(norm_p_tokens) - len(norm_focus_tokens) + 1):
        if norm_p_tokens[i:i + len(norm_focus_tokens)] == norm_focus_tokens:
            before = clean_player_name(" ".join(p_tokens[:i]))
            focus = clean_player_name(" ".join(p_tokens[i:i + len(focus_tokens)]))
            after = clean_player_name(" ".join(p_tokens[i + len(focus_tokens):]))

            if before and not after:
                return before, focus, "focus_second_tokens"

            if after and not before:
                return focus, after, "focus_first_tokens"

            if before and after:
                # Rare, but keep full context.
                return before, focus, "focus_middle_tokens"

    # Surname fallback: focus "Walton A." inside "Walton A Martin A" etc.
    focus_norm = normalize_text(focus_player)
    player_norm = normalize_text(player_text)

    if focus_norm and player_norm.startswith(focus_norm + " "):
        # Approximate with token count.
        n = len(focus_norm.split())
        focus = clean_player_name(" ".join(p_tokens[:n]))
        opponent = clean_player_name(" ".join(p_tokens[n:]))
        return focus, opponent, "focus_first_normalized"

    if focus_norm and player_norm.endswith(" " + focus_norm):
        n = len(focus_norm.split())
        focus = clean_player_name(" ".join(p_tokens[-n:]))
        opponent = clean_player_name(" ".join(p_tokens[:-n]))
        return opponent, focus, "focus_second_normalized"

    return "", "", "focus_not_found"


def infer_winner_loser(player_1, player_2, sets_1, sets_2, focus_player, result_marker):
    sets_1 = int(sets_1)
    sets_2 = int(sets_2)

    # Primary: set score.
    if sets_1 > sets_2:
        return player_1, player_2

    if sets_2 > sets_1:
        return player_2, player_1

    # Fallback: P/W = focus player won, I/L = focus player lost.
    marker = clean_str(result_marker).upper()

    if marker in {"P", "W"}:
        if names_equalish(player_1, focus_player):
            return player_1, player_2
        if names_equalish(player_2, focus_player):
            return player_2, player_1

    if marker in {"I", "L"}:
        if names_equalish(player_1, focus_player):
            return player_2, player_1
        if names_equalish(player_2, focus_player):
            return player_1, player_2

    return "", ""


def compact_section_text(text):
    lines = []
    for line in str(text or "").splitlines():
        line = clean_str(line)
        if not line:
            continue

        # Remove noisy labels that can break row parsing.
        if line.lower() in {
            "detalji",
            "tečajevi",
            "tecajevi",
            "omjer",
            "ždrijeb",
            "zdrijeb",
            "sve podloge",
            "trava",
            "tvrda podloga",
            "zemlja",
        }:
            continue

        lines.append(line)

    return " ".join(lines)


def extract_sections(text):
    """
    Returns list of:
    {
      "focus_player": "Walton A.",
      "text": "... rows until next POSLJEDNJI MEČEVI ..."
    }
    """
    text = str(text or "")
    matches = list(SECTION_RE.finditer(text))

    sections = []

    for idx, match in enumerate(matches):
        focus_player = clean_player_name(match.group("player"))

        # Header player can sometimes contain trailing UI text. Keep first line only.
        focus_player = focus_player.split("|")[0].strip()

        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section_text = text[start:end]

        sections.append({
            "focus_player": focus_player,
            "text": section_text,
        })

    return sections


def parse_rows_from_section(section, surface_filter="", source_url=""):
    focus_player = clean_player_name(section.get("focus_player"))
    collapsed = compact_section_text(section.get("text", ""))

    rows = []

    for match in ROW_RE.finditer(collapsed):
        date = clean_str(match.group("date"))
        tournament = clean_str(match.group("tournament")).upper()
        players_blob = clean_player_name(match.group("players"))
        sets_1 = clean_str(match.group("sets_1"))
        sets_2 = clean_str(match.group("sets_2"))
        result_marker = clean_str(match.group("result")).upper()

        player_1, player_2, split_method = find_focus_in_player_text(players_blob, focus_player)

        winner = ""
        loser = ""

        if player_1 and player_2:
            winner, loser = infer_winner_loser(
                player_1,
                player_2,
                sets_1,
                sets_2,
                focus_player,
                result_marker,
            )

        parse_status = "ok" if player_1 and player_2 and winner and loser else "needs_review"

        rows.append({
            "date": date,
            "tournament_code": tournament,
            "surface": "" if surface_filter == "all" else surface_filter,
            "surface_filter": surface_filter,
            "focus_player": focus_player,
            "player_1": player_1,
            "player_2": player_2,
            "sets_1": int(sets_1),
            "sets_2": int(sets_2),
            "winner": winner,
            "loser": loser,
            "result_marker": result_marker,
            "score": f"{sets_1}-{sets_2}",
            "raw_players": players_blob,
            "raw": match.group(0),
            "split_method": split_method,
            "parse_status": parse_status,
            "source": "flashscore",
            "source_url": source_url,
            "parser": "rezultati_h2h_text_v2",
        })

    return rows


def extract_history_rows(text, url):
    surface_filter = detect_surface_filter_from_url(url)
    if not surface_filter:
        surface_filter = detect_surface_from_text(text)

    sections = extract_sections(text)

    rows = []

    for section in sections:
        rows.extend(
            parse_rows_from_section(
                section,
                surface_filter=surface_filter,
                source_url=url,
            )
        )

    # Dedupe same visible row from repeated loaded sections.
    deduped = []
    seen = set()

    for row in rows:
        key = (
            row.get("date"),
            normalize_text(row.get("player_1")),
            normalize_text(row.get("player_2")),
            row.get("score"),
            row.get("tournament_code"),
            row.get("focus_player"),
        )

        if key in seen:
            continue

        seen.add(key)
        deduped.append(row)

    return deduped, sections, surface_filter


def parse_one_url(page, url, index):
    started_at = now_iso()
    url = normalize_flashscore_url(url)

    result = {
        "url": url,
        "parsed_at": started_at,
        "blocked": False,
        "error": "",
        "show_more_clicked": 0,
        "surface_filter": detect_surface_filter_from_url(url),
        "sections_found": 0,
        "rows": [],
        "debug_files": {},
    }

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=35000)
        page.wait_for_timeout(5500)

        result["show_more_clicked"] = click_show_more(page, max_clicks=12)

        html = page.content()
        text = page.locator("body").inner_text(timeout=10000)

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

        rows, sections, surface_filter = extract_history_rows(text, url)
        result["rows"] = rows
        result["sections_found"] = len(sections)
        result["surface_filter"] = surface_filter

        if not sections:
            result["error"] = "no POSLJEDNJI MECEVI sections found"
            return result

        if not rows:
            result["error"] = "no parsed history rows found"
            return result

        review_count = sum(1 for row in rows if row.get("parse_status") != "ok")
        if review_count:
            result["error"] = f"{review_count} rows need review"

        return result

    except Exception as e:
        result["error"] = str(e)
        return result


def calc_counts(pages):
    urls = len(pages)
    blocked = sum(1 for page in pages if page.get("blocked"))
    errors = sum(1 for page in pages if clean_str(page.get("error")))
    raw_rows = sum(len(page.get("rows") or []) for page in pages)
    ok_rows = sum(
        1
        for page in pages
        for row in (page.get("rows") or [])
        if row.get("parse_status") == "ok"
    )
    review_rows = raw_rows - ok_rows
    sections_found = sum(int(page.get("sections_found") or 0) for page in pages)

    return {
        "urls": urls,
        "blocked": blocked,
        "errors": errors,
        "sections_found": sections_found,
        "raw_rows": raw_rows,
        "ok_rows": ok_rows,
        "review_rows": review_rows,
    }


def main():
    urls = read_urls()

    output = {
        "generated_at": now_iso(),
        "source_file": str(WATCHLIST_FILE),
        "counts": {
            "urls": len(urls),
            "blocked": 0,
            "errors": 0,
            "sections_found": 0,
            "raw_rows": 0,
            "ok_rows": 0,
            "review_rows": 0,
        },
        "pages": [],
    }

    if not urls:
        save_json(FLASH_HISTORY_RAW_FILE, output)
        print("No URLs found in config/flashscore_urls.txt")
        return

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

        for i, url in enumerate(urls, start=1):
            print(f"Parsing {i}/{len(urls)}: {url}")
            page_result = parse_one_url(page, url, i)
            output["pages"].append(page_result)

        context.close()
        browser.close()

    output["counts"] = calc_counts(output["pages"])
    save_json(FLASH_HISTORY_RAW_FILE, output)

    print("")
    print("FLASHSCORE HISTORY PARSER DONE")
    print(output["counts"])
    print(f"Output: {FLASH_HISTORY_RAW_FILE}")
    print("")


if __name__ == "__main__":
    main()
