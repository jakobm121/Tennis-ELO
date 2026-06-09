import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from playwright.sync_api import Page, sync_playwright

from tennis_elo.config import CANONICAL_MATCHES_FILE, ROOT_DIR
from tennis_elo.utils import clean_str, load_json, now_iso, save_json, save_text


OUTPUT_FILE = (
    ROOT_DIR
    / "data"
    / "enrichment"
    / "historical_match_metadata_pilot.json"
)
REPORT_FILE = (
    ROOT_DIR
    / "data"
    / "reports"
    / "historical_match_metadata_pilot_report.json"
)
DEBUG_DIR = (
    ROOT_DIR
    / "data"
    / "reports"
    / "historical_match_metadata_debug"
)

PILOT_LIMIT = int(
    os.getenv("HISTORICAL_ENRICHMENT_PILOT_LIMIT", "30")
)
MAX_SOURCE_URLS = int(
    os.getenv("HISTORICAL_ENRICHMENT_MAX_SOURCE_URLS", "12")
)
WAIT_MS = int(
    os.getenv("HISTORICAL_ENRICHMENT_WAIT_MS", "5000")
)
ROW_CLICK_WAIT_MS = int(
    os.getenv("HISTORICAL_ENRICHMENT_ROW_CLICK_WAIT_MS", "3000")
)

TARGET_PLAYERS = [
    value.strip().lower()
    for value in os.getenv(
        "HISTORICAL_ENRICHMENT_TARGET_PLAYERS",
        "Rehberg,Carballes Baena",
    ).split(",")
    if value.strip()
]

BLOCK_TEXTS = (
    "The requested page can't be displayed",
    "Please try again later",
    "Access denied",
)

TOUR_PATTERNS = [
    ("grand_slam", re.compile(
        r"\b(australian open|roland garros|french open|wimbledon|us open)\b",
        re.I,
    )),
    ("atp", re.compile(r"\bATP\b", re.I)),
    ("wta", re.compile(r"\bWTA\b", re.I)),
    ("challenger", re.compile(r"\bCHALLENGER\b", re.I)),
    ("itf", re.compile(r"\bITF\b", re.I)),
]

SURFACE_PATTERNS = [
    ("hard", re.compile(
        r"\b(tvrda podloga|hard court|hard)\b",
        re.I,
    )),
    ("clay", re.compile(
        r"\b(zemlja|glina|clay)\b",
        re.I,
    )),
    ("grass", re.compile(
        r"\b(trava|grass)\b",
        re.I,
    )),
    ("carpet", re.compile(
        r"\b(tepih|carpet)\b",
        re.I,
    )),
]


def ensure_dirs() -> None:
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    DEBUG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def parse_date(value: Any):
    text = clean_str(value)

    for fmt in (
        "%d.%m.%y",
        "%d.%m.%Y",
        "%Y-%m-%d",
        "%Y/%m/%d",
    ):
        try:
            return datetime.strptime(
                text,
                fmt,
            ).date()
        except ValueError:
            pass

    return None


def compact(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        clean_str(value).lower(),
    )


def surname(value: Any) -> str:
    parts = re.findall(
        r"[a-z0-9]+",
        clean_str(value).lower(),
    )

    if not parts:
        return ""

    if len(parts) == 1:
        return parts[0]

    if len(parts[0]) <= 2:
        return "".join(parts[1:])

    return "".join(parts[:-1])


def accept_cookies(page: Page) -> None:
    for text in (
        "PrihvaÄam",
        "Prihvati",
        "SlaÅ¾em se",
        "Accept all",
        "I accept",
        "Accept",
        "OK",
    ):
        try:
            locator = page.get_by_text(
                text,
                exact=False,
            ).first

            if (
                locator.count() > 0
                and locator.is_visible(timeout=800)
            ):
                locator.click(timeout=2000)
                page.wait_for_timeout(800)
                return
        except Exception:
            pass


def page_text(page: Page) -> str:
    try:
        return page.locator("body").inner_text(
            timeout=10000
        )
    except Exception:
        return ""


def is_blocked(text: str) -> bool:
    return any(
        value.lower() in text.lower()
        for value in BLOCK_TEXTS
    )


def row_text_matches(
    row_text: str,
    match: dict[str, Any],
) -> bool:
    normalized = compact(row_text)
    date_value = parse_date(match.get("date"))

    date_variants = set()

    if date_value:
        date_variants = {
            compact(
                date_value.strftime("%d.%m.%y")
            ),
            compact(
                date_value.strftime("%d.%m.%Y")
            ),
            compact(
                date_value.strftime("%d/%m/%y")
            ),
        }

    player_1_surname = surname(
        match.get("player_1")
    )
    player_2_surname = surname(
        match.get("player_2")
    )

    has_date = (
        not date_variants
        or any(
            value and value in normalized
            for value in date_variants
        )
    )
    has_player_1 = (
        player_1_surname
        and player_1_surname in normalized
    )
    has_player_2 = (
        player_2_surname
        and player_2_surname in normalized
    )

    return has_date and has_player_1 and has_player_2


def detail_candidates(page: Page):
    selectors = [
        '[id^="g_2_"]',
        '[class*="h2h__row"]',
        '[class*="event__match"]',
        '[class*="eventRow"]',
        'a[href*="/utakmica/tenis/"]',
    ]

    seen = set()
    output = []

    for selector in selectors:
        try:
            locator = page.locator(selector)
            count = min(locator.count(), 500)
        except Exception:
            continue

        for index in range(count):
            try:
                item = locator.nth(index)
                text = clean_str(
                    item.inner_text(timeout=800)
                )
            except Exception:
                text = ""

            try:
                href = item.get_attribute("href")
            except Exception:
                href = None

            key = (
                selector,
                index,
                text[:200],
                href,
            )

            if key in seen:
                continue

            seen.add(key)
            output.append(
                {
                    "selector": selector,
                    "index": index,
                    "text": text,
                    "href": href,
                }
            )

    return output


def extract_metadata(
    text: str,
    page_url: str,
) -> dict[str, Any]:
    cleaned_lines = [
        clean_str(line)
        for line in text.splitlines()
        if clean_str(line)
    ]

    tour_level = "unknown"

    for level, pattern in TOUR_PATTERNS:
        if pattern.search(text):
            tour_level = level
            break

    surface = "unknown"

    for surface_name, pattern in SURFACE_PATTERNS:
        if pattern.search(text):
            surface = surface_name
            break

    header_line = ""

    for line in cleaned_lines[:80]:
        upper = line.upper()

        if any(
            token in upper
            for token in (
                "ATP",
                "WTA",
                "CHALLENGER",
                "ITF",
                "AUSTRALIAN OPEN",
                "ROLAND GARROS",
                "FRENCH OPEN",
                "WIMBLEDON",
                "US OPEN",
            )
        ):
            header_line = line
            break

    tournament_name = ""
    round_name = ""

    if header_line:
        parts = [
            clean_str(part)
            for part in re.split(
                r"\s*[>âº|]\s*",
                header_line,
            )
            if clean_str(part)
        ]

        if parts:
            tournament_name = parts[-1]

        round_match = re.search(
            r"\b("
            r"finale|polufinale|Äetvrtfinale|"
            r"1/\d+\s*finala|"
            r"round\s+\d+|qualification|kvalifikacije|"
            r"\d+\.\s*kolo"
            r")\b",
            header_line,
            re.I,
        )

        if round_match:
            round_name = round_match.group(1)

    return {
        "detail_url": page_url,
        "header_text": header_line,
        "tour_level": tour_level,
        "surface_from_detail": surface,
        "tournament_name": tournament_name,
        "round": round_name,
    }


def open_candidate(
    page: Page,
    source_url: str,
    candidate: dict[str, Any],
) -> tuple[bool, str]:
    href = clean_str(candidate.get("href"))

    if href:
        target = urljoin(
            source_url,
            href,
        )

        try:
            page.goto(
                target,
                wait_until="domcontentloaded",
                timeout=45000,
            )
            page.wait_for_timeout(
                ROW_CLICK_WAIT_MS
            )
            return True, ""
        except Exception as exc:
            return False, (
                f"direct_navigation_failed: "
                f"{type(exc).__name__}: {exc}"
            )

    try:
        selector = candidate["selector"]
        index = candidate["index"]
        item = page.locator(selector).nth(index)
        item.scroll_into_view_if_needed(
            timeout=3000
        )
        item.click(timeout=5000)
        page.wait_for_timeout(
            ROW_CLICK_WAIT_MS
        )
        return True, ""
    except Exception as exc:
        return False, (
            f"row_click_failed: "
            f"{type(exc).__name__}: {exc}"
        )


def select_pilot_matches(
    matches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in matches
        if isinstance(row, dict)
        and clean_str(row.get("source_url"))
        and parse_date(row.get("date"))
    ]

    priority = []
    normal = []

    for row in candidates:
        names = " ".join(
            (
                clean_str(row.get("player_1")),
                clean_str(row.get("player_2")),
            )
        ).lower()

        if any(
            target in names
            for target in TARGET_PLAYERS
        ):
            priority.append(row)
        else:
            normal.append(row)

    selected = []
    seen_ids = set()
    source_counts = {}

    for row in priority + normal:
        canonical_id = clean_str(
            row.get("canonical_match_id")
        )
        source_url = clean_str(
            row.get("source_url")
        )

        if not canonical_id or canonical_id in seen_ids:
            continue

        if (
            source_url not in source_counts
            and len(source_counts) >= MAX_SOURCE_URLS
        ):
            continue

        source_counts[source_url] = (
            source_counts.get(source_url, 0) + 1
        )
        seen_ids.add(canonical_id)
        selected.append(row)

        if len(selected) >= PILOT_LIMIT:
            break

    return selected


def save_debug(
    page: Page,
    slug: str,
) -> None:
    try:
        save_text(
            DEBUG_DIR / f"{slug}.txt",
            page_text(page),
        )
        save_text(
            DEBUG_DIR / f"{slug}.html",
            page.content(),
        )
        page.screenshot(
            path=str(
                DEBUG_DIR / f"{slug}.png"
            ),
            full_page=True,
        )
    except Exception:
        pass


def process_match(
    page: Page,
    match: dict[str, Any],
) -> dict[str, Any]:
    source_url = clean_str(
        match.get("source_url")
    )
    canonical_id = clean_str(
        match.get("canonical_match_id")
    )

    base = {
        "canonical_match_id": canonical_id,
        "date": match.get("date"),
        "player_1": match.get("player_1"),
        "player_2": match.get("player_2"),
        "winner": match.get("winner"),
        "score": match.get("score"),
        "tournament_code": match.get(
            "tournament_code"
        ),
        "surface_existing": match.get("surface"),
        "source_url": source_url,
    }

    try:
        page.goto(
            source_url,
            wait_until="domcontentloaded",
            timeout=45000,
        )
        page.wait_for_timeout(WAIT_MS)
        accept_cookies(page)
    except Exception as exc:
        return {
            **base,
            "status": "source_page_failed",
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }

    text = page_text(page)

    if is_blocked(text):
        return {
            **base,
            "status": "blocked",
            "error": "Rezultati page appears blocked",
        }

    candidates = detail_candidates(page)
    matched = [
        candidate
        for candidate in candidates
        if row_text_matches(
            candidate.get("text", ""),
            match,
        )
    ]

    if not matched:
        slug = compact(canonical_id)[:80]
        save_debug(
            page,
            f"row_not_found_{slug}",
        )

        return {
            **base,
            "status": "row_not_found",
            "candidate_rows_seen": len(candidates),
            "candidate_examples": candidates[:15],
        }

    errors = []

    for candidate_index, candidate in enumerate(
        matched[:5]
    ):
        try:
            page.goto(
                source_url,
                wait_until="domcontentloaded",
                timeout=45000,
            )
            page.wait_for_timeout(WAIT_MS)
        except Exception:
            pass

        opened, error = open_candidate(
            page,
            source_url,
            candidate,
        )

        if not opened:
            errors.append(error)
            continue

        detail_text = page_text(page)
        metadata = extract_metadata(
            detail_text,
            page.url,
        )

        if (
            metadata["tour_level"] != "unknown"
            or metadata["header_text"]
        ):
            return {
                **base,
                "status": "matched",
                "matched_row_text": candidate.get(
                    "text"
                ),
                "matched_row_href": candidate.get(
                    "href"
                ),
                "candidate_index": candidate_index,
                **metadata,
            }

        slug = compact(canonical_id)[:80]
        save_debug(
            page,
            f"detail_missing_metadata_{slug}",
        )
        errors.append(
            "detail_page_missing_category"
        )

    return {
        **base,
        "status": "detail_page_missing_category",
        "matched_rows": matched[:5],
        "errors": errors,
    }


def main() -> None:
    ensure_dirs()

    payload = load_json(
        CANONICAL_MATCHES_FILE,
        {},
    )
    matches = (
        payload.get("matches", [])
        if isinstance(payload, dict)
        else []
    )

    if not isinstance(matches, list):
        matches = []

    selected = select_pilot_matches(matches)
    results = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        context = browser.new_context(
            viewport={
                "width": 1400,
                "height": 1200,
            },
            locale="hr-HR",
            timezone_id="Europe/Ljubljana",
            user_agent=(
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        for index, match in enumerate(
            selected,
            start=1,
        ):
            print(
                f"[{index}/{len(selected)}]",
                match.get("date"),
                match.get("player_1"),
                "-",
                match.get("player_2"),
            )
            result = process_match(
                page,
                match,
            )
            results.append(result)
            print(
                " ->",
                result.get("status"),
                result.get("tour_level", ""),
                result.get("tournament_name", ""),
            )

        context.close()
        browser.close()

    status_counts = {}

    for row in results:
        status = clean_str(
            row.get("status")
        ) or "unknown"
        status_counts[status] = (
            status_counts.get(status, 0) + 1
        )

    matched_rows = [
        row
        for row in results
        if row.get("status") == "matched"
    ]

    output = {
        "generated_at": now_iso(),
        "source_file": str(
            CANONICAL_MATCHES_FILE
        ),
        "settings": {
            "pilot_limit": PILOT_LIMIT,
            "max_source_urls": MAX_SOURCE_URLS,
            "target_players": TARGET_PLAYERS,
            "wait_ms": WAIT_MS,
            "row_click_wait_ms": ROW_CLICK_WAIT_MS,
        },
        "summary": {
            "selected_matches": len(selected),
            "matched": len(matched_rows),
            "success_rate": (
                round(
                    len(matched_rows)
                    / len(selected),
                    4,
                )
                if selected
                else None
            ),
            "status_counts": status_counts,
        },
        "results": results,
    }

    report = {
        "generated_at": now_iso(),
        "settings": output["settings"],
        "summary": output["summary"],
        "matched_examples": matched_rows[:20],
        "failed_examples": [
            row
            for row in results
            if row.get("status") != "matched"
        ][:30],
    }

    save_json(OUTPUT_FILE, output)
    save_json(REPORT_FILE, report)

    print("")
    print("HISTORICAL MATCH ENRICHMENT PILOT DONE")
    print("SUMMARY:", output["summary"])

    for row in matched_rows[:20]:
        print(
            "MATCHED:",
            row.get("date"),
            row.get("player_1"),
            "-",
            row.get("player_2"),
            "|",
            row.get("tour_level"),
            "|",
            row.get("tournament_name"),
            "|",
            row.get("detail_url"),
        )

    print(f"Output: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")
    print(f"Debug:  {DEBUG_DIR}")
    print("")


if __name__ == "__main__":
    main()
