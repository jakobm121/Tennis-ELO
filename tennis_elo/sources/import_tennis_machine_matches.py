import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tennis_elo.config import ROOT_DIR
from tennis_elo.utils import now_iso, save_json, clean_str


OUTPUT_FILE = ROOT_DIR / "data" / "input" / "matches_today.json"
REPORT_FILE = ROOT_DIR / "data" / "reports" / "import_tennis_machine_matches_report.json"

# We try multiple forms because GitHub raw URLs can differ depending on branch/path.
DEFAULT_TENNIS_MACHINE_URLS = [
    "https://raw.githubusercontent.com/jakobm121/Tennis-Machine/refs/heads/main/tennis_machine/data/processed/flashscore_matches.json",
    "https://raw.githubusercontent.com/jakobm121/Tennis-Machine/main/tennis_machine/data/processed/flashscore_matches.json",
]

MAX_MATCHES = int(os.getenv("MAX_TENNIS_MACHINE_MATCHES", "10"))

INCLUDE_STATUSES = {
    x.strip().lower()
    for x in os.getenv("TM_INCLUDE_STATUSES", "scheduled,live").split(",")
    if x.strip()
}

INCLUDE_TOUR_LEVELS = {
    x.strip().lower()
    for x in os.getenv("TM_INCLUDE_TOUR_LEVELS", "atp,wta,itf,challenger").split(",")
    if x.strip()
}

PRIORITY_TOUR_LEVELS = {
    "atp": 100,
    "wta": 95,
    "challenger": 85,
    "itf": 70,
}

PRIORITY_STATUSES = {
    "live": 100,
    "scheduled": 80,
    "finished": 20,
}


def load_json_from_url(url):
    req = Request(
        url,
        headers={
            "User-Agent": "Tennis-ELO-Machine/1.0",
            "Accept": "application/json,text/plain,*/*",
        },
    )

    with urlopen(req, timeout=45) as response:
        raw = response.read().decode("utf-8")

    return json.loads(raw)


def load_json_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def try_load_url(url):
    try:
        payload = load_json_from_url(url)
        return payload, ""
    except HTTPError as e:
        return None, f"HTTP {e.code}: {e.reason}"
    except URLError as e:
        return None, f"URL error: {e.reason}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def load_existing_matches_today_if_any():
    if not OUTPUT_FILE.exists():
        return None

    try:
        payload = load_json_from_file(OUTPUT_FILE)
        rows = rows_from_payload(payload)
        if rows:
            return payload
    except Exception:
        return None

    return None


def load_tennis_machine_payload():
    """
    Sources, in order:

    1. TM_MATCHES_LOCAL_FILE env var
    2. TM_MATCHES_URL env var
    3. Built-in GitHub raw URL candidates
    4. Existing data/input/matches_today.json as fallback, so pipeline does not die
    """
    attempts = []

    local_file = clean_str(os.getenv("TM_MATCHES_LOCAL_FILE"))
    if local_file:
        try:
            payload = load_json_from_file(local_file)
            return payload, local_file, attempts
        except Exception as e:
            attempts.append({
                "source": local_file,
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
            })

    env_url = clean_str(os.getenv("TM_MATCHES_URL"))
    urls = []

    if env_url:
        urls.append(env_url)

    for url in DEFAULT_TENNIS_MACHINE_URLS:
        if url not in urls:
            urls.append(url)

    for url in urls:
        payload, error = try_load_url(url)
        attempts.append({
            "source": url,
            "ok": payload is not None,
            "error": error,
        })

        if payload is not None:
            return payload, url, attempts

    fallback = load_existing_matches_today_if_any()
    if fallback is not None:
        attempts.append({
            "source": str(OUTPUT_FILE),
            "ok": True,
            "error": "used existing matches_today.json fallback",
        })
        return fallback, str(OUTPUT_FILE), attempts

    raise RuntimeError(
        "Could not load Tennis-Machine matches from any URL and no existing "
        "data/input/matches_today.json fallback was available. See report attempts."
    )


def rows_from_payload(payload):
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ["matches", "rows", "data", "results"]:
            value = payload.get(key)
            if isinstance(value, list):
                return value

    return []


def normalize_rezultati_url(url):
    url = clean_str(url)

    if not url:
        return ""

    url = url.split("#", 1)[0]

    if "flashscore.com/match/tennis/" in url:
        url = url.replace(
            "https://www.flashscore.com/match/tennis/",
            "https://www.rezultati.com/utakmica/tenis/",
        )
        url = url.replace(
            "http://www.flashscore.com/match/tennis/",
            "https://www.rezultati.com/utakmica/tenis/",
        )

    if "rezultati.com/utakmica/tenis/" in url:
        return url

    return ""


def get_match_url(row):
    if not isinstance(row, dict):
        return ""

    for key in [
        "match_url",
        "url",
        "source_match_url",
        "flashscore_match_url",
        "rezultati_match_url",
    ]:
        value = normalize_rezultati_url(row.get(key))
        if value:
            return value

    value = normalize_rezultati_url(row.get("source_url"))
    if value:
        return value

    value = normalize_rezultati_url(row.get("odds_url"))
    if value:
        return value

    return ""


def is_singles_match(row):
    if not isinstance(row, dict):
        return False

    event_type = clean_str(row.get("event_type")).lower()
    if event_type and event_type != "singles":
        return False

    match_name = clean_str(row.get("match"))
    player_1 = clean_str(row.get("player_1"))
    player_2 = clean_str(row.get("player_2"))

    joined = f"{match_name} {player_1} {player_2}"
    if "/" in joined:
        return False

    if not player_1 or not player_2:
        if " - " not in match_name:
            return False

    return True


def row_priority(row):
    status = clean_str(row.get("status")).lower()
    tour = clean_str(row.get("tour_level")).lower()

    score = 0
    score += PRIORITY_STATUSES.get(status, 0)
    score += PRIORITY_TOUR_LEVELS.get(tour, 50)

    if get_match_url(row):
        score += 1000

    surface = clean_str(row.get("surface")).lower()
    if surface in {"clay", "hard", "grass"}:
        score += 25

    return score


def convert_row(row):
    match_url = get_match_url(row)

    player_1 = clean_str(row.get("player_1"))
    player_2 = clean_str(row.get("player_2"))
    match_name = clean_str(row.get("match"))

    if not match_name and player_1 and player_2:
        match_name = f"{player_1} - {player_2}"

    return {
        "match": match_name,
        "match_url": match_url,
        "date": clean_str(row.get("date")),
        "status": clean_str(row.get("status")),
        "tour_level": clean_str(row.get("tour_level")),
        "gender": clean_str(row.get("gender")),
        "tournament": clean_str(row.get("tournament")),
        "country": clean_str(row.get("country")),
        "surface": clean_str(row.get("surface")),
        "player_1": player_1,
        "player_2": player_2,
        "source": "tennis_machine",
        "source_match_id": clean_str(row.get("match_id")),
    }


def write_empty_output(source, attempts, skipped):
    output = {
        "generated_at": now_iso(),
        "source_file": source,
        "max_matches": MAX_MATCHES,
        "matches": [],
    }

    report = {
        "generated_at": now_iso(),
        "source_file": source,
        "output_file": str(OUTPUT_FILE),
        "settings": {
            "max_matches": MAX_MATCHES,
            "include_statuses": sorted(INCLUDE_STATUSES),
            "include_tour_levels": sorted(INCLUDE_TOUR_LEVELS),
        },
        "counts": {
            "source_rows": 0,
            "candidates": 0,
            "selected": 0,
            "skipped": len(skipped),
        },
        "attempts": attempts,
        "selected": [],
        "skipped": skipped[:300],
        "note": (
            "No matches selected. If Tennis-Machine JSON has no match_url/odds_url, "
            "update Tennis-Machine parser to save match_url."
        ),
    }

    save_json(OUTPUT_FILE, output)
    save_json(REPORT_FILE, report)


def main():
    attempts = []
    skipped = []

    try:
        payload, source, attempts = load_tennis_machine_payload()
    except Exception as e:
        skipped.append({
            "reason": "load_failed",
            "error": str(e),
        })
        write_empty_output("unavailable", attempts, skipped)
        print("")
        print("IMPORT TENNIS MACHINE MATCHES DONE")
        print({"source_rows": 0, "candidates": 0, "selected": 0, "skipped": len(skipped)})
        print("WARNING: Could not load Tennis-Machine matches. Wrote empty output/report.")
        print(f"Report: {REPORT_FILE}")
        print("")
        return

    rows = rows_from_payload(payload)

    candidates = []

    for row in rows:
        if not isinstance(row, dict):
            skipped.append({"reason": "row_not_dict", "row": row})
            continue

        status = clean_str(row.get("status")).lower()
        tour = clean_str(row.get("tour_level")).lower()

        if INCLUDE_STATUSES and status and status not in INCLUDE_STATUSES:
            skipped.append({
                "reason": "status_filtered",
                "status": status,
                "match": row.get("match"),
            })
            continue

        if INCLUDE_TOUR_LEVELS and tour and tour not in INCLUDE_TOUR_LEVELS:
            skipped.append({
                "reason": "tour_filtered",
                "tour_level": tour,
                "match": row.get("match"),
            })
            continue

        if not is_singles_match(row):
            skipped.append({
                "reason": "not_singles_or_missing_players",
                "match": row.get("match"),
            })
            continue

        match_url = get_match_url(row)
        if not match_url:
            skipped.append({
                "reason": "missing_supported_match_url",
                "match": row.get("match"),
                "match_id": row.get("match_id"),
                "source_url": row.get("source_url"),
                "odds_url": row.get("odds_url"),
            })
            continue

        converted = convert_row(row)
        converted["_priority"] = row_priority(row)
        candidates.append(converted)

    candidates.sort(
        key=lambda r: (
            -int(r.get("_priority", 0)),
            str(r.get("tournament") or ""),
            str(r.get("match") or ""),
        )
    )

    selected = []
    seen_urls = set()

    for row in candidates:
        if len(selected) >= MAX_MATCHES:
            break

        url = row.get("match_url")
        if url in seen_urls:
            continue

        seen_urls.add(url)
        clean = dict(row)
        clean.pop("_priority", None)
        selected.append(clean)

    output = {
        "generated_at": now_iso(),
        "source_file": source,
        "max_matches": MAX_MATCHES,
        "matches": selected,
    }

    report = {
        "generated_at": now_iso(),
        "source_file": source,
        "output_file": str(OUTPUT_FILE),
        "settings": {
            "max_matches": MAX_MATCHES,
            "include_statuses": sorted(INCLUDE_STATUSES),
            "include_tour_levels": sorted(INCLUDE_TOUR_LEVELS),
        },
        "counts": {
            "source_rows": len(rows),
            "candidates": len(candidates),
            "selected": len(selected),
            "skipped": len(skipped),
        },
        "attempts": attempts,
        "selected": selected,
        "skipped": skipped[:300],
        "note": (
            "If selected is 0 but source_rows is >0, Tennis-Machine likely needs "
            "to save match_url/odds_url in flashscore_matches.json."
        ),
    }

    save_json(OUTPUT_FILE, output)
    save_json(REPORT_FILE, report)

    print("")
    print("IMPORT TENNIS MACHINE MATCHES DONE")
    print(report["counts"])
    print(f"Source: {source}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")

    if len(selected) == 0 and len(rows) > 0:
        print("")
        print("WARNING: Loaded Tennis-Machine JSON, but selected 0 matches.")
        print("Most likely flashscore_matches.json does not contain match_url/odds_url yet.")
        print("Fix Tennis-Machine parser to save match_url.")
    print("")


if __name__ == "__main__":
    main()
