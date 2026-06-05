import json
import os
import re
from urllib.request import Request, urlopen

from tennis_elo.config import ROOT_DIR
from tennis_elo.utils import now_iso, save_json, clean_str


OUTPUT_FILE = ROOT_DIR / "data" / "input" / "matches_today.json"
REPORT_FILE = ROOT_DIR / "data" / "reports" / "import_tennis_machine_matches_report.json"

DEFAULT_TENNIS_MACHINE_RAW_URL = (
    "https://raw.githubusercontent.com/jakobm121/Tennis-Machine/refs/heads/main/"
    "tennis_machine/data/processed/flashscore_matches.json"
)

# Conservative default. Raise later when parser/rate is stable.
MAX_MATCHES = int(os.getenv("MAX_TENNIS_MACHINE_MATCHES", "10"))

# Optional filters.
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

# Prefer matches that matter for our model expansion.
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


def load_tennis_machine_payload():
    """
    Sources, in order:

    1. TM_MATCHES_LOCAL_FILE env var
       Example:
       TM_MATCHES_LOCAL_FILE=data/external/flashscore_matches.json

    2. TM_MATCHES_URL env var

    3. Default raw GitHub Tennis-Machine processed matches file.
    """
    local_file = clean_str(os.getenv("TM_MATCHES_LOCAL_FILE"))

    if local_file:
        payload = load_json_from_file(local_file)
        return payload, local_file

    url = clean_str(os.getenv("TM_MATCHES_URL")) or DEFAULT_TENNIS_MACHINE_RAW_URL
    payload = load_json_from_url(url)
    return payload, url


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
    """
    Our history parser is proven on rezultati.com /utakmica/tenis/... URLs.

    Tennis-Machine may output flashscore.com URLs.
    Most URL path structure is compatible enough that replacing host and route words works
    for our generated history URL approach.
    """
    url = clean_str(url)

    if not url:
        return ""

    # Remove hash routes.
    url = url.split("#", 1)[0]

    # Convert Flashscore match URL into Rezultati-style URL if needed.
    if "flashscore.com/match/tennis/" in url:
        url = url.replace("https://www.flashscore.com/match/tennis/", "https://www.rezultati.com/utakmica/tenis/")
        url = url.replace("http://www.flashscore.com/match/tennis/", "https://www.rezultati.com/utakmica/tenis/")

    # Keep already-good rezultati URL.
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

    # source_url is often just homepage. Use only if it is a real match URL.
    value = normalize_rezultati_url(row.get("source_url"))
    if value:
        return value

    # odds_url may be available and usually contains the match path.
    value = normalize_rezultati_url(row.get("odds_url"))
    if value:
        return value

    return ""


def is_singles_match(row):
    if not isinstance(row, dict):
        return False

    match_name = clean_str(row.get("match"))
    player_1 = clean_str(row.get("player_1"))
    player_2 = clean_str(row.get("player_2"))

    # Doubles usually contain slash.
    joined = f"{match_name} {player_1} {player_2}"
    if "/" in joined:
        return False

    if not player_1 or not player_2:
        # If match field exists with separator, allow.
        if " - " not in match_name:
            return False

    return True


def row_priority(row):
    status = clean_str(row.get("status")).lower()
    tour = clean_str(row.get("tour_level")).lower()

    score = 0
    score += PRIORITY_STATUSES.get(status, 0)
    score += PRIORITY_TOUR_LEVELS.get(tour, 50)

    # Prefer rows with match URL.
    if get_match_url(row):
        score += 1000

    # Prefer today's relevant surfaces, but do not exclude unknown.
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


def main():
    payload, source = load_tennis_machine_payload()
    rows = rows_from_payload(payload)

    candidates = []
    skipped = []

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
                "match_url": row.get("match_url"),
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
        "selected": selected,
        "skipped": skipped[:300],
    }

    save_json(OUTPUT_FILE, output)
    save_json(REPORT_FILE, report)

    print("")
    print("IMPORT TENNIS MACHINE MATCHES DONE")
    print(report["counts"])
    print(f"Source: {source}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")
    print("")


if __name__ == "__main__":
    main()
