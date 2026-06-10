import csv
import hashlib
import io
import os
import re
from collections import Counter
from datetime import datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tennis_elo.config import ROOT_DIR
from tennis_elo.utils import now_iso, save_json

START_YEAR = int(os.getenv("TLE_START_YEAR", "2023"))
END_YEAR = int(os.getenv("TLE_END_YEAR", str(datetime.utcnow().year)))
TIMEOUT = int(os.getenv("TLE_HTTP_TIMEOUT", "90"))

OUTPUT_FILE = ROOT_DIR / "data" / "tle" / "processed" / "tle_sackmann_matches.json"
REPORT_FILE = ROOT_DIR / "data" / "tle" / "reports" / "tle_import_sackmann_report.json"

ATP = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
WTA = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master"

SOURCES = (
    ("men", "main", ATP + "/atp_matches_{year}.csv"),
    ("men", "qual_chall", ATP + "/atp_matches_qual_chall_{year}.csv"),
    ("men", "futures", ATP + "/atp_matches_futures_{year}.csv"),
    ("women", "main", WTA + "/wta_matches_{year}.csv"),
    ("women", "qual_itf", WTA + "/wta_matches_qual_itf_{year}.csv"),
)


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def to_int(value: Any) -> int | None:
    try:
        return int(float(clean(value))) if clean(value) else None
    except (TypeError, ValueError):
        return None


def to_float(value: Any) -> float | None:
    try:
        return float(clean(value)) if clean(value) else None
    except (TypeError, ValueError):
        return None


def iso_date(value: Any) -> str | None:
    text = clean(value)
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    return None


def surface(value: Any) -> str:
    text = clean(value).lower()
    return text if text in {"hard", "clay", "grass", "carpet"} else "unknown"


def level(family: str, raw: str, name: str) -> str:
    raw = clean(raw).upper()
    name = clean(name).lower()
    if family == "futures":
        return "itf"
    if family == "qual_itf":
        return "qualifying" if raw == "Q" else "itf"
    if family == "qual_chall":
        if raw == "C" or "challenger" in name:
            return "challenger"
        return "qualifying"
    if raw == "G":
        return "grand_slam"
    if raw in {"M", "A", "D", "F"}:
        return "atp_wta"
    if raw == "C":
        return "challenger"
    if raw in {"S", "I"}:
        return "itf"
    if raw == "Q":
        return "qualifying"
    return "atp_wta"


def player(row: dict[str, Any], side: str) -> dict[str, Any]:
    return {
        "sackmann_player_id": to_int(row.get(f"{side}_id")),
        "name": re.sub(r"\s+", " ", clean(row.get(f"{side}_name"))),
        "hand": clean(row.get(f"{side}_hand")) or None,
        "height_cm": to_int(row.get(f"{side}_ht")),
        "country": clean(row.get(f"{side}_ioc")) or None,
        "age": to_float(row.get(f"{side}_age")),
        "ranking": to_int(row.get(f"{side}_rank")),
        "ranking_points": to_int(row.get(f"{side}_rank_points")),
        "seed": clean(row.get(f"{side}_seed")) or None,
        "entry": clean(row.get(f"{side}_entry")) or None,
    }


def fetch(url: str) -> tuple[list[dict[str, Any]], str | None]:
    req = Request(url, headers={"User-Agent": "Tennis-ELO TLE importer/1.0"})
    try:
        with urlopen(req, timeout=TIMEOUT) as response:
            text = response.read().decode("utf-8-sig", errors="replace").strip()
    except HTTPError as exc:
        return [], "not_found" if exc.code == 404 else f"http_{exc.code}"
    except URLError as exc:
        return [], f"url_error_{exc.reason}"
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
    if not text:
        return [], "empty_file"
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return [], "missing_header"
    return [dict(row) for row in reader], None


def convert(row: dict[str, Any], gender: str, family: str, url: str, year: int):
    date = iso_date(row.get("tourney_date"))
    winner = player(row, "winner")
    loser = player(row, "loser")
    score = re.sub(r"\s+", " ", clean(row.get("score")))
    if not date:
        return None, "invalid_date"
    if not winner["name"] or not loser["name"]:
        return None, "missing_player_name"
    if winner["name"] == loser["name"]:
        return None, "same_player"
    if score.upper() in {"WO", "W/O"} or "W/O" in score.upper():
        return None, "walkover"

    tourney_id = clean(row.get("tourney_id"))
    round_name = clean(row.get("round"))
    raw = "|".join((gender, date, tourney_id, str(winner["sackmann_player_id"] or ""), str(loser["sackmann_player_id"] or ""), winner["name"].lower(), loser["name"].lower(), round_name))
    match_id = "tle_sackmann_" + hashlib.sha1(raw.encode()).hexdigest()[:20]
    raw_level = clean(row.get("tourney_level"))
    tourney_name = re.sub(r"\s+", " ", clean(row.get("tourney_name")))
    retired = any(x in score.upper() for x in ("RET", "ABD", "DEF"))

    return {
        "tle_match_id": match_id,
        "date": date,
        "year": year,
        "gender": gender,
        "tour_level": level(family, raw_level, tourney_name),
        "source_family": family,
        "source": "sackmann",
        "source_url": url,
        "source_row": {
            "tourney_id": tourney_id or None,
            "match_num": to_int(row.get("match_num")),
            "tourney_level_raw": raw_level or None,
        },
        "tournament": {
            "id": tourney_id or None,
            "name": tourney_name or None,
            "surface": surface(row.get("surface")),
            "draw_size": to_int(row.get("draw_size")),
            "level_raw": raw_level or None,
        },
        "round": round_name or None,
        "best_of": to_int(row.get("best_of")),
        "score": score or None,
        "retired": retired,
        "walkover": False,
        "winner": winner,
        "loser": loser,
        "ready_for_tle": not retired,
    }, None


def main() -> None:
    matches = []
    seen = set()
    files = []
    skipped = Counter()
    by_level = Counter()
    by_gender = Counter()
    by_surface = Counter()
    by_family = Counter()

    for year in range(START_YEAR, END_YEAR + 1):
        for gender, family, template in SOURCES:
            url = template.format(year=year)
            rows, error = fetch(url)
            info = {"year": year, "gender": gender, "source_family": family, "url": url, "raw_rows": len(rows), "accepted_rows": 0, "duplicate_rows": 0, "error": error}
            if error:
                files.append(info)
                print("SOURCE:", year, gender, family, "->", error)
                continue
            for row in rows:
                item, reason = convert(row, gender, family, url, year)
                if item is None:
                    skipped[reason or "unknown"] += 1
                    continue
                if item["tle_match_id"] in seen:
                    info["duplicate_rows"] += 1
                    continue
                seen.add(item["tle_match_id"])
                matches.append(item)
                info["accepted_rows"] += 1
                by_level[item["tour_level"]] += 1
                by_gender[item["gender"]] += 1
                by_surface[item["tournament"]["surface"]] += 1
                by_family[item["source_family"]] += 1
            files.append(info)
            print("SOURCE:", year, gender, family, "raw=", info["raw_rows"], "accepted=", info["accepted_rows"], "duplicates=", info["duplicate_rows"])

    matches.sort(key=lambda row: (row["date"], row["gender"], row["tour_level"], row["tle_match_id"]))
    summary = {
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "matches_total": len(matches),
        "ready_for_tle": sum(bool(row["ready_for_tle"]) for row in matches),
        "retired_matches": sum(bool(row["retired"]) for row in matches),
        "files_requested": (END_YEAR - START_YEAR + 1) * len(SOURCES),
        "files_loaded": sum(not row["error"] for row in files),
        "files_missing_or_failed": sum(bool(row["error"]) for row in files),
        "by_gender": dict(by_gender),
        "by_tour_level": dict(by_level),
        "by_surface": dict(by_surface),
        "by_source_family": dict(by_family),
        "skipped": dict(skipped),
    }
    output = {
        "generated_at": now_iso(),
        "model_family": "tle",
        "schema_version": 1,
        "source": {
            "name": "Jeff Sackmann tennis_atp / tennis_wta",
            "license": "CC BY-NC-SA 4.0; attribution required; non-commercial use only",
            "atp_repository": "https://github.com/JeffSackmann/tennis_atp",
            "wta_repository": "https://github.com/JeffSackmann/tennis_wta",
        },
        "summary": summary,
        "files": files,
        "matches": matches,
    }
    report = {"generated_at": output["generated_at"], "model_family": "tle", "summary": summary, "files": files, "first_matches": matches[:10], "last_matches": matches[-10:]}
    save_json(OUTPUT_FILE, output)
    save_json(REPORT_FILE, report)
    print("\nTLE SACKMANN IMPORT DONE")
    print("SUMMARY:", summary)
    print("Output:", OUTPUT_FILE)
    print("Report:", REPORT_FILE)


if __name__ == "__main__":
    main()
