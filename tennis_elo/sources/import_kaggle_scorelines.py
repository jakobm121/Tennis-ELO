import csv
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from tennis_elo.config import ROOT_DIR
from tennis_elo.utils import clean_str, load_json, now_iso, save_json


INPUT_DIR = Path(
    os.getenv(
        "KAGGLE_TENNIS_INPUT_DIR",
        str(ROOT_DIR / "data" / "external" / "kaggle_tennis"),
    )
)
ARCHIVE_FILE = ROOT_DIR / "data" / "totals" / "scorelines_archive.json"
IMPORT_FILE = ROOT_DIR / "data" / "totals" / "kaggle_scorelines_import.json"
REPORT_FILE = ROOT_DIR / "data" / "reports" / "import_kaggle_scorelines_report.json"

LOOKBACK_DAYS = int(os.getenv("TOTALS_LOOKBACK_DAYS", "365"))
REFERENCE_DATE = clean_str(os.getenv("TOTALS_REFERENCE_DATE"))

INVALID_MARKERS = (
    "RET", "W/O", " WALKOVER", " DEF", "ABD", "ABN",
    "CANCELLED", "CANCELED", "SUSPENDED",
)

ALIASES = {
    "date": ("date", "match_date", "tourney_date"),
    "tournament": ("tournament", "tourney_name", "event", "tourney"),
    "surface": ("surface", "court"),
    "round": ("round", "rnd"),
    "best_of": ("best of", "best_of", "bestof"),
    "winner": ("winner", "winner_name", "w_name"),
    "loser": ("loser", "loser_name", "l_name"),
    "score": ("score", "match_score", "result"),
    "comment": ("comment", "status"),
    "winner_rank": ("wrank", "winner_rank", "w_rank"),
    "loser_rank": ("lrank", "loser_rank", "l_rank"),
    "winner_odds": ("b365w", "psw", "avgw", "maxw", "winner_odds"),
    "loser_odds": ("b365l", "psl", "avgl", "maxl", "loser_odds"),
    "location": ("court", "location"),
}


def norm_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean_str(value).lower())


def make_mapping(fieldnames: list[str]) -> dict[str, str]:
    available = {norm_header(name): name for name in fieldnames if name}
    result: dict[str, str] = {}

    for logical, aliases in ALIASES.items():
        for alias in aliases:
            actual = available.get(norm_header(alias))
            if actual:
                result[logical] = actual
                break

    return result


def value(row: dict[str, Any], mapping: dict[str, str], key: str) -> str:
    column = mapping.get(key)
    return clean_str(row.get(column)) if column else ""


def as_int(raw: str) -> int | None:
    try:
        return int(float(clean_str(raw)))
    except (TypeError, ValueError):
        return None


def as_float(raw: str) -> float | None:
    try:
        return float(clean_str(raw).replace(",", "."))
    except (TypeError, ValueError):
        return None


def parse_date(raw: str) -> date | None:
    raw = clean_str(raw)

    for fmt in (
        "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d.%m.%Y",
        "%m/%d/%Y", "%Y%m%d", "%d-%m-%Y",
    ):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            pass

    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def current_reference_date() -> date:
    return parse_date(REFERENCE_DATE) or date.today()


def normalize_surface(raw: str) -> str:
    lower = clean_str(raw).lower()

    if "clay" in lower:
        return "clay"
    if "grass" in lower:
        return "grass"
    if "hard" in lower:
        return "hard"
    if "carpet" in lower:
        return "carpet"

    return lower


def gender_from_path(path: Path) -> str:
    lower = str(path).lower()
    return "women" if "wta" in lower or "women" in lower else "men"


def grand_slam(tournament: str) -> bool:
    lower = tournament.lower()
    return any(
        name in lower
        for name in (
            "australian open", "french open", "roland garros",
            "wimbledon", "us open", "u.s. open",
        )
    )


def tour_level(tournament: str, gender: str) -> str:
    lower = tournament.lower()

    if grand_slam(tournament):
        return "grand_slam"
    if "challenger" in lower:
        return "challenger"
    if "itf" in lower:
        return "itf"

    return "wta" if gender == "women" else "atp"


def parse_set(token: str) -> dict[str, Any] | None:
    token = clean_str(token).strip(",;")

    if not token or token.startswith("["):
        return None

    match = re.fullmatch(
        r"(\d{1,2})-(\d{1,2})(?:\((\d{1,2})(?:-(\d{1,2}))?\))?",
        token,
    )
    if not match:
        return None

    p1 = int(match.group(1))
    p2 = int(match.group(2))

    if max(p1, p2) > 20 or p1 == p2:
        return None

    is_tb = (p1, p2) in {(7, 6), (6, 7)}
    result: dict[str, Any] = {
        "p1_games": p1,
        "p2_games": p2,
        "tiebreak": is_tb,
    }

    tb_a = int(match.group(3)) if match.group(3) else None
    tb_b = int(match.group(4)) if match.group(4) else None

    if is_tb and tb_a is not None:
        if tb_b is not None:
            result["tiebreak_p1"] = tb_a
            result["tiebreak_p2"] = tb_b
        elif p1 > p2:
            result["tiebreak_p2"] = tb_a
        else:
            result["tiebreak_p1"] = tb_a

    return result


def parse_score(raw: str) -> list[dict[str, Any]] | None:
    raw = clean_str(raw).replace("â", "-").replace("â", "-")
    raw = re.sub(r"\s+", " ", raw)

    sets: list[dict[str, Any]] = []

    for token in raw.split():
        # Ignore final match-tiebreak notation; it is not a normal set total.
        if re.fullmatch(r"\[\d{1,2}-\d{1,2}\]", token):
            continue

        parsed = parse_set(token)
        if parsed is None:
            return None

        parsed["set_number"] = len(sets) + 1
        sets.append(parsed)

    return sets or None


def derived_fields(sets: list[dict[str, Any]]) -> dict[str, Any]:
    p1_sets = sum(1 for item in sets if item["p1_games"] > item["p2_games"])
    p2_sets = sum(1 for item in sets if item["p2_games"] > item["p1_games"])
    p1_games = sum(item["p1_games"] for item in sets)
    p2_games = sum(item["p2_games"] for item in sets)
    first = sets[0]
    tb_count = sum(1 for item in sets if item.get("tiebreak"))

    return {
        "sets_1": p1_sets,
        "sets_2": p2_sets,
        "final_score": f"{p1_sets}-{p2_sets}",
        "first_set_score": f"{first['p1_games']}-{first['p2_games']}",
        "first_set_games": first["p1_games"] + first["p2_games"],
        "first_set_tiebreak": bool(first.get("tiebreak")),
        "total_games": p1_games + p2_games,
        "p1_total_games_won": p1_games,
        "p2_total_games_won": p2_games,
        "game_margin_p1": p1_games - p2_games,
        "total_sets": len(sets),
        "straight_sets": p1_sets == 0 or p2_sets == 0,
        "deciding_set": len(sets) in {3, 5},
        "had_tiebreak": tb_count > 0,
        "tiebreak_count": tb_count,
    }


def invalid_score(score: str, comment: str) -> bool:
    combined = f" {score} {comment} ".upper()
    return not score or any(marker in combined for marker in INVALID_MARKERS)


def row_to_match(
    row: dict[str, Any],
    mapping: dict[str, str],
    source_file: Path,
    cutoff: date,
) -> tuple[dict[str, Any] | None, str]:
    played = parse_date(value(row, mapping, "date"))
    if not played:
        return None, "invalid_date"
    if played < cutoff:
        return None, "outside_window"

    winner = value(row, mapping, "winner")
    loser = value(row, mapping, "loser")
    score = value(row, mapping, "score")
    comment = value(row, mapping, "comment")

    if not winner or not loser:
        return None, "missing_players"
    if invalid_score(score, comment):
        return None, "invalid_or_incomplete"

    sets = parse_score(score)
    if not sets:
        return None, "score_parse_failed"

    derived = derived_fields(sets)

    # Kaggle files normally provide score from the winner's perspective.
    if derived["sets_1"] <= derived["sets_2"]:
        return None, "winner_score_mismatch"

    gender = gender_from_path(source_file)
    tournament = value(row, mapping, "tournament")
    is_gs = grand_slam(tournament)

    best_of = as_int(value(row, mapping, "best_of"))
    if best_of not in {3, 5}:
        best_of = 5 if is_gs and gender == "men" else 3

    location_text = " ".join(
        [value(row, mapping, "location"), tournament]
    ).lower()

    indoor: bool | None = None
    if "indoor" in location_text:
        indoor = True
    elif "outdoor" in location_text:
        indoor = False

    match: dict[str, Any] = {
        "match_id": "",
        "match_url": "",
        "source": "kaggle_tennis",
        "source_file": source_file.name,
        "date": played.isoformat(),
        "tournament": tournament,
        "tour_level": tour_level(tournament, gender),
        "round": value(row, mapping, "round"),
        "gender": gender,
        "surface": normalize_surface(value(row, mapping, "surface")),
        "indoor": indoor,
        "best_of": best_of,
        "is_grand_slam": is_gs,
        "player_1": winner,
        "player_2": loser,
        "match": f"{winner} - {loser}",
        "winner": winner,
        "loser": loser,
        "set_scores": sets,
        "completed": True,
        "retired": False,
        "walkover": False,
        "raw_score": score,
        "winner_rank": as_int(value(row, mapping, "winner_rank")),
        "loser_rank": as_int(value(row, mapping, "loser_rank")),
        "bookmaker": {
            "winner_odds": as_float(value(row, mapping, "winner_odds")),
            "loser_odds": as_float(value(row, mapping, "loser_odds")),
        },
    }
    match.update(derived)
    return match, "ok"


def compact_name(raw: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean_str(raw).lower())


def archive_key(row: dict[str, Any]) -> str:
    players = sorted(
        [compact_name(row.get("player_1")), compact_name(row.get("player_2"))]
    )

    return "|".join(
        [
            clean_str(row.get("date")),
            players[0],
            players[1],
            compact_name(row.get("tournament")),
            clean_str(row.get("first_set_score")),
            clean_str(row.get("final_score")),
        ]
    )


def merge_rows(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    merged = dict(old)

    for key, item in new.items():
        if merged.get(key) in [None, "", [], {}] and item not in [None, "", [], {}]:
            merged[key] = item

    return merged


def load_archive() -> list[dict[str, Any]]:
    payload = load_json(ARCHIVE_FILE, {})

    if isinstance(payload, dict):
        rows = payload.get("matches") or payload.get("scorelines") or []
        return rows if isinstance(rows, list) else []

    return payload if isinstance(payload, list) else []


def merge_archive(
    existing: list[dict[str, Any]],
    imported: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for row in existing + imported:
        key = archive_key(row)
        if not key:
            continue

        if key not in by_key:
            by_key[key] = row
            order.append(key)
        else:
            by_key[key] = merge_rows(by_key[key], row)

    return [by_key[key] for key in order]


def import_csv(
    path: Path,
    cutoff: date,
) -> tuple[list[dict[str, Any]], dict[str, int], list[str]]:
    imported: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        mapping = make_mapping(headers)

        missing = {"date", "winner", "loser", "score"} - set(mapping)
        if missing:
            counts["missing_required_columns"] = 1
            return imported, counts, headers

        for row in reader:
            parsed, status = row_to_match(row, mapping, path, cutoff)
            counts[status] = counts.get(status, 0) + 1
            if parsed:
                imported.append(parsed)

    return imported, counts, headers


def main() -> None:
    reference = current_reference_date()
    cutoff = reference - timedelta(days=LOOKBACK_DAYS)
    csv_files = sorted(INPUT_DIR.rglob("*.csv"))

    all_imported: list[dict[str, Any]] = []
    parse_counts: dict[str, int] = {}
    file_reports: list[dict[str, Any]] = []

    for csv_file in csv_files:
        imported, counts, headers = import_csv(csv_file, cutoff)
        all_imported.extend(imported)

        for key, count in counts.items():
            parse_counts[key] = parse_counts.get(key, 0) + count

        file_reports.append(
            {
                "file": str(csv_file),
                "gender": gender_from_path(csv_file),
                "headers": headers,
                "counts": counts,
                "imported": len(imported),
            }
        )

    unique: dict[str, dict[str, Any]] = {}
    for row in all_imported:
        key = archive_key(row)
        unique[key] = merge_rows(unique[key], row) if key in unique else row

    imported_unique = list(unique.values())
    archive_before = load_archive()
    archive_after = merge_archive(archive_before, imported_unique)

    import_payload = {
        "generated_at": now_iso(),
        "reference_date": reference.isoformat(),
        "cutoff_date": cutoff.isoformat(),
        "lookback_days": LOOKBACK_DAYS,
        "counts": {
            "csv_files": len(csv_files),
            "imported_before_dedupe": len(all_imported),
            "imported_unique": len(imported_unique),
        },
        "matches": imported_unique,
    }

    archive_payload = {
        "generated_at": now_iso(),
        "source": "kaggle_plus_rezultati",
        "counts": {
            "archive_before": len(archive_before),
            "kaggle_imported_unique": len(imported_unique),
            "archive_after": len(archive_after),
            "added": max(0, len(archive_after) - len(archive_before)),
        },
        "matches": archive_after,
    }

    coverage = {
        "men": sum(row.get("gender") == "men" for row in archive_after),
        "women": sum(row.get("gender") == "women" for row in archive_after),
        "clay": sum(row.get("surface") == "clay" for row in archive_after),
        "hard": sum(row.get("surface") == "hard" for row in archive_after),
        "grass": sum(row.get("surface") == "grass" for row in archive_after),
        "with_odds": sum(
            bool((row.get("bookmaker") or {}).get("winner_odds"))
            and bool((row.get("bookmaker") or {}).get("loser_odds"))
            for row in archive_after
        ),
    }

    report = {
        "generated_at": now_iso(),
        "input_dir": str(INPUT_DIR),
        "reference_date": reference.isoformat(),
        "cutoff_date": cutoff.isoformat(),
        "lookback_days": LOOKBACK_DAYS,
        "counts": archive_payload["counts"],
        "coverage": coverage,
        "parse_counts": parse_counts,
        "files": file_reports,
    }

    save_json(IMPORT_FILE, import_payload)
    save_json(ARCHIVE_FILE, archive_payload)
    save_json(REPORT_FILE, report)

    print("")
    print("IMPORT KAGGLE SCORELINES DONE")
    print(report["counts"])
    print("Coverage:", coverage)
    print("Parse:", parse_counts)
    print(f"Input:   {INPUT_DIR}")
    print(f"Import:  {IMPORT_FILE}")
    print(f"Archive: {ARCHIVE_FILE}")
    print(f"Report:  {REPORT_FILE}")
    print("")


if __name__ == "__main__":
    main()
