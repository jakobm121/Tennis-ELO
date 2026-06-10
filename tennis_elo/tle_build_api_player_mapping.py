from __future__ import annotations

import argparse
import gzip
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tennis_elo.config import ROOT_DIR


DEFAULT_SACKMANN_MANIFEST = (
    ROOT_DIR
    / "data"
    / "tle"
    / "processed"
    / "sackmann"
    / "tle_sackmann_manifest.json"
)

DEFAULT_ENRICHED = (
    ROOT_DIR
    / "data"
    / "tle"
    / "source"
    / "api"
    / "tle_api_results_backfill_enriched.json"
)

DEFAULT_OUTPUT = (
    ROOT_DIR
    / "data"
    / "tle"
    / "mappings"
    / "api_player_to_sackmann.json"
)

DEFAULT_REPORT = (
    ROOT_DIR
    / "data"
    / "tle"
    / "reports"
    / "tle_build_api_player_mapping_report.json"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def ascii_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", clean_text(value))
    return "".join(char for char in text if not unicodedata.combining(char))


def normalize_name(value: Any) -> str:
    text = ascii_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def name_tokens(value: Any) -> list[str]:
    return normalize_name(value).split()


def surname_initial_key(value: Any) -> str:
    tokens = name_tokens(value)
    if len(tokens) < 2:
        return ""

    first = tokens[0]
    last = tokens[-1]
    return f"{last}|{first[:1]}"


def reversed_name(value: Any) -> str:
    tokens = name_tokens(value)
    if len(tokens) < 2:
        return normalize_name(value)

    return " ".join([tokens[-1], *tokens[:-1]])


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON object: {path}")

    return payload


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_sackmann_matches(manifest_path: Path):
    manifest = load_json(manifest_path)

    for item in manifest.get("year_files") or []:
        relative = item.get("path")
        if not relative:
            continue

        path = Path(relative)
        if not path.is_absolute():
            path = ROOT_DIR / path

        for match in read_jsonl_gz(path):
            yield match


def build_sackmann_players(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, list[str]]]:
    players: dict[str, Any] = {}

    for match in iter_sackmann_matches(manifest_path):
        gender = clean_text(match.get("gender")).lower()
        if gender not in {"men", "women"}:
            continue

        for side in ("winner", "loser"):
            player = match.get(side) or {}
            if not isinstance(player, dict):
                continue

            sackmann_id = player.get("sackmann_player_id")
            name = clean_text(player.get("name"))

            if sackmann_id in {None, ""} or not name:
                continue

            try:
                sackmann_id = int(sackmann_id)
            except (TypeError, ValueError):
                continue

            key = f"{gender}:sackmann:{sackmann_id}"

            if key not in players:
                players[key] = {
                    "player_key": key,
                    "gender": gender,
                    "sackmann_player_id": sackmann_id,
                    "name": name,
                    "names": Counter(),
                    "matches": 0,
                    "latest_date": "",
                }

            entry = players[key]
            entry["names"][name] += 1
            entry["matches"] += 1

            date_text = clean_text(match.get("date"))
            if date_text > entry["latest_date"]:
                entry["latest_date"] = date_text

    exact_index: dict[str, list[str]] = defaultdict(list)
    surname_initial_index: dict[str, list[str]] = defaultdict(list)

    for key, player in players.items():
        gender = player["gender"]

        aliases = set(player["names"].keys())
        aliases.add(player["name"])

        for alias in aliases:
            exact_index[f"{gender}|{normalize_name(alias)}"].append(key)
            exact_index[f"{gender}|{reversed_name(alias)}"].append(key)

            si = surname_initial_key(alias)
            if si:
                surname_initial_index[f"{gender}|{si}"].append(key)

    # Deduplicate index values.
    exact_index = {
        key: sorted(set(values))
        for key, values in exact_index.items()
    }
    surname_initial_index = {
        key: sorted(set(values))
        for key, values in surname_initial_index.items()
    }

    return players, exact_index, surname_initial_index


def collect_api_players(enriched_path: Path) -> dict[str, Any]:
    payload = load_json(enriched_path)
    matches = payload.get("matches") or []

    api_players: dict[str, Any] = {}

    for match in matches:
        gender = clean_text(match.get("gender")).lower()
        if gender not in {"men", "women"}:
            continue

        for side, key_field in (
            ("player_1", "first_player_key"),
            ("player_2", "second_player_key"),
        ):
            api_key = match.get(key_field)
            name = clean_text(match.get(side))

            if api_key in {None, ""} or not name:
                continue

            try:
                api_key = int(api_key)
            except (TypeError, ValueError):
                continue

            key = f"{gender}:api:{api_key}"

            if key not in api_players:
                api_players[key] = {
                    "api_player_key": api_key,
                    "gender": gender,
                    "api_name": name,
                    "names": Counter(),
                    "matches_seen": 0,
                    "levels": Counter(),
                    "latest_date": "",
                }

            entry = api_players[key]
            entry["names"][name] += 1
            entry["matches_seen"] += 1
            entry["levels"][clean_text(match.get("tour_level")).lower()] += 1

            date_text = clean_text(match.get("date"))
            if date_text > entry["latest_date"]:
                entry["latest_date"] = date_text

    return api_players


def choose_mapping(
    api_player: dict[str, Any],
    sackmann_players: dict[str, Any],
    exact_index: dict[str, list[str]],
    surname_initial_index: dict[str, list[str]],
) -> dict[str, Any]:
    gender = api_player["gender"]
    api_name = api_player["api_name"]

    exact_candidates = exact_index.get(
        f"{gender}|{normalize_name(api_name)}",
        [],
    )

    if len(exact_candidates) == 1:
        player = sackmann_players[exact_candidates[0]]
        return {
            "status": "matched",
            "method": "exact_name_gender",
            "confidence": 1.0,
            "sackmann_player_id": player["sackmann_player_id"],
            "sackmann_player_key": player["player_key"],
            "sackmann_name": player["name"],
            "candidate_count": 1,
        }

    reversed_candidates = exact_index.get(
        f"{gender}|{reversed_name(api_name)}",
        [],
    )

    if len(reversed_candidates) == 1:
        player = sackmann_players[reversed_candidates[0]]
        return {
            "status": "matched",
            "method": "reversed_exact_name_gender",
            "confidence": 0.99,
            "sackmann_player_id": player["sackmann_player_id"],
            "sackmann_player_key": player["player_key"],
            "sackmann_name": player["name"],
            "candidate_count": 1,
        }

    si = surname_initial_key(api_name)
    si_candidates = (
        surname_initial_index.get(f"{gender}|{si}", [])
        if si
        else []
    )

    # Surname + initial je uporabljen samo, Äe je kandidat enoliÄen.
    if len(si_candidates) == 1:
        player = sackmann_players[si_candidates[0]]
        return {
            "status": "matched",
            "method": "unique_surname_initial_gender",
            "confidence": 0.92,
            "sackmann_player_id": player["sackmann_player_id"],
            "sackmann_player_key": player["player_key"],
            "sackmann_name": player["name"],
            "candidate_count": 1,
        }

    candidates = []
    for candidate_key in sorted(set(exact_candidates + reversed_candidates + si_candidates)):
        player = sackmann_players[candidate_key]
        candidates.append(
            {
                "sackmann_player_id": player["sackmann_player_id"],
                "sackmann_player_key": player["player_key"],
                "sackmann_name": player["name"],
                "matches": player["matches"],
                "latest_date": player["latest_date"],
            }
        )

    return {
        "status": "unmatched" if not candidates else "review",
        "method": "no_safe_match" if not candidates else "ambiguous",
        "confidence": 0.0,
        "sackmann_player_id": None,
        "sackmann_player_key": None,
        "sackmann_name": None,
        "candidate_count": len(candidates),
        "candidates": candidates[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Zgradi API player_key -> Sackmann player_id mapping za TLE."
        )
    )

    parser.add_argument(
        "--sackmann-manifest",
        default=str(DEFAULT_SACKMANN_MANIFEST),
    )

    parser.add_argument(
        "--enriched",
        default=str(DEFAULT_ENRICHED),
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
    )

    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
    )

    args = parser.parse_args()

    sackmann_players, exact_index, si_index = build_sackmann_players(
        Path(args.sackmann_manifest)
    )
    api_players = collect_api_players(Path(args.enriched))

    mappings = {}
    decisions = []
    counters = Counter()
    matches_by_status = Counter()

    for api_key, api_player in sorted(api_players.items()):
        decision = choose_mapping(
            api_player,
            sackmann_players,
            exact_index,
            si_index,
        )

        status = decision["status"]
        counters[status] += 1
        matches_by_status[status] += api_player["matches_seen"]

        mapping_key = str(api_player["api_player_key"])

        mappings[mapping_key] = {
            "api_player_key": api_player["api_player_key"],
            "api_name": api_player["api_name"],
            "gender": api_player["gender"],
            "matches_seen": api_player["matches_seen"],
            "levels": dict(api_player["levels"]),
            "status": status,
            "method": decision["method"],
            "confidence": decision["confidence"],
            "sackmann_player_id": decision["sackmann_player_id"],
            "sackmann_player_key": decision["sackmann_player_key"],
            "sackmann_name": decision["sackmann_name"],
            "needs_review": status != "matched",
        }

        decisions.append(
            {
                "api_mapping_key": mapping_key,
                **mappings[mapping_key],
                "candidates": decision.get("candidates", []),
            }
        )

    unresolved = [
        row
        for row in decisions
        if row["status"] != "matched"
    ]

    output_payload = {
        "schema_version": 1,
        "generated_at": now_iso(),
        "source": {
            "sackmann_manifest": str(args.sackmann_manifest),
            "enriched_api_results": str(args.enriched),
        },
        "summary": {
            "api_players": len(api_players),
            "sackmann_players": len(sackmann_players),
            "mapping_statuses": dict(sorted(counters.items())),
            "matches_by_status": dict(sorted(matches_by_status.items())),
            "matched_players": counters["matched"],
            "unresolved_players": len(unresolved),
        },
        "players": dict(
            sorted(
                mappings.items(),
                key=lambda item: (
                    item[1]["needs_review"],
                    -item[1]["matches_seen"],
                    item[1]["api_name"],
                ),
            )
        ),
    }

    report_payload = {
        "generated_at": output_payload["generated_at"],
        "summary": output_payload["summary"],
        "unresolved": sorted(
            unresolved,
            key=lambda row: (-row["matches_seen"], row["api_name"]),
        ),
        "all_decisions": decisions,
    }

    save_json(Path(args.output), output_payload)
    save_json(Path(args.report), report_payload)

    print("TLE API PLAYER MAPPING DONE")
    print(json.dumps(output_payload["summary"], indent=2, ensure_ascii=False))
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
