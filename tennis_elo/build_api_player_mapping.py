import json
import os
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any
from urllib.request import Request, urlopen

from tennis_elo.config import ROOT_DIR
from tennis_elo.utils import clean_str, load_json, now_iso, save_json


ODDS_URL = os.getenv(
    "TENNIS_ODDS_URL",
    "https://raw.githubusercontent.com/jakobm121/Ai/refs/heads/main/data/tennis_odds_today.json",
)
RATINGS_FILE = (
    ROOT_DIR
    / "data"
    / "ratings"
    / "player_ratings_latest.json"
)
MAPPING_FILE = (
    ROOT_DIR
    / "data"
    / "mappings"
    / "api_tennis_player_map.json"
)
REPORT_FILE = (
    ROOT_DIR
    / "data"
    / "reports"
    / "api_tennis_player_mapping_report.json"
)

HTTP_TIMEOUT = int(
    os.getenv("TENNIS_ODDS_HTTP_TIMEOUT", "45")
)
FUZZY_REVIEW_THRESHOLD = float(
    os.getenv("API_PLAYER_MAP_FUZZY_REVIEW_THRESHOLD", "0.82")
)
AUTO_CONFIRM_EXACT = (
    os.getenv("API_PLAYER_MAP_AUTO_CONFIRM_EXACT", "1") == "1"
)
AUTO_CONFIRM_UNIQUE_INITIAL_SURNAME = (
    os.getenv(
        "API_PLAYER_MAP_AUTO_CONFIRM_UNIQUE_INITIAL_SURNAME",
        "1",
    ) == "1"
)


def fetch_json(url: str) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "User-Agent": "Tennis-ELO API player mapping/1.0",
            "Accept": "application/json",
        },
    )

    with urlopen(request, timeout=HTTP_TIMEOUT) as response:
        payload = json.load(response)

    return payload if isinstance(payload, dict) else {}


def compact(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        clean_str(value).lower(),
    )


def tokens(value: Any) -> list[str]:
    return re.findall(
        r"[a-z0-9]+",
        clean_str(value).lower(),
    )


def split_name(value: Any) -> tuple[list[str], list[str]]:
    parts = tokens(value)

    if not parts:
        return [], []

    if len(parts) == 1:
        return [], parts

    first_is_initial = len(parts[0]) == 1
    last_is_initial = len(parts[-1]) == 1

    if first_is_initial and not last_is_initial:
        return [parts[0]], parts[1:]

    if last_is_initial and not first_is_initial:
        return [parts[-1]], parts[:-1]

    return [parts[0]], parts[1:]


def surname(value: Any) -> str:
    _, surname_parts = split_name(value)
    return "".join(surname_parts)


def initial(value: Any) -> str:
    given_parts, _ = split_name(value)
    return given_parts[0][:1] if given_parts else ""


def normalized_full_name(value: Any) -> str:
    given_parts, surname_parts = split_name(value)

    if not surname_parts:
        return compact(value)

    return "".join(surname_parts + given_parts)


def unique_key(
    surname_value: str,
    initial_value: str,
    gender: str,
) -> str:
    return "|".join(
        (
            surname_value,
            initial_value,
            gender or "unknown",
        )
    )


def rating_name(rating: dict[str, Any]) -> str:
    return clean_str(
        rating.get("player_name")
        or rating.get("name")
        or rating.get("player_key")
    )


def rating_gender(rating: dict[str, Any]) -> str:
    value = clean_str(
        rating.get("gender")
        or rating.get("sex")
        or rating.get("tour_gender")
    ).lower()

    if value in {"m", "male", "men", "atp"}:
        return "men"

    if value in {"f", "female", "women", "wta"}:
        return "women"

    return "unknown"


def load_mapping() -> dict[str, Any]:
    payload = load_json(MAPPING_FILE, {})

    if not isinstance(payload, dict):
        payload = {}

    players = payload.get("players")

    if not isinstance(players, dict):
        players = {}

    return {
        "schema_version": 1,
        "generated_at": payload.get("generated_at"),
        "players": players,
    }


def build_rating_indexes(
    ratings: list[dict[str, Any]],
) -> dict[str, Any]:
    exact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    initial_surname: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_gender: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for rating in ratings:
        name = rating_name(rating)

        if not name:
            continue

        gender = rating_gender(rating)
        exact[normalized_full_name(name)].append(rating)
        initial_surname[
            unique_key(
                surname(name),
                initial(name),
                gender,
            )
        ].append(rating)
        initial_surname[
            unique_key(
                surname(name),
                initial(name),
                "unknown",
            )
        ].append(rating)
        by_gender[gender].append(rating)
        by_gender["unknown"].append(rating)

    return {
        "exact": dict(exact),
        "initial_surname": dict(initial_surname),
        "by_gender": dict(by_gender),
    }


def api_players_from_odds(
    odds_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    players: dict[str, dict[str, Any]] = {}

    for match in odds_payload.get("matches", []):
        if not isinstance(match, dict):
            continue

        gender = clean_str(match.get("gender")).lower() or "unknown"

        for side in ("player_1", "player_2"):
            api_name = clean_str(match.get(side))
            api_key = match.get(
                "first_player_key"
                if side == "player_1"
                else "second_player_key"
            )

            if api_key in (None, ""):
                continue

            key_text = str(api_key)

            players[key_text] = {
                "api_player_key": api_key,
                "api_name": api_name,
                "gender": gender,
                "source_event_key": match.get("event_key"),
                "tournament": match.get("tournament"),
                "tournament_key": match.get("tournament_key"),
            }

    return list(players.values())


def candidate_payload(
    rating: dict[str, Any],
    score: float,
    method: str,
) -> dict[str, Any]:
    return {
        "elo_player_name": rating_name(rating),
        "elo_player_key": rating.get("player_key"),
        "matches_total": int(
            rating.get("matches_total") or 0
        ),
        "gender": rating_gender(rating),
        "score": round(score, 4),
        "method": method,
    }


def fuzzy_candidates(
    api_name: str,
    gender: str,
    indexes: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates = []

    for rating in indexes["by_gender"].get(
        gender,
        indexes["by_gender"].get("unknown", []),
    ):
        elo_name = rating_name(rating)

        if not elo_name:
            continue

        surname_score = SequenceMatcher(
            None,
            surname(api_name),
            surname(elo_name),
        ).ratio()
        full_score = SequenceMatcher(
            None,
            normalized_full_name(api_name),
            normalized_full_name(elo_name),
        ).ratio()

        score = (
            0.65 * surname_score
            + 0.35 * full_score
        )

        if initial(api_name) and initial(elo_name):
            if initial(api_name) == initial(elo_name):
                score += 0.08
            else:
                score -= 0.08

        candidates.append(
            candidate_payload(
                rating,
                min(1.0, max(0.0, score)),
                "fuzzy",
            )
        )

    candidates.sort(
        key=lambda item: (
            item["score"],
            item["matches_total"],
        ),
        reverse=True,
    )

    return candidates[:5]


def propose_mapping(
    api_player: dict[str, Any],
    indexes: dict[str, Any],
) -> dict[str, Any]:
    api_name = api_player["api_name"]
    gender = api_player["gender"]

    exact_candidates = indexes["exact"].get(
        normalized_full_name(api_name),
        [],
    )

    if len(exact_candidates) == 1:
        rating = exact_candidates[0]

        return {
            "status": (
                "confirmed"
                if AUTO_CONFIRM_EXACT
                else "manual_review"
            ),
            "method": "exact_normalized_name",
            "confidence": 1.0,
            "elo_player_name": rating_name(rating),
            "elo_player_key": rating.get("player_key"),
            "candidates": [
                candidate_payload(
                    rating,
                    1.0,
                    "exact_normalized_name",
                )
            ],
        }

    initial_candidates = indexes[
        "initial_surname"
    ].get(
        unique_key(
            surname(api_name),
            initial(api_name),
            gender,
        ),
        [],
    )

    if not initial_candidates:
        initial_candidates = indexes[
            "initial_surname"
        ].get(
            unique_key(
                surname(api_name),
                initial(api_name),
                "unknown",
            ),
            [],
        )

    unique_initial_candidates = {
        rating_name(candidate): candidate
        for candidate in initial_candidates
    }

    if len(unique_initial_candidates) == 1:
        rating = next(
            iter(unique_initial_candidates.values())
        )

        return {
            "status": (
                "confirmed"
                if AUTO_CONFIRM_UNIQUE_INITIAL_SURNAME
                else "manual_review"
            ),
            "method": "unique_initial_surname",
            "confidence": 0.96,
            "elo_player_name": rating_name(rating),
            "elo_player_key": rating.get("player_key"),
            "candidates": [
                candidate_payload(
                    rating,
                    0.96,
                    "unique_initial_surname",
                )
            ],
        }

    candidates = fuzzy_candidates(
        api_name,
        gender,
        indexes,
    )
    best = candidates[0] if candidates else None
    second = candidates[1] if len(candidates) > 1 else None

    if (
        best
        and best["score"] >= FUZZY_REVIEW_THRESHOLD
        and (
            second is None
            or best["score"] - second["score"] >= 0.08
        )
    ):
        return {
            "status": "manual_review",
            "method": "fuzzy_candidate",
            "confidence": best["score"],
            "elo_player_name": best["elo_player_name"],
            "elo_player_key": best["elo_player_key"],
            "candidates": candidates,
        }

    return {
        "status": "unmatched",
        "method": "no_safe_match",
        "confidence": (
            best["score"]
            if best
            else 0.0
        ),
        "elo_player_name": None,
        "elo_player_key": None,
        "candidates": candidates,
    }


def main() -> None:
    odds_payload = fetch_json(ODDS_URL)
    ratings_payload = load_json(
        RATINGS_FILE,
        {},
    )
    mapping_payload = load_mapping()

    ratings = ratings_payload.get("ratings", [])

    if not isinstance(ratings, list):
        ratings = []

    api_players = api_players_from_odds(
        odds_payload
    )
    indexes = build_rating_indexes(ratings)
    mappings = mapping_payload["players"]

    counters: dict[str, int] = defaultdict(int)
    changed = 0

    for api_player in api_players:
        api_key = str(
            api_player["api_player_key"]
        )
        existing = mappings.get(api_key)

        if (
            isinstance(existing, dict)
            and existing.get("status") == "confirmed"
        ):
            counters["existing_confirmed"] += 1
            continue

        proposal = propose_mapping(
            api_player,
            indexes,
        )

        mappings[api_key] = {
            "api_player_key": api_player[
                "api_player_key"
            ],
            "api_name": api_player["api_name"],
            "gender": api_player["gender"],
            "status": proposal["status"],
            "method": proposal["method"],
            "confidence": round(
                float(proposal["confidence"]),
                4,
            ),
            "elo_player_name": proposal[
                "elo_player_name"
            ],
            "elo_player_key": proposal[
                "elo_player_key"
            ],
            "candidates": proposal["candidates"],
            "last_seen_event_key": api_player[
                "source_event_key"
            ],
            "last_seen_tournament": api_player[
                "tournament"
            ],
            "last_seen_tournament_key": api_player[
                "tournament_key"
            ],
            "updated_at": now_iso(),
        }
        counters[proposal["status"]] += 1
        changed += 1

    output = {
        "schema_version": 1,
        "generated_at": now_iso(),
        "source_odds_url": ODDS_URL,
        "odds_generated_at": odds_payload.get(
            "generated_at"
        ),
        "ratings_generated_at": ratings_payload.get(
            "generated_at"
        ),
        "players": mappings,
    }

    active_rows = [
        mappings[str(player["api_player_key"])]
        for player in api_players
        if str(player["api_player_key"]) in mappings
    ]

    report = {
        "generated_at": now_iso(),
        "summary": {
            "api_players_today": len(api_players),
            "ratings_players": len(ratings),
            "mapping_records_total": len(mappings),
            "mapping_records_changed": changed,
            **dict(counters),
        },
        "status_counts_today": dict(
            Counter(
                row.get("status", "unknown")
                for row in active_rows
            )
        ),
        "manual_review": [
            row
            for row in active_rows
            if row.get("status") == "manual_review"
        ],
        "unmatched": [
            row
            for row in active_rows
            if row.get("status") == "unmatched"
        ],
    }

    save_json(MAPPING_FILE, output)
    save_json(REPORT_FILE, report)

    print("")
    print("API TENNIS PLAYER MAPPING DONE")
    print("SUMMARY:", report["summary"])
    print(
        "STATUS COUNTS TODAY:",
        report["status_counts_today"],
    )

    for row in report["manual_review"][:30]:
        print(
            "REVIEW:",
            row.get("api_player_key"),
            row.get("api_name"),
            "->",
            row.get("elo_player_name"),
            "confidence=",
            row.get("confidence"),
        )

    for row in report["unmatched"][:30]:
        print(
            "UNMATCHED:",
            row.get("api_player_key"),
            row.get("api_name"),
        )

    print(f"Mapping: {MAPPING_FILE}")
    print(f"Report:  {REPORT_FILE}")
    print("")


if __name__ == "__main__":
    main()
