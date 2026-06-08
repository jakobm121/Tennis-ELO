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

    leading_initials = []
    for part in parts:
        if len(part) == 1:
            leading_initials.append(part)
        else:
            break

    trailing_initials = []
    for part in reversed(parts):
        if len(part) == 1:
            trailing_initials.append(part)
        else:
            break
    trailing_initials.reverse()

    if leading_initials and len(leading_initials) < len(parts):
        return leading_initials, parts[len(leading_initials):]

    if trailing_initials and len(trailing_initials) < len(parts):
        return trailing_initials, parts[:-len(trailing_initials)]

    # Full names without initials are assumed to be "given surname...".
    return [parts[0]], parts[1:]


def surname(value: Any) -> str:
    _, surname_parts = split_name(value)
    return "".join(surname_parts)


def initials(value: Any) -> str:
    given_parts, _ = split_name(value)
    return "".join(part[:1] for part in given_parts if part)


def initial(value: Any) -> str:
    value_initials = initials(value)
    return value_initials[:1]


def name_variants(value: Any) -> set[str]:
    parts = tokens(value)
    given_parts, surname_parts = split_name(value)

    if not parts:
        return set()

    surname_text = "".join(surname_parts)
    given_text = "".join(given_parts)
    initials_text = "".join(part[:1] for part in given_parts if part)

    variants = {
        "".join(parts),
        surname_text + given_text,
        given_text + surname_text,
        surname_text + initials_text,
        initials_text + surname_text,
    }

    return {compact(item) for item in variants if item}


def normalized_full_name(value: Any) -> str:
    variants = name_variants(value)
    return max(variants, key=len) if variants else compact(value)


def unique_key(
    surname_value: str,
    initials_value: str,
    gender: str,
) -> str:
    return "|".join(
        (
            surname_value,
            initials_value,
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
        "schema_version": 2,
        "generated_at": payload.get("generated_at"),
        "players": players,
    }


def build_rating_indexes(
    ratings: list[dict[str, Any]],
) -> dict[str, Any]:
    exact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    surname_initials: dict[str, list[dict[str, Any]]] = defaultdict(list)
    surname_first_initial: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_gender: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for rating in ratings:
        name = rating_name(rating)

        if not name:
            continue

        gender = rating_gender(rating)

        for variant in name_variants(name):
            exact[variant].append(rating)

        full_initials = initials(name)
        first_initial = full_initials[:1]
        surname_value = surname(name)

        for gender_key in {gender, "unknown"}:
            surname_initials[
                unique_key(
                    surname_value,
                    full_initials,
                    gender_key,
                )
            ].append(rating)
            surname_first_initial[
                unique_key(
                    surname_value,
                    first_initial,
                    gender_key,
                )
            ].append(rating)

        by_gender[gender].append(rating)
        by_gender["unknown"].append(rating)

    return {
        "exact": dict(exact),
        "surname_initials": dict(surname_initials),
        "surname_first_initial": dict(surname_first_initial),
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
    api_variants = name_variants(api_name)
    api_surname = surname(api_name)
    api_initials = initials(api_name)

    for rating in indexes["by_gender"].get(
        gender,
        indexes["by_gender"].get("unknown", []),
    ):
        elo_name = rating_name(rating)

        if not elo_name:
            continue

        elo_variants = name_variants(elo_name)
        full_score = max(
            (
                SequenceMatcher(None, left, right).ratio()
                for left in api_variants
                for right in elo_variants
            ),
            default=0.0,
        )
        surname_score = SequenceMatcher(
            None,
            api_surname,
            surname(elo_name),
        ).ratio()
        initials_score = SequenceMatcher(
            None,
            api_initials,
            initials(elo_name),
        ).ratio() if api_initials else 0.0

        score = (
            0.60 * surname_score
            + 0.30 * full_score
            + 0.10 * initials_score
        )

        if api_initials and initials(elo_name):
            if api_initials == initials(elo_name):
                score += 0.08
            elif api_initials[:1] == initials(elo_name)[:1]:
                score += 0.03
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


def unique_candidates(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return list(
        {
            rating_name(row): row
            for row in rows
            if rating_name(row)
        }.values()
    )


def confirmed_proposal(
    rating: dict[str, Any],
    method: str,
    confidence: float,
    auto_confirm: bool = True,
) -> dict[str, Any]:
    return {
        "status": "confirmed" if auto_confirm else "manual_review",
        "method": method,
        "confidence": confidence,
        "elo_player_name": rating_name(rating),
        "elo_player_key": rating.get("player_key"),
        "candidates": [
            candidate_payload(
                rating,
                confidence,
                method,
            )
        ],
    }


def propose_mapping(
    api_player: dict[str, Any],
    indexes: dict[str, Any],
) -> dict[str, Any]:
    api_name = api_player["api_name"]
    gender = api_player["gender"]

    exact_rows = []
    for variant in name_variants(api_name):
        exact_rows.extend(indexes["exact"].get(variant, []))
    exact_candidates = unique_candidates(exact_rows)

    if len(exact_candidates) == 1:
        return confirmed_proposal(
            exact_candidates[0],
            "exact_name_variant",
            1.0,
            AUTO_CONFIRM_EXACT,
        )

    full_initials = initials(api_name)
    first_initial = full_initials[:1]
    surname_value = surname(api_name)

    full_initial_rows = []
    for gender_key in (gender, "unknown"):
        full_initial_rows.extend(
            indexes["surname_initials"].get(
                unique_key(
                    surname_value,
                    full_initials,
                    gender_key,
                ),
                [],
            )
        )
    full_initial_candidates = unique_candidates(full_initial_rows)

    if full_initials and len(full_initial_candidates) == 1:
        return confirmed_proposal(
            full_initial_candidates[0],
            "exact_compound_surname_all_initials",
            0.99,
            AUTO_CONFIRM_UNIQUE_INITIAL_SURNAME,
        )

    first_initial_rows = []
    for gender_key in (gender, "unknown"):
        first_initial_rows.extend(
            indexes["surname_first_initial"].get(
                unique_key(
                    surname_value,
                    first_initial,
                    gender_key,
                ),
                [],
            )
        )
    first_initial_candidates = unique_candidates(first_initial_rows)

    if first_initial and len(first_initial_candidates) == 1:
        return confirmed_proposal(
            first_initial_candidates[0],
            "unique_compound_surname_first_initial",
            0.96,
            AUTO_CONFIRM_UNIQUE_INITIAL_SURNAME,
        )

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
        "confidence": best["score"] if best else 0.0,
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
        "schema_version": 2,
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
    print("API TENNIS PLAYER MAPPING V2 DONE")
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
