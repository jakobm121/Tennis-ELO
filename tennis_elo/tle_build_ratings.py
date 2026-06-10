import gzip
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from tennis_elo.config import ROOT_DIR
from tennis_elo.utils import now_iso, save_json


MANIFEST_FILE = (
    ROOT_DIR
    / "data"
    / "tle"
    / "processed"
    / "sackmann"
    / "tle_sackmann_manifest.json"
)
RATINGS_FILE = (
    ROOT_DIR
    / "data"
    / "tle"
    / "ratings"
    / "tle_player_ratings.json.gz"
)
RATINGS_MANIFEST_FILE = (
    ROOT_DIR
    / "data"
    / "tle"
    / "ratings"
    / "tle_player_ratings_manifest.json"
)
REPORT_FILE = (
    ROOT_DIR
    / "data"
    / "tle"
    / "reports"
    / "tle_build_ratings_report.json"
)

DEFAULT_ELO = float(os.getenv("TLE_DEFAULT_ELO", "1500"))
GLOBAL_K = float(os.getenv("TLE_GLOBAL_K", "24"))
GLOBAL_SURFACE_K = float(
    os.getenv("TLE_GLOBAL_SURFACE_K", "20")
)
LEVEL_K = float(os.getenv("TLE_LEVEL_K", "24"))
LEVEL_SURFACE_K = float(
    os.getenv("TLE_LEVEL_SURFACE_K", "20")
)

VALID_LEVELS = {
    "grand_slam",
    "atp_wta",
    "challenger",
    "qualifying",
    "itf",
}
VALID_SURFACES = {
    "hard",
    "clay",
    "grass",
    "carpet",
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_name(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        clean(value),
    )


def normalize_key(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        clean(value).lower(),
    ).strip()


def expected_score(
    rating_a: float,
    rating_b: float,
) -> float:
    return 1.0 / (
        1.0
        + math.pow(
            10.0,
            (rating_b - rating_a) / 400.0,
        )
    )


def update_pair(
    winner_rating: float,
    loser_rating: float,
    k_factor: float,
) -> tuple[float, float]:
    winner_expected = expected_score(
        winner_rating,
        loser_rating,
    )
    change = k_factor * (1.0 - winner_expected)

    return (
        winner_rating + change,
        loser_rating - change,
    )


def player_identity(
    player: dict[str, Any],
    gender: str,
) -> tuple[str, str]:
    player_id = player.get("sackmann_player_id")
    name = normalize_name(player.get("name"))

    if player_id not in (None, ""):
        return (
            f"{gender}:sackmann:{int(player_id)}",
            name,
        )

    return (
        f"{gender}:name:{normalize_key(name)}",
        name,
    )


def new_surface_state() -> dict[str, Any]:
    return {
        "elo": DEFAULT_ELO,
        "matches": 0,
        "wins": 0,
    }


def new_level_state() -> dict[str, Any]:
    return {
        "overall_elo": DEFAULT_ELO,
        "matches": 0,
        "wins": 0,
        "surfaces": {},
    }


def new_player_state(
    player_key: str,
    display_name: str,
    gender: str,
    sackmann_player_id: int | None,
) -> dict[str, Any]:
    return {
        "player_key": player_key,
        "display_name": display_name,
        "gender": gender,
        "sackmann_player_id": sackmann_player_id,
        "global": {
            "overall_elo": DEFAULT_ELO,
            "matches": 0,
            "wins": 0,
            "surfaces": {},
        },
        "levels": {},
        "first_match_date": None,
        "last_match_date": None,
        "countries": Counter(),
    }


def ensure_surface(
    container: dict[str, Any],
    surface: str,
) -> dict[str, Any]:
    surfaces = container["surfaces"]

    if surface not in surfaces:
        surfaces[surface] = new_surface_state()

    return surfaces[surface]


def ensure_level(
    player: dict[str, Any],
    level: str,
) -> dict[str, Any]:
    levels = player["levels"]

    if level not in levels:
        levels[level] = new_level_state()

    return levels[level]


def read_manifest() -> dict[str, Any]:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Missing TLE Sackmann manifest: {MANIFEST_FILE}"
        )

    return json.loads(
        MANIFEST_FILE.read_text(
            encoding="utf-8"
        )
    )


def iter_matches(
    manifest: dict[str, Any],
):
    year_files = manifest.get("year_files", [])

    if not isinstance(year_files, list):
        raise ValueError(
            "Invalid year_files in TLE Sackmann manifest"
        )

    for item in sorted(
        year_files,
        key=lambda row: int(row.get("year", 0)),
    ):
        path = Path(item["path"])

        if not path.is_absolute():
            path = ROOT_DIR / path

        if not path.exists():
            raise FileNotFoundError(
                f"Missing TLE year file: {path}"
            )

        rows = []

        with gzip.open(
            path,
            "rt",
            encoding="utf-8",
        ) as handle:
            for line in handle:
                line = line.strip()

                if not line:
                    continue

                rows.append(json.loads(line))

        rows.sort(
            key=lambda row: (
                clean(row.get("date")),
                clean(row.get("tle_match_id")),
            )
        )

        for row in rows:
            yield row


def update_date_range(
    player: dict[str, Any],
    match_date: str,
) -> None:
    if (
        not player["first_match_date"]
        or match_date < player["first_match_date"]
    ):
        player["first_match_date"] = match_date

    if (
        not player["last_match_date"]
        or match_date > player["last_match_date"]
    ):
        player["last_match_date"] = match_date


def update_rating_layer(
    winner_container: dict[str, Any],
    loser_container: dict[str, Any],
    rating_field: str,
    k_factor: float,
) -> None:
    winner_rating = float(
        winner_container[rating_field]
    )
    loser_rating = float(
        loser_container[rating_field]
    )

    (
        winner_container[rating_field],
        loser_container[rating_field],
    ) = update_pair(
        winner_rating,
        loser_rating,
        k_factor,
    )


def process_match(
    match: dict[str, Any],
    players: dict[str, dict[str, Any]],
    counters: Counter,
) -> None:
    if not match.get("ready_for_tle"):
        counters["not_ready_for_tle"] += 1
        return

    gender = clean(match.get("gender")).lower()
    level = clean(match.get("tour_level")).lower()
    surface = clean(
        match.get("tournament", {}).get("surface")
    ).lower()
    match_date = clean(match.get("date"))

    if gender not in {"men", "women"}:
        counters["invalid_gender"] += 1
        return

    if level not in VALID_LEVELS:
        counters["invalid_level"] += 1
        return

    winner_raw = match.get("winner", {})
    loser_raw = match.get("loser", {})

    if not isinstance(winner_raw, dict):
        counters["invalid_winner"] += 1
        return

    if not isinstance(loser_raw, dict):
        counters["invalid_loser"] += 1
        return

    winner_key, winner_name = player_identity(
        winner_raw,
        gender,
    )
    loser_key, loser_name = player_identity(
        loser_raw,
        gender,
    )

    if (
        not winner_name
        or not loser_name
        or winner_key == loser_key
    ):
        counters["invalid_player_identity"] += 1
        return

    if winner_key not in players:
        players[winner_key] = new_player_state(
            player_key=winner_key,
            display_name=winner_name,
            gender=gender,
            sackmann_player_id=winner_raw.get(
                "sackmann_player_id"
            ),
        )

    if loser_key not in players:
        players[loser_key] = new_player_state(
            player_key=loser_key,
            display_name=loser_name,
            gender=gender,
            sackmann_player_id=loser_raw.get(
                "sackmann_player_id"
            ),
        )

    winner = players[winner_key]
    loser = players[loser_key]

    winner["display_name"] = winner_name
    loser["display_name"] = loser_name

    update_date_range(
        winner,
        match_date,
    )
    update_date_range(
        loser,
        match_date,
    )

    winner_country = clean(
        winner_raw.get("country")
    )
    loser_country = clean(
        loser_raw.get("country")
    )

    if winner_country:
        winner["countries"][
            winner_country
        ] += 1

    if loser_country:
        loser["countries"][
            loser_country
        ] += 1

    # 1. Global overall ELO
    update_rating_layer(
        winner["global"],
        loser["global"],
        "overall_elo",
        GLOBAL_K,
    )

    winner["global"]["matches"] += 1
    loser["global"]["matches"] += 1
    winner["global"]["wins"] += 1

    # 2. Global surface ELO
    if surface in VALID_SURFACES:
        winner_global_surface = ensure_surface(
            winner["global"],
            surface,
        )
        loser_global_surface = ensure_surface(
            loser["global"],
            surface,
        )

        update_rating_layer(
            winner_global_surface,
            loser_global_surface,
            "elo",
            GLOBAL_SURFACE_K,
        )

        winner_global_surface["matches"] += 1
        loser_global_surface["matches"] += 1
        winner_global_surface["wins"] += 1
    else:
        counters["unknown_surface"] += 1

    # 3. Level overall ELO
    winner_level = ensure_level(
        winner,
        level,
    )
    loser_level = ensure_level(
        loser,
        level,
    )

    update_rating_layer(
        winner_level,
        loser_level,
        "overall_elo",
        LEVEL_K,
    )

    winner_level["matches"] += 1
    loser_level["matches"] += 1
    winner_level["wins"] += 1

    # 4. Level-specific surface ELO
    if surface in VALID_SURFACES:
        winner_level_surface = ensure_surface(
            winner_level,
            surface,
        )
        loser_level_surface = ensure_surface(
            loser_level,
            surface,
        )

        update_rating_layer(
            winner_level_surface,
            loser_level_surface,
            "elo",
            LEVEL_SURFACE_K,
        )

        winner_level_surface["matches"] += 1
        loser_level_surface["matches"] += 1
        winner_level_surface["wins"] += 1

    counters["processed_matches"] += 1
    counters[f"processed_{gender}"] += 1
    counters[f"processed_level_{level}"] += 1


def serializable_player(
    player: dict[str, Any],
) -> dict[str, Any]:
    output = {
        "player_key": player["player_key"],
        "display_name": player["display_name"],
        "gender": player["gender"],
        "sackmann_player_id": player[
            "sackmann_player_id"
        ],
        "first_match_date": player[
            "first_match_date"
        ],
        "last_match_date": player[
            "last_match_date"
        ],
        "country": (
            player["countries"].most_common(1)[0][0]
            if player["countries"]
            else None
        ),
        "global": {
            "overall_elo": round(
                float(
                    player["global"][
                        "overall_elo"
                    ]
                ),
                3,
            ),
            "matches": int(
                player["global"]["matches"]
            ),
            "wins": int(
                player["global"]["wins"]
            ),
            "win_rate": round(
                (
                    player["global"]["wins"]
                    / player["global"]["matches"]
                ),
                4,
            )
            if player["global"]["matches"]
            else None,
            "surfaces": {},
        },
        "levels": {},
    }

    for surface, state in sorted(
        player["global"]["surfaces"].items()
    ):
        output["global"]["surfaces"][surface] = {
            "elo": round(float(state["elo"]), 3),
            "matches": int(state["matches"]),
            "wins": int(state["wins"]),
            "win_rate": round(
                state["wins"] / state["matches"],
                4,
            )
            if state["matches"]
            else None,
        }

    for level, state in sorted(
        player["levels"].items()
    ):
        level_output = {
            "overall_elo": round(
                float(state["overall_elo"]),
                3,
            ),
            "matches": int(state["matches"]),
            "wins": int(state["wins"]),
            "win_rate": round(
                state["wins"] / state["matches"],
                4,
            )
            if state["matches"]
            else None,
            "surfaces": {},
        }

        for surface, surface_state in sorted(
            state["surfaces"].items()
        ):
            level_output["surfaces"][surface] = {
                "elo": round(
                    float(surface_state["elo"]),
                    3,
                ),
                "matches": int(
                    surface_state["matches"]
                ),
                "wins": int(
                    surface_state["wins"]
                ),
                "win_rate": round(
                    (
                        surface_state["wins"]
                        / surface_state["matches"]
                    ),
                    4,
                )
                if surface_state["matches"]
                else None,
            }

        output["levels"][level] = level_output

    return output


def top_players(
    players: list[dict[str, Any]],
    gender: str,
    rating_path: tuple[str, ...],
    minimum_matches: int,
    limit: int = 20,
) -> list[dict[str, Any]]:
    rows = []

    for player in players:
        if player["gender"] != gender:
            continue

        current: Any = player

        try:
            for key in rating_path:
                current = current[key]
        except (KeyError, TypeError):
            continue

        if not isinstance(current, dict):
            continue

        matches = int(current.get("matches", 0))
        rating = current.get(
            "overall_elo",
            current.get("elo"),
        )

        if (
            rating is None
            or matches < minimum_matches
        ):
            continue

        rows.append(
            {
                "player_key": player["player_key"],
                "display_name": player[
                    "display_name"
                ],
                "rating": rating,
                "matches": matches,
            }
        )

    rows.sort(
        key=lambda row: (
            row["rating"],
            row["matches"],
        ),
        reverse=True,
    )

    return rows[:limit]


def main() -> None:
    manifest = read_manifest()
    players: dict[str, dict[str, Any]] = {}
    counters: Counter = Counter()

    for match in iter_matches(manifest):
        process_match(
            match,
            players,
            counters,
        )

    serializable_players = [
        serializable_player(player)
        for player in players.values()
    ]
    serializable_players.sort(
        key=lambda row: (
            row["gender"],
            row["display_name"].lower(),
            row["player_key"],
        )
    )

    RATINGS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ratings_payload = {
        "generated_at": now_iso(),
        "model_family": "tle",
        "model": "tle_ratings_v1",
        "schema_version": 1,
        "settings": {
            "default_elo": DEFAULT_ELO,
            "global_k": GLOBAL_K,
            "global_surface_k": GLOBAL_SURFACE_K,
            "level_k": LEVEL_K,
            "level_surface_k": LEVEL_SURFACE_K,
        },
        "source_manifest": str(
            MANIFEST_FILE.relative_to(ROOT_DIR)
        ),
        "summary": {
            "players_total": len(
                serializable_players
            ),
            "players_men": sum(
                row["gender"] == "men"
                for row in serializable_players
            ),
            "players_women": sum(
                row["gender"] == "women"
                for row in serializable_players
            ),
            **dict(counters),
        },
        "players": serializable_players,
    }

    with gzip.open(
        RATINGS_FILE,
        "wt",
        encoding="utf-8",
    ) as handle:
        json.dump(
            ratings_payload,
            handle,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    file_size_mb = (
        RATINGS_FILE.stat().st_size
        / 1024
        / 1024
    )

    ratings_manifest = {
        "generated_at": ratings_payload[
            "generated_at"
        ],
        "model": ratings_payload["model"],
        "schema_version": 1,
        "ratings_file": str(
            RATINGS_FILE.relative_to(ROOT_DIR)
        ),
        "ratings_file_size_mb": round(
            file_size_mb,
            3,
        ),
        "summary": ratings_payload["summary"],
        "settings": ratings_payload["settings"],
    }

    save_json(
        RATINGS_MANIFEST_FILE,
        ratings_manifest,
    )

    report = {
        "generated_at": ratings_payload[
            "generated_at"
        ],
        "model": ratings_payload["model"],
        "settings": ratings_payload["settings"],
        "summary": ratings_payload["summary"],
        "ratings_file": ratings_manifest[
            "ratings_file"
        ],
        "ratings_file_size_mb": ratings_manifest[
            "ratings_file_size_mb"
        ],
        "top_global": {
            "men": top_players(
                serializable_players,
                "men",
                ("global",),
                minimum_matches=20,
            ),
            "women": top_players(
                serializable_players,
                "women",
                ("global",),
                minimum_matches=20,
            ),
        },
        "top_by_level": {},
    }

    for level in sorted(VALID_LEVELS):
        report["top_by_level"][level] = {
            "men": top_players(
                serializable_players,
                "men",
                ("levels", level),
                minimum_matches=10,
            ),
            "women": top_players(
                serializable_players,
                "women",
                ("levels", level),
                minimum_matches=10,
            ),
        }

    save_json(
        REPORT_FILE,
        report,
    )

    print("")
    print("TLE RATINGS BUILD DONE")
    print("SUMMARY:", ratings_payload["summary"])
    print(
        "RATINGS FILE SIZE MB:",
        ratings_manifest[
            "ratings_file_size_mb"
        ],
    )
    print("\nTOP GLOBAL MEN:")

    for row in report["top_global"]["men"][:10]:
        print(row)

    print("\nTOP GLOBAL WOMEN:")

    for row in report["top_global"]["women"][:10]:
        print(row)

    print(f"\nRatings:  {RATINGS_FILE}")
    print(f"Manifest: {RATINGS_MANIFEST_FILE}")
    print(f"Report:   {REPORT_FILE}")
    print("")


if __name__ == "__main__":
    main()
