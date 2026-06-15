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

try:
    from tennis_elo.config import ROOT_DIR
except Exception:  # pragma: no cover
    ROOT_DIR = Path.cwd()

DEFAULT_MAPPING = ROOT_DIR / "data" / "tle" / "mappings" / "api_player_to_sackmann.json"
DEFAULT_CANONICAL_MANIFEST = ROOT_DIR / "data" / "tle" / "processed" / "canonical" / "tle_matches_manifest.json"
DEFAULT_API_MANIFEST = ROOT_DIR / "data" / "tle" / "source" / "api" / "tle_api_matches_manifest.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "tle" / "reports" / "player_mapping_quality"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def norm(value: Any) -> str:
    text = unicodedata.normalize("NFKD", clean_text(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def surname_norm(value: Any) -> str:
    parts = norm(value).split()
    if not parts:
        return ""
    return parts[-1]


def initials(value: Any) -> str:
    return "".join(part[:1] for part in norm(value).split() if part)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def manifest_files(manifest_path: Path) -> list[Path]:
    payload = load_json(manifest_path)
    files: list[Path] = []
    for item in payload.get("year_files") or []:
        rel = item.get("path")
        if not rel:
            continue
        p = Path(rel)
        if not p.is_absolute():
            p = ROOT_DIR / p
        files.append(p)
    return files


def player_key_from_player(player: dict[str, Any], gender: str) -> str:
    key = clean_text(player.get("player_key"))
    if key.startswith(("men:", "women:")):
        return key
    # API importer stores fallback player_key as api:<id> or name:<norm> without gender prefix.
    if key.startswith("api:") or key.startswith("name:"):
        return f"{gender}:{key}"
    sack_id = player.get("sackmann_player_id")
    if sack_id not in {None, ""}:
        try:
            return f"{gender}:sackmann:{int(sack_id)}"
        except Exception:
            pass
    api_key = player.get("api_player_key")
    if api_key not in {None, ""}:
        try:
            return f"{gender}:api:{int(api_key)}"
        except Exception:
            return f"{gender}:api:{clean_text(api_key)}"
    name = clean_text(player.get("name"))
    return f"{gender}:name:{norm(name).replace(' ', '_')}" if name else ""


def iter_match_players(match: dict[str, Any]):
    gender = clean_text(match.get("gender")).lower()
    if gender not in {"men", "women"}:
        gender = "unknown"
    date = clean_text(match.get("date"))
    level = clean_text(match.get("tour_level")) or clean_text(match.get("level"))
    source = clean_text(match.get("source"))
    tournament = match.get("tournament") or {}
    if not isinstance(tournament, dict):
        tournament = {}
    tournament_name = clean_text(tournament.get("name"))
    surface = clean_text(tournament.get("surface"))

    for side in ("winner", "loser"):
        player = match.get(side) or {}
        if not isinstance(player, dict):
            continue
        name = clean_text(player.get("name"))
        key = player_key_from_player(player, gender)
        if not key:
            continue
        yield {
            "key": key,
            "name": name,
            "gender": gender,
            "date": date,
            "level": level,
            "source": source,
            "surface": surface,
            "tournament": tournament_name,
            "mapping_source": clean_text(player.get("mapping_source")),
            "api_player_key": player.get("api_player_key"),
            "sackmann_player_id": player.get("sackmann_player_id"),
        }


def add_player_seen(stats: dict[str, dict[str, Any]], seen: dict[str, Any]) -> None:
    key = seen["key"]
    row = stats.setdefault(
        key,
        {
            "key": key,
            "name_counts": Counter(),
            "matches": 0,
            "sources": Counter(),
            "levels": Counter(),
            "surfaces": Counter(),
            "latest_date": "",
            "latest_tournament": "",
            "api_player_keys": Counter(),
            "mapping_sources": Counter(),
        },
    )
    row["matches"] += 1
    if seen.get("name"):
        row["name_counts"][seen["name"]] += 1
    if seen.get("source"):
        row["sources"][seen["source"]] += 1
    if seen.get("level"):
        row["levels"][seen["level"]] += 1
    if seen.get("surface"):
        row["surfaces"][seen["surface"]] += 1
    if seen.get("mapping_source"):
        row["mapping_sources"][seen["mapping_source"]] += 1
    api_key = seen.get("api_player_key")
    if api_key not in {None, ""}:
        row["api_player_keys"][str(api_key)] += 1
    date = seen.get("date") or ""
    if date >= row["latest_date"]:
        row["latest_date"] = date
        row["latest_tournament"] = seen.get("tournament") or ""


def compact_counter(counter: Counter, limit: int = 8) -> dict[str, int]:
    return {str(k): int(v) for k, v in counter.most_common(limit)}


def display_name(row: dict[str, Any]) -> str:
    c = row.get("name_counts") or Counter()
    if not c:
        return ""
    return str(c.most_common(1)[0][0])


def serialise_player_stats(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for key, row in stats.items():
        out[key] = {
            "key": key,
            "name": display_name(row),
            "matches": int(row["matches"]),
            "sources": compact_counter(row["sources"]),
            "levels": compact_counter(row["levels"]),
            "surfaces": compact_counter(row["surfaces"]),
            "latest_date": row["latest_date"],
            "latest_tournament": row["latest_tournament"],
            "api_player_keys": compact_counter(row["api_player_keys"]),
            "mapping_sources": compact_counter(row["mapping_sources"]),
        }
    return out


def make_md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], limit: int) -> str:
    rows = rows[:limit]
    if not rows:
        return "_No rows._\n"
    header = "| " + " | ".join(title for title, _ in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, sep]
    for row in rows:
        vals = []
        for _, key in columns:
            val = row.get(key, "")
            if isinstance(val, (dict, list)):
                val = json.dumps(val, ensure_ascii=False)
            vals.append(clean_text(val).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit API -> Sackmann mapping quality against canonical TLE DB.")
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    parser.add_argument("--canonical-manifest", default=str(DEFAULT_CANONICAL_MANIFEST))
    parser.add_argument("--api-manifest", default=str(DEFAULT_API_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top", type=int, default=250)
    args = parser.parse_args()

    mapping_path = Path(args.mapping)
    canonical_manifest = Path(args.canonical_manifest)
    api_manifest = Path(args.api_manifest)
    output_dir = Path(args.output_dir)

    mapping_payload = load_json(mapping_path)
    mapping_players: dict[str, dict[str, Any]] = mapping_payload.get("players") or {}

    canonical_stats: dict[str, dict[str, Any]] = {}
    files_seen = []
    for path in manifest_files(canonical_manifest):
        files_seen.append(str(path.relative_to(ROOT_DIR) if str(path).startswith(str(ROOT_DIR)) else path))
        for match in read_jsonl_gz(path):
            for seen in iter_match_players(match):
                add_player_seen(canonical_stats, seen)

    api_source_stats: dict[str, dict[str, Any]] = {}
    api_manifest_exists = api_manifest.exists()
    if api_manifest_exists:
        for path in manifest_files(api_manifest):
            if not path.exists():
                continue
            for match in read_jsonl_gz(path):
                for seen in iter_match_players(match):
                    add_player_seen(api_source_stats, seen)

    canonical_by_norm: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    canonical_by_surname_initial: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for key, row in canonical_stats.items():
        name = display_name(row)
        gender = key.split(":", 1)[0] if ":" in key else "unknown"
        n = norm(name)
        if n:
            canonical_by_norm[(gender, n)].append({"key": key, "name": name, "matches": row["matches"]})
            canonical_by_surname_initial[(gender, surname_norm(name), initials(name)[:1])].append({"key": key, "name": name, "matches": row["matches"]})

    rows = []
    warning_counts = Counter()
    duplicate_api_only = []
    unresolved = []
    mapped_no_history = []
    suspicious = []

    for api_key, mp in mapping_players.items():
        api_name = clean_text(mp.get("api_name"))
        gender = clean_text(mp.get("gender")).lower()
        status = clean_text(mp.get("status"))
        method = clean_text(mp.get("method"))
        sack_key = clean_text(mp.get("sackmann_player_key"))
        sack_name = clean_text(mp.get("sackmann_name"))
        api_only_key = f"{gender}:api:{api_key}"

        warnings: list[str] = []
        mapped_stats = canonical_stats.get(sack_key) if sack_key else None
        api_only_stats = canonical_stats.get(api_only_key)
        source_api_only_stats = api_source_stats.get(api_only_key)

        canonical_matches = int(mapped_stats["matches"]) if mapped_stats else 0
        api_only_matches = int(api_only_stats["matches"]) if api_only_stats else 0
        source_api_only_matches = int(source_api_only_stats["matches"]) if source_api_only_stats else 0

        if status != "matched":
            warnings.append("UNRESOLVED_MAPPING")
        if status == "matched" and not sack_key:
            warnings.append("MATCHED_WITHOUT_SACKMANN_KEY")
        if status == "matched" and canonical_matches == 0:
            warnings.append("MAPPED_NO_CANONICAL_HISTORY")
        if status == "matched" and api_only_matches > 0:
            warnings.append("DUPLICATE_API_ONLY_IN_CANONICAL")
        if status == "matched" and source_api_only_matches > 0:
            warnings.append("API_SOURCE_STILL_UNMAPPED_FOR_THIS_KEY")
        if status == "matched" and sack_name and api_name:
            if surname_norm(api_name) != surname_norm(sack_name):
                warnings.append("SURNAME_MISMATCH")
            # api initial must be compatible with mapped first initial if API has an abbreviation.
            api_init = initials(api_name)[:1]
            sack_init = initials(sack_name)[:1]
            if api_init and sack_init and api_init != sack_init:
                warnings.append("FIRST_INITIAL_MISMATCH")

        for w in warnings:
            warning_counts[w] += 1

        row = {
            "api_key": api_key,
            "api_name": api_name,
            "gender": gender,
            "status": status,
            "method": method,
            "sackmann_key": sack_key,
            "sackmann_name": sack_name,
            "canonical_matches": canonical_matches,
            "api_only_key": api_only_key,
            "api_only_matches_canonical": api_only_matches,
            "api_only_matches_api_source": source_api_only_matches,
            "latest_date": mapped_stats.get("latest_date", "") if mapped_stats else "",
            "latest_tournament": mapped_stats.get("latest_tournament", "") if mapped_stats else "",
            "warnings": ", ".join(warnings),
        }
        rows.append(row)

        if "UNRESOLVED_MAPPING" in warnings:
            # add quick canonical candidates by exact normalized name and surname+initial
            cand = []
            cand.extend(canonical_by_norm.get((gender, norm(api_name)), []))
            si = (gender, surname_norm(api_name), initials(api_name)[:1])
            cand.extend(canonical_by_surname_initial.get(si, []))
            seen_cand = {}
            for c in cand:
                seen_cand[c["key"]] = c
            row["candidate_count"] = len(seen_cand)
            row["candidates"] = "; ".join(
                f"{c['key']} {c['name']} ({c['matches']})" for c in sorted(seen_cand.values(), key=lambda x: -x["matches"])[:8]
            )
            unresolved.append(row)
        if "DUPLICATE_API_ONLY_IN_CANONICAL" in warnings or "API_SOURCE_STILL_UNMAPPED_FOR_THIS_KEY" in warnings:
            duplicate_api_only.append(row)
        if "MAPPED_NO_CANONICAL_HISTORY" in warnings:
            mapped_no_history.append(row)
        if any(w in warnings for w in ["SURNAME_MISMATCH", "FIRST_INITIAL_MISMATCH", "MATCHED_WITHOUT_SACKMANN_KEY"]):
            suspicious.append(row)

    api_only_canonical_players = []
    for key, st in canonical_stats.items():
        if ":api:" in key or ":name:" in key:
            api_only_canonical_players.append({
                "key": key,
                "name": display_name(st),
                "matches": int(st["matches"]),
                "latest_date": st["latest_date"],
                "levels": compact_counter(st["levels"]),
                "sources": compact_counter(st["sources"]),
            })
    api_only_canonical_players.sort(key=lambda r: (-r["matches"], r["key"]))

    rows.sort(key=lambda r: (0 if r["warnings"] else 1, -int(r["canonical_matches"]), r["api_name"]))
    unresolved.sort(key=lambda r: (-int(r.get("candidate_count", 0)), -int(r.get("api_only_matches_api_source", 0)), r["api_name"]))
    duplicate_api_only.sort(key=lambda r: (-int(r["api_only_matches_canonical"]), -int(r["api_only_matches_api_source"]), r["api_name"]))
    mapped_no_history.sort(key=lambda r: (r["api_name"], r["api_key"]))
    suspicious.sort(key=lambda r: (r["warnings"], r["api_name"]))

    summary = {
        "generated_at": now_iso(),
        "mapping_file": str(mapping_path),
        "canonical_manifest": str(canonical_manifest),
        "canonical_files": files_seen,
        "mapping_summary": mapping_payload.get("summary") or {},
        "canonical_players_seen": len(canonical_stats),
        "api_source_players_seen": len(api_source_stats),
        "api_only_canonical_players": len(api_only_canonical_players),
        "warnings": dict(warning_counts),
        "unresolved_rows": len(unresolved),
        "duplicate_api_only_rows": len(duplicate_api_only),
        "mapped_no_history_rows": len(mapped_no_history),
        "suspicious_rows": len(suspicious),
    }

    payload = {
        "schema_version": 1,
        "file_type": "tle_player_mapping_quality_audit",
        "summary": summary,
        "warning_rows": rows,
        "unresolved": unresolved,
        "duplicate_api_only": duplicate_api_only,
        "mapped_no_history": mapped_no_history,
        "suspicious": suspicious,
        "api_only_canonical_players": api_only_canonical_players[: args.top],
        "canonical_player_stats_sample": serialise_player_stats(dict(list(canonical_stats.items())[: args.top])),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "tle_player_mapping_quality_audit_latest.json"
    md_path = output_dir / "tle_player_mapping_quality_audit_latest.md"
    save_json(json_path, payload)

    md = []
    md.append("# TLE Player Mapping Quality Audit\n")
    md.append(f"Generated: `{summary['generated_at']}`\n")
    md.append("## Summary\n")
    for k, v in summary.items():
        if k == "canonical_files":
            continue
        md.append(f"- {k}: `{json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}`")
    md.append("\n## Unresolved mappings\n")
    md.append(make_md_table(unresolved, [
        ("API key", "api_key"), ("API name", "api_name"), ("Gender", "gender"), ("Status", "status"),
        ("Method", "method"), ("API source unmapped sides", "api_only_matches_api_source"), ("Candidates", "candidates"), ("Warnings", "warnings"),
    ], args.top))
    md.append("\n## Duplicate API-only identities\n")
    md.append(make_md_table(duplicate_api_only, [
        ("API key", "api_key"), ("API name", "api_name"), ("Mapped key", "sackmann_key"), ("Mapped name", "sackmann_name"),
        ("Mapped matches", "canonical_matches"), ("Canonical API-only", "api_only_matches_canonical"), ("API source API-only", "api_only_matches_api_source"), ("Warnings", "warnings"),
    ], args.top))
    md.append("\n## Matched but no canonical history\n")
    md.append(make_md_table(mapped_no_history, [
        ("API key", "api_key"), ("API name", "api_name"), ("Mapped key", "sackmann_key"), ("Mapped name", "sackmann_name"), ("Method", "method"), ("Warnings", "warnings"),
    ], args.top))
    md.append("\n## Suspicious identity mismatches\n")
    md.append(make_md_table(suspicious, [
        ("API key", "api_key"), ("API name", "api_name"), ("Mapped key", "sackmann_key"), ("Mapped name", "sackmann_name"), ("Method", "method"), ("Warnings", "warnings"),
    ], args.top))
    md.append("\n## Canonical API-only players\n")
    md.append(make_md_table(api_only_canonical_players, [
        ("Key", "key"), ("Name", "name"), ("Matches", "matches"), ("Latest", "latest_date"), ("Levels", "levels"), ("Sources", "sources"),
    ], args.top))

    md_path.write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
