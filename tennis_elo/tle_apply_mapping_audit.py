#!/usr/bin/env python3
"""Apply safe player mapping fixes from the AI TLE mapping audit.

Target location in Tennis-ELO repo:
  tennis_elo/tle_apply_mapping_audit.py

Default behavior:
  - reads AI audit latest JSON from GitHub raw
  - reads local data/tle/mappings/api_player_to_sackmann.json
  - auto-applies only review rows with resolve_method == unique_surname_initial
    where audit already has a Sackmann key and display name
  - does NOT auto-map api_key_unmapped rows; writes them to review table
  - writes a concise Markdown repair table
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

DEFAULT_AUDIT_URL = (
    "https://raw.githubusercontent.com/jakobm121/Ai/refs/heads/main/"
    "data/tle/mapping_audit/ai_tle_mapping_audit_latest.json"
)
DEFAULT_MAPPING_PATH = Path("data/tle/mappings/api_player_to_sackmann.json")
DEFAULT_TABLE_PATH = Path("data/tle/mappings/mapping_repair_table.md")

SAFE_AUTO_METHODS = {"unique_surname_initial"}
MANUAL_METHODS = {"api_key_unmapped", "name_fallback", "unresolved", "invalid_gender"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())


def read_json(path_or_url: str | Path) -> dict[str, Any]:
    text_ref = str(path_or_url)
    if text_ref.startswith("http://") or text_ref.startswith("https://"):
        req = Request(text_ref, headers={"User-Agent": "tle-apply-mapping-audit"})
        with urlopen(req, timeout=30) as resp:  # nosec - intended GitHub raw read
            return json.loads(resp.read().decode("utf-8"))
    with Path(path_or_url).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_sackmann_key(tle_key: str) -> tuple[str, int, str] | None:
    # example: women:sackmann:221278
    m = re.fullmatch(r"(men|women):sackmann:(\d+)", clean_text(tle_key))
    if not m:
        return None
    gender = m.group(1)
    sackmann_id = int(m.group(2))
    return gender, sackmann_id, f"{gender}:sackmann:{sackmann_id}"


def mapping_entry_from_review(row: dict[str, Any]) -> dict[str, Any] | None:
    parsed = parse_sackmann_key(row.get("tle_key", ""))
    if not parsed:
        return None
    gender, sackmann_id, sackmann_key = parsed
    suggested_name = clean_text(row.get("tle_display_name"))
    api_name = clean_text(row.get("api_name"))
    api_key = row.get("api_player_key")
    if not suggested_name or api_key in (None, ""):
        return None
    return {
        "api_player_key": int(api_key),
        "api_name": api_name,
        "gender": gender,
        "matches_seen": 0,
        "levels": {},
        "status": "matched",
        "method": "manual_audit_unique_surname_initial",
        "confidence": 1.0,
        "sackmann_player_id": sackmann_id,
        "sackmann_player_key": sackmann_key,
        "sackmann_name": suggested_name,
        "needs_review": False,
        "audit_source_event_key": clean_text(row.get("event_key")),
        "audit_source_generated_at": "",
    }


def recompute_summary(mapping: dict[str, Any]) -> None:
    players = mapping.get("players") or {}
    status_counts: Counter[str] = Counter()
    matches_by_status: Counter[str] = Counter()
    for p in players.values():
        status = clean_text(p.get("status")) or "unknown"
        status_counts[status] += 1
        try:
            matches_seen = int(p.get("matches_seen") or 0)
        except Exception:
            matches_seen = 0
        matches_by_status[status] += matches_seen

    summary = mapping.setdefault("summary", {})
    summary["api_players"] = len(players)
    summary["mapping_statuses"] = dict(sorted(status_counts.items()))
    summary["matches_by_status"] = dict(sorted(matches_by_status.items()))
    summary["matched_players"] = status_counts.get("matched", 0)
    summary["unresolved_players"] = sum(v for k, v in status_counts.items() if k != "matched")


def md_escape(value: Any) -> str:
    return clean_text(value).replace("|", "\\|")


def build_table(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# TLE Mapping Repair Table")
    lines.append("")
    lines.append(f"Generated: `{utc_now()}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key in [
        "audit_generated_at",
        "review_players",
        "auto_added",
        "already_ok",
        "conflicts",
        "needs_manual_review",
        "skipped",
    ]:
        lines.append(f"- {key}: `{summary.get(key, 0)}`")
    lines.append("")
    lines.append("## Rows")
    lines.append("")
    lines.append("| Action | API key | API name | Gender | Method | Suggested TLE key | Suggested name | Event | Match/Tournament | Note |")
    lines.append("|---|---:|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    md_escape(r.get("action")),
                    md_escape(r.get("api_player_key")),
                    md_escape(r.get("api_name")),
                    md_escape(r.get("gender")),
                    md_escape(r.get("resolve_method")),
                    md_escape(r.get("tle_key")),
                    md_escape(r.get("tle_display_name")),
                    md_escape(r.get("event_key")),
                    md_escape(r.get("context")),
                    md_escape(r.get("note")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def make_event_context(events_by_key: dict[str, dict[str, Any]], event_key: str) -> str:
    ev = events_by_key.get(clean_text(event_key)) or {}
    match = clean_text(ev.get("match"))
    tournament = clean_text(ev.get("tournament"))
    round_name = clean_text(ev.get("round"))
    parts = [x for x in [match, tournament, round_name] if x]
    return " / ".join(parts[:3])


def apply_audit(
    audit: dict[str, Any],
    mapping: dict[str, Any],
    *,
    apply: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    result_mapping = copy.deepcopy(mapping)
    players = result_mapping.setdefault("players", {})
    audit_generated_at = clean_text(((audit.get("summary") or {}).get("generated_at")))

    events = audit.get("events") or []
    events_by_key = {clean_text(e.get("event_key")): e for e in events if clean_text(e.get("event_key"))}

    review_players = audit.get("review_players") or []
    # Deduplicate by api player key, keeping first seen row.
    by_api_key: dict[str, dict[str, Any]] = {}
    for row in review_players:
        api_key = clean_text(row.get("api_player_key"))
        if api_key and api_key not in by_api_key:
            by_api_key[api_key] = row

    table_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()

    for api_key_str, row in sorted(by_api_key.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 999999999):
        method = clean_text(row.get("resolve_method"))
        existing = players.get(api_key_str)
        suggested_entry = mapping_entry_from_review(row)
        action = "SKIPPED"
        note = ""

        if method in SAFE_AUTO_METHODS and suggested_entry:
            suggested_entry["audit_source_generated_at"] = audit_generated_at
            if existing:
                existing_status = clean_text(existing.get("status"))
                existing_key = clean_text(existing.get("sackmann_player_key"))
                suggested_key = clean_text(suggested_entry.get("sackmann_player_key"))
                if existing_status == "matched" and existing_key == suggested_key:
                    action = "ALREADY_OK"
                    counters["already_ok"] += 1
                elif existing_status in {"review", "unmatched", ""} or not existing_key:
                    # Preserve old usage counts/levels when available.
                    suggested_entry["matches_seen"] = int(existing.get("matches_seen") or 0) if isinstance(existing, dict) else 0
                    suggested_entry["levels"] = existing.get("levels") or {} if isinstance(existing, dict) else {}
                    if apply:
                        players[api_key_str] = suggested_entry
                    action = "ADD_MAPPING"
                    counters["auto_added"] += 1
                    note = f"replaced existing status={existing_status or 'unknown'}"
                else:
                    action = "CONFLICT"
                    counters["conflicts"] += 1
                    note = f"existing {existing_key} != suggested {suggested_key}"
            else:
                if apply:
                    players[api_key_str] = suggested_entry
                action = "ADD_MAPPING"
                counters["auto_added"] += 1
        elif method in MANUAL_METHODS:
            action = "NEEDS_MANUAL_REVIEW"
            counters["needs_manual_review"] += 1
            if method == "api_key_unmapped":
                note = "API player key is not mapped to Sackmann yet"
        else:
            action = "SKIPPED"
            counters["skipped"] += 1
            if not suggested_entry:
                note = "missing valid Sackmann suggestion"

        table_rows.append(
            {
                "action": action,
                "api_player_key": row.get("api_player_key"),
                "api_name": clean_text(row.get("api_name")),
                "gender": clean_text(row.get("gender")),
                "resolve_method": method,
                "tle_key": clean_text(row.get("tle_key")),
                "tle_display_name": clean_text(row.get("tle_display_name")),
                "event_key": clean_text(row.get("event_key")),
                "context": make_event_context(events_by_key, clean_text(row.get("event_key"))),
                "note": note,
            }
        )

    repair_meta = {
        "updated_at": utc_now(),
        "audit_generated_at": audit_generated_at,
        "auto_methods": sorted(SAFE_AUTO_METHODS),
        "apply_mode": apply,
        "auto_added": counters["auto_added"],
        "already_ok": counters["already_ok"],
        "conflicts": counters["conflicts"],
        "needs_manual_review": counters["needs_manual_review"],
        "skipped": counters["skipped"],
    }
    result_mapping.setdefault("repair_history", []).append(repair_meta)
    recompute_summary(result_mapping)

    summary = dict(repair_meta)
    summary["review_players"] = len(by_api_key)
    return result_mapping, table_rows, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply safe mappings from AI TLE mapping audit.")
    parser.add_argument("--audit", default=DEFAULT_AUDIT_URL, help="Audit JSON path or URL")
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING_PATH), help="Local api_player_to_sackmann.json path")
    parser.add_argument("--table", default=str(DEFAULT_TABLE_PATH), help="Output markdown repair table path")
    parser.add_argument("--dry-run", action="store_true", help="Do not write mapping JSON; only write table")
    args = parser.parse_args(argv)

    audit = read_json(args.audit)
    mapping_path = Path(args.mapping)
    table_path = Path(args.table)
    mapping = read_json(mapping_path)

    apply_mode = not args.dry_run
    updated_mapping, rows, summary = apply_audit(audit, mapping, apply=apply_mode)

    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(build_table(rows, summary), encoding="utf-8")

    if apply_mode:
        write_json(mapping_path, updated_mapping)

    print("TLE mapping audit repair complete")
    print(f"audit_generated_at: {summary.get('audit_generated_at')}")
    print(f"review_players: {summary.get('review_players')}")
    print(f"auto_added: {summary.get('auto_added')}")
    print(f"already_ok: {summary.get('already_ok')}")
    print(f"conflicts: {summary.get('conflicts')}")
    print(f"needs_manual_review: {summary.get('needs_manual_review')}")
    print(f"skipped: {summary.get('skipped')}")
    print(f"table: {table_path}")
    if args.dry_run:
        print("dry_run: mapping JSON was not changed")
    else:
        print(f"mapping: {mapping_path}")
    return 0 if summary.get("conflicts", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
