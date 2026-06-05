# Tennis Elo Machine

Clean starter repo for building our own tennis ELO from normalized match history.

Current first version:

1. `flashscore_history_parser.py` — proof-of-concept parser for one or more Flashscore match/player URLs.
2. `normalize_matches.py` — converts raw parsed rows into a common match format.
3. `dedupe_matches.py` — creates canonical match keys and removes duplicate matches.
4. `build_elo.py` — builds simple overall + surface ELO from canonical matches.
5. `run_pipeline.py` — runs normalize -> dedupe -> build ELO.

Important:
- This is a starter/proof-of-concept.
- It does not bypass blocked pages.
- If Flashscore shows an error page, the parser saves debug output and marks it as blocked.
