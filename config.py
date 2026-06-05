from pathlib import Path

TZ_NAME = "Europe/Ljubljana"

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
NORMALIZED_DIR = DATA_DIR / "normalized"
PROCESSED_DIR = DATA_DIR / "processed"
RATINGS_DIR = DATA_DIR / "ratings"
REPORTS_DIR = DATA_DIR / "reports"
CONFIG_DIR = ROOT_DIR / "config"

FLASH_HISTORY_RAW_FILE = RAW_DIR / "flashscore_history_raw.json"
FLASH_HISTORY_NORMALIZED_FILE = NORMALIZED_DIR / "flashscore_matches_normalized.json"
CANONICAL_MATCHES_FILE = PROCESSED_DIR / "canonical_matches.json"

PLAYER_RATINGS_FILE = RATINGS_DIR / "player_ratings_latest.json"
ELO_REPORT_FILE = REPORTS_DIR / "elo_report.md"

WATCHLIST_FILE = CONFIG_DIR / "flashscore_urls.txt"

DEFAULT_ELO = 1500.0
K_FACTOR = 24.0
SURFACE_K_FACTOR = 20.0
