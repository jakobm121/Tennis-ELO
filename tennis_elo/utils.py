import json
import os
import re
import unicodedata
from datetime import datetime
from zoneinfo import ZoneInfo

from tennis_elo.config import TZ_NAME


def now_iso():
    return datetime.now(ZoneInfo(TZ_NAME)).isoformat()


def ensure_parent(path):
    folder = os.path.dirname(str(path))
    if folder:
        os.makedirs(folder, exist_ok=True)


def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def save_text(path, text):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def clean_str(value):
    return str(value or "").strip()


def normalize_text(value):
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace(",", " ")
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = text.replace(".", " ")
    text = text.replace("'", " ")
    text = text.replace("’", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonical_player_name(value):
    return normalize_text(value)


def canonical_key_part(value):
    return normalize_text(value).replace(" ", "_")
