import json
import os
from pathlib import Path


def load_json_object(config_path):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return payload


def require_text(payload, key):
    if key not in payload:
        raise ValueError(f"Missing required config key: {key}")
    value = str(payload[key]).strip()
    if not value:
        raise ValueError(f"Config key '{key}' cannot be empty")
    return value


def require_positive_int(payload, key):
    if key not in payload:
        raise ValueError(f"Missing required config key: {key}")
    value = int(payload[key])
    if value <= 0:
        raise ValueError(f"Config key '{key}' must be > 0, got: {value}")
    return value


def normalize_path_text(value):
    raw = str(value).strip()
    if not raw:
        return raw
    if os.sep == "/":
        return raw.replace("\\", "/")
    return raw.replace("/", "\\")


def coerce_path(value):
    return Path(normalize_path_text(value))


def path_to_portable_str(path):
    return Path(path).as_posix()
