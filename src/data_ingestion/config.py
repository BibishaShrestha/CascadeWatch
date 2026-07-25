"""Small credential loader for data ingestion scripts.

Reads from environment variables first, falling back to a gitignored
`.env` file at the repo root (KEY=value per line, no quoting). Deliberately
tiny - not pulling in python-dotenv for two variables.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _REPO_ROOT / ".env"


def _load_dotenv_once() -> dict[str, str]:
    values: dict[str, str] = {}
    if _ENV_FILE.exists():
        for line in _ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


_DOTENV = _load_dotenv_once()


def get_opentopography_key() -> str | None:
    return os.environ.get("OPENTOPOGRAPHY_API_KEY") or _DOTENV.get("OPENTOPOGRAPHY_API_KEY")


def get_gee_project_id() -> str | None:
    return os.environ.get("GEE_PROJECT_ID") or _DOTENV.get("GEE_PROJECT_ID")


GEE_SERVICE_ACCOUNT_PATH = _REPO_ROOT / "secrets" / "gee-service-account.json"


def get_gee_service_account_path() -> Path | None:
    return GEE_SERVICE_ACCOUNT_PATH if GEE_SERVICE_ACCOUNT_PATH.exists() else None
