"""Load API keys from fintel's local ``.env/`` directory.

Keys live in ``<repo>/.env/keys.env`` (gitignored). ``bootstrap_env`` copies
them into ``os.environ`` without overwriting values already set in the shell.
This is the only secrets path — OpenClaw profiles are not read.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# fintel/utils/secrets.py → repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_DIR = _REPO_ROOT / ".env"
KEYS_FILENAME = "keys.env"

_KEY_LINE = re.compile(
    r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$"
)


def env_dir(path: Path | str | None = None) -> Path:
    return Path(path) if path else DEFAULT_ENV_DIR


def keys_path(directory: Path | str | None = None) -> Path:
    return env_dir(directory) / KEYS_FILENAME


def load_dotenv(path: Path | str) -> dict[str, str]:
    """Parse a KEY=VALUE env file. Missing file → empty dict."""
    file = Path(path)
    if not file.is_file():
        return {}
    out: dict[str, str] = {}
    try:
        text = file.read_text(encoding="utf-8")
    except OSError:
        return {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _KEY_LINE.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def load_env_dir(directory: Path | str | None = None) -> dict[str, str]:
    """Load ``keys.env`` plus any other ``*.env`` files in the directory.

    Later files in sorted order override earlier ones; ``keys.env`` is loaded
    last so it wins over optional extras.
    """
    root = env_dir(directory)
    if not root.is_dir():
        return {}
    merged: dict[str, str] = {}
    extras = sorted(p for p in root.glob("*.env") if p.name != KEYS_FILENAME)
    for path in extras:
        merged.update(load_dotenv(path))
    merged.update(load_dotenv(root / KEYS_FILENAME))
    return merged


def bootstrap_env(
    *,
    env_dir: Path | str | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Populate ``os.environ`` from ``.env/``. Returns what was newly set.

    Never overwrites a variable already present in the environment.
    """
    populated: dict[str, str] = {}

    def _set(name: str, value: str | None) -> None:
        if not value or os.environ.get(name):
            return
        os.environ[name] = value
        populated[name] = value

    for name, value in load_env_dir(env_dir).items():
        _set(name, value)
    for name, value in (extra or {}).items():
        _set(name, value)
    return populated


__all__ = [
    "DEFAULT_ENV_DIR",
    "KEYS_FILENAME",
    "bootstrap_env",
    "env_dir",
    "keys_path",
    "load_dotenv",
    "load_env_dir",
]
