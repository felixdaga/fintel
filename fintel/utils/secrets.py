"""Pull API keys from the openclaw profiles the delorean runs used.

fintel itself never stores secrets. The keys live where the operator put them
for openclaw: the OpenRouter key in ``~/.openclaw/openclaw.json`` (under
``models.providers.openrouter.apiKey``) and the data-source keys (MASSIVE,
BRAVE) in the delorean profile's MCP env block at
``~/.openclaw-delorean/openclaw.json`` (``mcp.servers.delorean.env``).

``bootstrap_env`` copies those into ``os.environ`` (without overwriting a value
the shell already set), so a run configured against the same profiles delorean
used needs no extra setup. This is the only place that reads those files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_PROFILE = Path.home() / ".openclaw"
DELOREAN_PROFILE = Path.home() / ".openclaw-delorean"


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def openrouter_key(profile_dir: Path | str | None = None) -> str | None:
    base = Path(profile_dir) if profile_dir else DEFAULT_PROFILE
    data = _load(base / "openclaw.json")
    return ((data.get("models") or {}).get("providers") or {}).get(
        "openrouter", {}
    ).get("apiKey")


def mcp_env(profile_dir: Path | str | None = None) -> dict[str, str]:
    """The ``mcp.servers.<name>.env`` blocks from a profile, flattened.

    delorean kept its data keys in ``mcp.servers.delorean.env``; we surface every
    server's env so a profile with keys under a different server name still works.
    """
    base = Path(profile_dir) if profile_dir else DELOREAN_PROFILE
    data = _load(base / "openclaw.json")
    servers = (data.get("mcp") or {}).get("servers") or {}
    out: dict[str, str] = {}
    for _name, cfg in servers.items():
        for k, v in (cfg.get("env") or {}).items():
            out.setdefault(k, str(v))
    return out


def bootstrap_env(
    *,
    profile: Path | str | None = None,
    delorean_profile: Path | str | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Populate ``os.environ`` from the openclaw profiles. Returns what was set.

    Never overwrites a variable already present in the environment, so an
    explicit shell export always wins. The OpenRouter key is exported as
    ``OPENROUTER_API_KEY``; everything in the delorean MCP env is exported
    under its own name (MASSIVE_API_KEY, BRAVE_API_KEY, ...).
    """
    populated: dict[str, str] = {}

    def _set(name: str, value: str | None) -> None:
        if not value or os.environ.get(name):
            return
        os.environ[name] = value
        populated[name] = value

    _set("OPENROUTER_API_KEY", openrouter_key(profile))
    for name, value in mcp_env(delorean_profile).items():
        _set(name, value)
    for name, value in (extra or {}).items():
        _set(name, value)
    return populated


__all__ = ["bootstrap_env", "mcp_env", "openrouter_key"]
