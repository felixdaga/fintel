"""User-side configuration: where the cache lives, and credentials.

The one thing the *user* configures. A strategy points at universes and data by
name; it never carries a key or a path to someone else's disk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_CACHE = "FINTEL_CACHE"
ENV_OFFLINE = "FINTEL_OFFLINE"
ENV_MASSIVE_KEY = "MASSIVE_API_KEY"
ENV_BRAVE_KEY = "BRAVE_API_KEY"
ENV_FRED_KEY = "FRED_API_KEY"
ENV_AV_KEY = "ALPHA_VANTAGE_API_KEY"

_TRUE = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class MarketConfig:
    """`cache_root` defaults into the strategy package so a package ships with
    the data that reproduces it. `offline` turns a cache miss into an error
    instead of a network call, which is what makes a replay honest."""

    cache_root: Path
    offline: bool = False
    massive_api_key: str | None = None
    brave_api_key: str | None = None
    fred_api_key: str | None = None
    alphavantage_api_key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cache_root, Path):
            object.__setattr__(self, "cache_root", Path(self.cache_root).expanduser())

    @classmethod
    def from_env(cls, cache_root: str | Path | None = None) -> MarketConfig:
        root = cache_root or os.environ.get(ENV_CACHE) or "cache"
        return cls(
            cache_root=Path(root).expanduser(),
            offline=os.environ.get(ENV_OFFLINE, "").lower() in _TRUE,
            massive_api_key=os.environ.get(ENV_MASSIVE_KEY) or None,
            brave_api_key=os.environ.get(ENV_BRAVE_KEY) or None,
            fred_api_key=os.environ.get(ENV_FRED_KEY) or None,
            alphavantage_api_key=os.environ.get(ENV_AV_KEY) or None,
        )

    def to_dict(self, *, secrets: bool = False) -> dict:
        """Serialize for bindings.json. Keys stay out of the session dir by
        default (session layout forbids secrets); the MCP subprocess picks them
        up from its own env block instead."""
        data = {
            "cache_root": str(self.cache_root),
            "offline": self.offline,
        }
        if secrets:
            if self.massive_api_key:
                data["massive_api_key"] = self.massive_api_key
            if self.brave_api_key:
                data["brave_api_key"] = self.brave_api_key
        return data

    @classmethod
    def from_dict(cls, data: dict | None) -> MarketConfig:
        """Rebuild from bindings.json. Missing keys fall back to the process
        env — what the OpenClaw MCP server env block already injects."""
        data = data or {}
        if "cache_root" not in data:
            raise ValueError(
                "bindings.json is missing config.cache_root; the orchestrator "
                "must persist MarketConfig before launching a subprocess agent"
            )
        return cls(
            cache_root=Path(data["cache_root"]).expanduser(),
            offline=bool(data.get("offline", False)),
            massive_api_key=data.get("massive_api_key") or os.environ.get(ENV_MASSIVE_KEY) or None,
            brave_api_key=data.get("brave_api_key") or os.environ.get(ENV_BRAVE_KEY) or None,
        )

    def dir(self, *parts: str) -> Path:
        p = self.cache_root.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def require_key(self, name: str, value: str | None) -> str:
        if not value:
            raise RuntimeError(
                f"{name} is required for this data source; set it in the environment "
                f"or run with {ENV_OFFLINE}=1 against a populated cache"
            )
        return value
