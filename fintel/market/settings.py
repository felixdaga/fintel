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

_TRUE = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class MarketConfig:
    """`cache_root` defaults into the strategy package so a package ships with
    the data that reproduces it. `offline` turns a cache miss into an error
    instead of a network call, which is what makes a replay honest."""

    cache_root: Path
    offline: bool = False
    massive_api_key: str | None = None

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
