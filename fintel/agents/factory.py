"""Name → agent. The only place an adapter gets constructed.

Valid names are derived from `AGENTS`, never listed separately. Harbor keeps a
name enum beside its factory map and the two have already drifted — two of its
names pass validation and then fail to import. One dict cannot disagree with
itself.
"""

from __future__ import annotations

from typing import Any

from fintel.agents.base import Agent
from fintel.utils.import_path import resolve

AGENTS: dict[str, str] = {
    "scripted": "fintel.agents.scripted:ScriptedAgent",
    "constant": "fintel.agents.scripted:ConstantAgent",
}


def names() -> list[str]:
    return sorted(AGENTS)


def register(name: str, target: str, *, replace: bool = False) -> None:
    if name in AGENTS and not replace:
        raise ValueError(f"agent {name!r} is already registered as {AGENTS[name]!r}")
    AGENTS[name] = target


def build(name: str, **params: Any) -> Agent:
    """Build a named builtin, or any `module:Class` — so a one-off adapter needs
    no entry here, and a package can ship its own."""
    target = AGENTS.get(name)
    if target is None:
        if ":" not in name:
            raise ValueError(
                f"unknown agent {name!r}. Available: {', '.join(names())}. "
                "A custom adapter is addressed as 'module.path:ClassName'."
            )
        target = name
    cls = resolve(target)
    try:
        return cls(**params)
    except TypeError as exc:
        raise TypeError(f"cannot build agent {name!r} with {sorted(params)}: {exc}") from exc
