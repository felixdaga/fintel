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
    "llm": "fintel.agents.installed.llm_agent:LLMAgent",
    "optimized": "fintel.agents.adapters.optimized:OptimizedFintelAgent",
    "djia_strategy_adapter_for_llm_agent": "fintel.agents.adapters.djia_strategy_adapter_for_llm_agent:DjiaStrategyAdapterForLlmAgent",
    "openclaw": "fintel.agents.adapters.openclaw:OpenClawAgent",
    "claude-code": "fintel.agents.adapters.claude_code:ClaudeCodeAgent",
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


def preflight(name: str, **params: Any) -> list[str]:
    """Whether a named agent can actually run, without building or invoking it.

    Always checks the standard adapter requirements (``pit_enforcement`` must
    be ``access`` or ``cli_deny``), then the adapter's own ``preflight_checks``
    hook if it declares one. An adapter with no hook still has to declare PIT
    enforcement — that is not optional.
    """
    target = AGENTS.get(name, name if ":" in name else None)
    if target is None:
        return [f"unknown agent {name!r}. Available: {', '.join(names())}."]
    cls = resolve(target)
    problems: list[str] = []
    problems.extend(_check_pit_enforcement(name, cls))

    hook = getattr(cls, "preflight_checks", None)
    if hook is not None:
        problems.extend(hook(**params))
    return problems


def _check_pit_enforcement(name: str, cls: type) -> list[str]:
    allowed = ("access", "cli_deny")
    enforcement = getattr(cls, "pit_enforcement", None)
    if enforcement is None:
        return [f"agent {name!r} does not declare pit_enforcement (must be one of {allowed})"]
    if enforcement not in allowed:
        return [
            f"agent {name!r} has unknown pit_enforcement {enforcement!r} (must be one of {allowed})"
        ]
    if enforcement != "cli_deny":
        return []
    # SubprocessAgent.enforce_pit_policy raises NotImplementedError; a real
    # cli_deny adapter must override it. Identity check against the base.
    try:
        from fintel.agents.adapters.base import SubprocessAgent
    except ImportError:
        return []
    if not isinstance(cls, type) or not issubclass(cls, SubprocessAgent):
        # Custom cli_deny host — must still expose the hook.
        if not callable(getattr(cls, "enforce_pit_policy", None)):
            return [
                f"agent {name!r} declares pit_enforcement='cli_deny' but has no enforce_pit_policy"
            ]
        return []
    if cls.enforce_pit_policy is SubprocessAgent.enforce_pit_policy:
        return [
            f"agent {name!r} declares pit_enforcement='cli_deny' but "
            "does not override enforce_pit_policy"
        ]
    return []
