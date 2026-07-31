"""Agent adapters. One per agent; two hosts underneath.

The seven agents the old platform grew are points in a small cube — where the
code runs, how data reaches it, how the answer comes back — not seven kinds of
thing. In-process desks and subprocess CLIs are the only two hosts that matter;
per-symbol fan-out is the platform's job and lives in `evaluate/`, so an adapter
only ever sees one cell.
"""

from fintel.agents.base import (
    Abstained,
    Agent,
    AgentError,
    AgentTimeout,
    Channel,
    ContextOverflow,
    MalformedOutput,
    ProviderUnavailable,
    RateLimited,
    SafetyRefusal,
)
from fintel.agents.factory import AGENTS, build, names, register
from fintel.agents.run import classify, invoke
from fintel.agents.scripted import ConstantAgent, ScriptedAgent

__all__ = [
    "AGENTS",
    "Abstained",
    "Agent",
    "AgentError",
    "AgentTimeout",
    "Channel",
    "ConstantAgent",
    "ContextOverflow",
    "MalformedOutput",
    "ProviderUnavailable",
    "RateLimited",
    "SafetyRefusal",
    "ScriptedAgent",
    "build",
    "classify",
    "invoke",
    "names",
    "register",
]
