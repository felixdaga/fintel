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
from fintel.agents.factory import AGENTS, build, names, preflight, register
from fintel.agents.fingerprint import Fingerprint, fingerprint
from fintel.agents.installed.llm_agent import LLMAgent
from fintel.agents.llm import LLM, Completion, OpenRouter, ToolCall
from fintel.agents.pit_policy import (
    CLAUDE_CODE_DENY,
    FINTEL_MCP_SERVER,
    OPENCLAW_DENY,
    PitEnforcement,
)
from fintel.agents.run import classify, invoke
from fintel.agents.scripted import ConstantAgent, ScriptedAgent

__all__ = [
    "AGENTS",
    "CLAUDE_CODE_DENY",
    "FINTEL_MCP_SERVER",
    "Fingerprint",
    "LLM",
    "OPENCLAW_DENY",
    "Abstained",
    "Agent",
    "AgentError",
    "AgentTimeout",
    "Channel",
    "Completion",
    "ConstantAgent",
    "ContextOverflow",
    "LLMAgent",
    "MalformedOutput",
    "OpenRouter",
    "PitEnforcement",
    "ProviderUnavailable",
    "RateLimited",
    "SafetyRefusal",
    "ScriptedAgent",
    "ToolCall",
    "build",
    "classify",
    "fingerprint",
    "invoke",
    "names",
    "preflight",
    "register",
]
