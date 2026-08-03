"""Installed agents: CLI hosts (OpenClaw, Claude Code) and in-process desks
that ship as named adapters (Optimized).
"""

from fintel.agents.base import (
    AgentError,
    AgentTimeout,
    ContextOverflow,
    MalformedOutput,
    ProviderUnavailable,
    RateLimited,
    SafetyRefusal,
)
from fintel.agents.installed.base import (
    ERROR_PATTERNS,
    CliFlag,
    ErrorPattern,
    SubprocessAgent,
    build_argv,
    classify_exit,
)
from fintel.agents.installed.claude_code import ClaudeCodeAgent
from fintel.agents.installed.openclaw import OpenClawAgent
from fintel.agents.installed.optimized import OptimizedFintelAgent

__all__ = [
    "ERROR_PATTERNS",
    "AgentError",
    "AgentTimeout",
    "ClaudeCodeAgent",
    "CliFlag",
    "ContextOverflow",
    "ErrorPattern",
    "MalformedOutput",
    "OpenClawAgent",
    "OptimizedFintelAgent",
    "ProviderUnavailable",
    "RateLimited",
    "SafetyRefusal",
    "SubprocessAgent",
    "build_argv",
    "classify_exit",
]
