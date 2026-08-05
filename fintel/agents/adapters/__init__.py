"""Fintel agent adapters: CLI hosts and in-process hosts for external/native pipelines.

- Subprocess CLIs: OpenClaw, Claude Code
- In-process host: Optimized (wires ``installed.optimized_agent`` to fintel cells)
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
from fintel.agents.adapters.base import (
    ERROR_PATTERNS,
    CliFlag,
    ErrorPattern,
    SubprocessAgent,
    build_argv,
    classify_exit,
)
from fintel.agents.adapters.claude_code import ClaudeCodeAgent
from fintel.agents.adapters.openclaw import OpenClawAgent
from fintel.agents.adapters.optimized import OptimizedFintelAgent

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
