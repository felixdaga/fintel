"""Installed CLI agents: OpenClaw and Claude Code.

One host underneath (SubprocessAgent), one transport (the fintel MCP server),
two thin adapters that differ only in argv and config-file format. The
one-cell-per-process rule is structural: the MCP server is a stdio subprocess
that dies with the CLI, so a reused gateway cannot keep serving the first cell.
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
    "ProviderUnavailable",
    "RateLimited",
    "SafetyRefusal",
    "SubprocessAgent",
    "build_argv",
    "classify_exit",
]
