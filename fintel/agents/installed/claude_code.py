"""Claude Code adapter.

Claude Code is a CLI that reads MCP server config from `~/.claude.json` and takes
its instruction via stdin. Same host as OpenClaw; different config format and
delivery channel.

Per cell this adapter:

  * passes ``--disallowedTools`` for every PIT-threat native tool (web, bash,
    free fs) — session-scoped, so the operator's daily Claude is not permanently
    crippled;
  * isolates ``mcpServers`` to only ``fintel``, stashing others and restoring
    them when the cell ends.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fintel.agents.installed.base import CliFlag, SubprocessAgent
from fintel.agents.pit_policy import (
    CLAUDE_CODE_DENY,
    FINTEL_MCP_SERVER,
    isolate_fintel_mcp,
    restore_mcp,
)


def _unique_tmp(path: Path) -> Path:
    """A temp path unique to this process + call, so two parallel runs mutating
    the same shared config (`~/.claude.json`) never collide on a fixed
    `.tmp` name (the openclaw profile race that crashed parallel runs)."""
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")


@dataclass
class ClaudeCodeAgent(SubprocessAgent):
    """Runs `claude --print --output-format=stream-json` with stdin instruction."""

    model: str = "claude-sonnet-4-20250514"
    binary: str = "claude"
    name: str = "claude-code"
    version: str = "1"
    timeout_s: float = 600.0
    max_turns: int = 20
    _mcp_stash: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    CLI_FLAGS = [
        CliFlag("max_turns", "--max-turns", "int"),
    ]
    CONFIG_PATH = Path.home() / ".claude.json"

    def build_command(self, env, mcp_server_cmd: list[str]) -> list[str]:
        args = [
            self.binary,
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            "--max-turns",
            str(self.max_turns),
        ]
        # Session-scoped PIT deny — does not permanently mutate the operator's
        # permissions.deny. One flag per tool (Claude Code's CLI form).
        for tool in CLAUDE_CODE_DENY:
            args += ["--disallowedTools", tool]
        return args

    def enforce_pit_policy(self, env, mcp_server_cmd: list[str]) -> None:
        """Isolate mcpServers to fintel only; native denies ride on argv."""
        assert env.session is not None
        path = self.CONFIG_PATH
        try:
            data = json.loads(path.read_text()) if path.is_file() else {}
        except json.JSONDecodeError:
            data = {}
        servers = data.setdefault("mcpServers", {})
        if not isinstance(servers, dict):
            servers = {}
            data["mcpServers"] = servers
        fintel_entry = {
            "type": "stdio",
            "command": mcp_server_cmd[0],
            "args": mcp_server_cmd[1:],
            "env": {"FINTEL_SESSION_DIR": str(env.session.path)},
        }
        self._mcp_stash = isolate_fintel_mcp(servers, fintel_entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = _unique_tmp(path)
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)
        env.log.append(
            "pit_policy_applied",
            adapter=self.name,
            deny=list(CLAUDE_CODE_DENY),
            stashed_mcp=sorted(self._mcp_stash),
            mcp_servers=[FINTEL_MCP_SERVER],
        )

    def cleanup_cell(self, env) -> None:
        if not self._mcp_stash:
            return
        path = self.CONFIG_PATH
        try:
            data = json.loads(path.read_text()) if path.is_file() else {}
        except json.JSONDecodeError:
            data = {}
        servers = data.setdefault("mcpServers", {})
        if not isinstance(servers, dict):
            return
        restore_mcp(servers, self._mcp_stash)
        tmp = _unique_tmp(path)
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)
        self._mcp_stash = {}

    def materialize_mcp(self, env, mcp_server_cmd: list[str]) -> None:
        """Folded into ``enforce_pit_policy``."""
        return None

    def mcp_config(self, env, mcp_server_cmd: list[str]) -> dict | None:
        return None

    def stdin_input(self, env) -> str:
        return self._compose_instruction(env)

    @classmethod
    def preflight_checks(cls, **params: Any) -> list[str]:
        problems = super().preflight_checks(**params)
        # Deny rides on argv; we only need the config path writable so we can
        # isolate mcpServers for the cell. Missing file is fine — we create it.
        import os

        path = cls.CONFIG_PATH
        if path.exists() and not os.access(path, os.W_OK):
            problems.append(f"claude config is not writable: {path}")
        return problems
