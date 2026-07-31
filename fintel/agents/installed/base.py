"""The subprocess host: spawn a CLI, hand it the MCP server, read back the answer.

A CLI agent (OpenClaw, Claude Code) is a black box that talks to its tools over
MCP and writes an answer somewhere. This base handles the parts that are the same
for every CLI:

  * write `bindings.json` so the MCP server can rebuild the environment;
  * spawn the CLI with the MCP server as a stdio subprocess and the session dir
    in the environment;
  * wait for it to finish, classifying a non-zero exit into a typed outcome;
  * read `result.json` if the agent wrote one, or record an empty/timeout.

What varies per CLI is the argv, the env, and the config-file format for MCP —
all declared, not hand-rolled, via `CLI_FLAGS`, `ENV_VARS` and `mcp_config`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from fintel.agents.base import (
    AgentError,
    AgentTimeout,
    ContextOverflow,
    ProviderUnavailable,
    RateLimited,
    SafetyRefusal,
)
from fintel.agents.emit import abstained, parse_views
from fintel.environment import Environment
from fintel.models.common import Outcome
from fintel.models.decision import AgentResponse
from fintel.models.trace import ReasoningTrace, TraceStep

logger = logging.getLogger(__name__)

MCP_SERVER_MODULE = "fintel.environment.mcp_server"
BINDINGS_FILE = "bindings.json"

# Regexes over stdout+stderr, checked in order; the last match wins (the real
# failure is usually at the end, after the banner). Modeled on Harbor's
# ErrorPatterns, which is the one part of its adapter base worth copying.
ERROR_PATTERNS: tuple[tuple[str, type[AgentError]], ...] = (
    (r"rate.?limit|too many requests", RateLimited),
    (r"quota exceeded|usage limit", RateLimited),
    (r"overloaded|temporarily unavailable", ProviderUnavailable),
    (r"500 internal|502|503|504", ProviderUnavailable),
    (r"context (length|window)|too many tokens|prompt is too long", ContextOverflow),
    (r"content.?filter|content.?policy|safety|blocked by", SafetyRefusal),
    (r"not logged in|unauthorized|api key", AgentError),
)


@dataclass
class CliFlag:
    kwarg: str
    cli: str
    type: Literal["str", "int", "bool"] = "str"
    default: Any = None
    format: str | None = None


@dataclass
class ErrorPattern:
    pattern: str
    exception: type[AgentError]


def classify_exit(command: str, returncode: int, stdout: str, stderr: str) -> AgentError:
    output = f"{stdout or ''}\n{stderr or ''}"
    last: tuple[int, type[AgentError]] | None = None
    for compiled, exc in ((re.compile(p, re.IGNORECASE), e) for p, e in ERROR_PATTERNS):
        for m in compiled.finditer(output):
            if last is None or m.end() > last[0]:
                last = (m.end(), exc)
    detail = f"command exited {returncode}: {command}\n{output[-800:]}"
    if last is not None:
        return last[1](detail)
    return AgentError(detail)


def build_argv(flags: list[CliFlag], kwargs: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for flag in flags:
        value = kwargs.get(flag.kwarg, flag.default)
        if value is None:
            continue
        if flag.type == "bool":
            if value:
                out.append(flag.cli)
        elif flag.format:
            out.append(flag.format.format(value=value))
        else:
            out.extend([flag.cli, str(value)])
    return out


@dataclass
class SubprocessAgent:
    """Base for CLI agents. Subclasses declare `CLI_FLAGS` and `build_command`."""

    binary: str = ""
    timeout_s: float = 600.0
    name: str = "subprocess"
    version: str = "1"
    extra_env: dict[str, str] = field(default_factory=dict)
    CLI_FLAGS: ClassVar[list[CliFlag]] = []

    def decide(self, env: Environment) -> AgentResponse:
        assert env.session is not None, "subprocess agents require a session directory"
        self._write_bindings(env)
        started = time.perf_counter()
        command, child_env = self._launch(env)
        try:
            result = subprocess.run(
                command,
                env=child_env,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired:
            elapsed = (time.perf_counter() - started) * 1000
            env.log.append("agent_timeout", command=command[0], elapsed_ms=round(elapsed, 1))
            raise AgentTimeout(f"{self.name} timed out after {self.timeout_s}s") from None

        elapsed = (time.perf_counter() - started) * 1000
        env.log.append(
            "subprocess_done",
            returncode=result.returncode,
            elapsed_ms=round(elapsed, 1),
            stdout_tail=result.stdout[-400:] if result.stdout else "",
            stderr_tail=result.stderr[-400:] if result.stderr else "",
        )

        if result.returncode != 0:
            raise classify_exit(" ".join(map(shlex.quote, command)), result.returncode,
                                result.stdout, result.stderr)

        return self._collect(env, result, elapsed)

    def build_command(self, env: Environment, mcp_server_cmd: list[str]) -> list[str]:
        """Build the full argv. Override per CLI."""
        raise NotImplementedError

    def mcp_config(self, env: Environment, mcp_server_cmd: list[str]) -> dict | None:
        """MCP server config in the CLI's native format, or None if the CLI
        discovers the server from argv alone."""
        return None

    # ── internals ────────────────────────────────────────────────────────────

    def _write_bindings(self, env: Environment) -> None:
        """Store enough for the MCP server to rebuild the environment."""
        assert env.session is not None
        bindings_path = env.session.path / BINDINGS_FILE

        bindings = []
        for kind, source in env.tools.bound.items():
            bindings.append({"kind": kind, "source": source})
        bindings_path.write_text(
            json.dumps(
                {
                    "bindings": bindings,
                    "kinds": list(env.kinds),
                    "universe": list(env.policy.decidable | env.policy.peers),
                    "peers": bool(env.policy.peers),
                },
                indent=2,
            )
        )

    def _mcp_server_cmd(self) -> list[str]:
        """The command that launches the fintel MCP server as a stdio subprocess."""
        return [self._python(), "-m", MCP_SERVER_MODULE]

    @staticmethod
    def _python() -> str:
        return sys.executable

    def _launch(self, env: Environment) -> tuple[list[str], dict[str, str]]:
        assert env.session is not None
        mcp_cmd = self._mcp_server_cmd()
        command = self.build_command(env, mcp_cmd)
        child_env = {
            **os.environ,
            **env.session.env(),
            **self.extra_env,
        }
        return command, child_env

    def _collect(
        self, env: Environment, result: subprocess.CompletedProcess, elapsed_ms: float
    ) -> AgentResponse:
        assert env.session is not None
        result_path = env.session.result
        step = TraceStep(
            step_id=f"{self.name}-{int(elapsed_ms)}",
            kind="subprocess",
            started_at=__import__("datetime").datetime.now(),
            duration_ms=int(elapsed_ms),
            model=self.name,
            payload={"returncode": result.returncode, "stdout_tail": result.stdout[-400:]},
        )
        trace = ReasoningTrace(steps=[step], metadata={"agent": self.name})

        if not result_path.is_file():
            return AgentResponse(
                views={}, outcome="empty",
                detail=f"{self.name} exited 0 but wrote no result.json",
                trace=trace,
            )

        payload = json.loads(result_path.read_text())
        reason = abstained(payload)
        views, notes = parse_views(payload, decidable=env.policy.decidable)
        if notes:
            trace.metadata["coercions"] = notes

        if reason and not views:
            return AgentResponse(views={}, outcome="abstained", detail=reason, trace=trace)
        outcome: Outcome = "ok" if views else "empty"
        detail = "; ".join(notes) if not views else ""
        return AgentResponse(views=views, outcome=outcome, detail=detail, trace=trace)
