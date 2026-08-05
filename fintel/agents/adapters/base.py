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
from pathlib import Path
from typing import Any, ClassVar, Literal

from fintel.agents import prompts
from fintel.agents.base import (
    AgentError,
    AgentTimeout,
    ContextOverflow,
    ProviderUnavailable,
    RateLimited,
    SafetyRefusal,
)
from fintel.agents.emit import abstained, parse_views
from fintel.agents.pit_policy import PitEnforcement
from fintel.environment import Environment
from fintel.models.common import Outcome
from fintel.models.decision import AgentResponse
from fintel.models.trace import ReasoningTrace, TraceStep

logger = logging.getLogger(__name__)

MCP_SERVER_MODULE = "fintel.environment.mcp_server"
BINDINGS_FILE = "bindings.json"

# Kill the subprocess after this many consecutive tool errors — don't burn the
# full 600s timeout when every tool call is already dead.
FAIL_FAST_ERRORS = 3

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
    """Base for CLI agents. Subclasses declare `CLI_FLAGS` and `build_command`.

    PIT enforcement is mandatory for this host (`pit_enforcement = "cli_deny"`):
    every cell must apply the platform's deny list and isolate the fintel MCP
    server before the CLI starts, then restore the operator's profile afterward.
    Subclasses implement that in `enforce_pit_policy` / `cleanup_cell`.
    """

    binary: str = ""
    timeout_s: float = 600.0
    name: str = "subprocess"
    version: str = "1"
    extra_env: dict[str, str] = field(default_factory=dict)
    # The strategy pack's mission.md / output_schema.json, wired in by
    # `simulate.cell.build_agent`. Folded into the single `--message`/stdin
    # instruction every CLI agent takes, alongside the tools manual — a CLI
    # agent has no separate system/user split to put them in.
    mission_text: str = ""
    output_schema_text: str = ""
    CLI_FLAGS: ClassVar[list[CliFlag]] = []
    # A CLI that reads its MCP servers from a fixed config file (Claude Code's
    # `~/.claude.json`) declares this; the default `materialize_mcp` below then
    # merge-writes `mcp_config()` into it. OpenClaw's config path is per-profile
    # (computed from `self.profile`), so it overrides `materialize_mcp` directly
    # instead of using this.
    CONFIG_PATH: ClassVar[Path | None] = None
    # Standard adapter requirement: CLI hosts strip PIT-threat native tools.
    pit_enforcement: ClassVar[PitEnforcement] = "cli_deny"

    def decide(self, env: Environment) -> AgentResponse:
        assert env.session is not None, "subprocess agents require a session directory"
        self._write_bindings(env)
        started = time.perf_counter()

        # Generate the per-cell session id BEFORE building the catcher, so the
        # catcher's transcript path matches the file the CLI will actually write.
        # (OpenClaw writes `fintel-<id>.jsonl`; the id used to be generated
        # inside `build_command`, which runs only after the catcher was already
        # pointed at a `.../sessions/.jsonl` path with an empty id — so the
        # catcher never found the transcript and emitted no staging events.)
        self._prepare_session(env)

        from fintel.agents.adapters.catcher import _TranscriptCatcher

        catcher = _TranscriptCatcher(
            transcript_path=self.session_transcript_path(env),
            access_log=env.log,
            fail_fast_errors=FAIL_FAST_ERRORS,
            nerve=env.nerve,
            cell=env.cell.name,
            decision_date=env.cell.decision_date.isoformat(),
        )
        catcher.start()

        try:
            command, child_env = self._launch(env)
            try:
                result = subprocess.run(
                    command,
                    env=child_env,
                    input=self.stdin_input(env),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    # Run the agent CLI in its own session/process group so a crash
                    # inside it (e.g. openclaw signalling its pgroup on exit) can't
                    # take the fintel parent with it — the cause of the silent
                    # "jobs die right after cell_start" failures.
                    start_new_session=True,
                )
            except subprocess.TimeoutExpired:
                elapsed = (time.perf_counter() - started) * 1000
                catcher.stop()
                env.log.append(
                    "agent_timeout", command=command[0], elapsed_ms=round(elapsed, 1)
                )
                raise AgentTimeout(f"{self.name} timed out after {self.timeout_s}s") from None

            catcher.stop()
            elapsed = (time.perf_counter() - started) * 1000
            env.log.append(
                "subprocess_done",
                returncode=result.returncode,
                elapsed_ms=round(elapsed, 1),
                stdout_tail=result.stdout[-800:] if result.stdout else "",
                stderr_tail=result.stderr[-400:] if result.stderr else "",
                tool_calls=len(catcher.tool_calls),
                tool_errors=len(catcher.tool_errors),
            )

            # Capture the transcript into the cell dir and log tool errors.
            catcher.finalize(env.session.path)

            if result.returncode != 0:
                raise classify_exit(
                    " ".join(map(shlex.quote, command)),
                    result.returncode,
                    result.stdout,
                    result.stderr,
                )

            return self._collect(env, result, elapsed)
        finally:
            catcher.stop()
            # Always restore the operator's profile (stashed MCP servers, …),
            # even when the CLI crashes or times out.
            self.cleanup_cell(env)

    def session_transcript_path(self, env: Environment) -> Path | None:
        """Where the CLI writes its own session transcript. Override per CLI.

        OpenClaw writes ``fintel-<session_id>.jsonl`` under
        ``~/.openclaw-<profile>/agents/<agent_id>/sessions/``. If a CLI doesn't
        expose a transcript, returning None disables live catching.
        """
        return None

    def _prepare_session(self, env: Environment) -> None:
        """Generate any per-cell session identity the catcher needs BEFORE it
        computes the transcript path. Default: nothing. OpenClaw overrides to
        mint its ``fintel-<id>`` session id here (not inside ``build_command``)
        so the catcher tails the file the CLI will actually write."""
        return None

    def build_command(self, env: Environment, mcp_server_cmd: list[str]) -> list[str]:
        """Build the full argv. Override per CLI."""
        raise NotImplementedError

    def mcp_config(self, env: Environment, mcp_server_cmd: list[str]) -> dict | None:
        """MCP server config in the CLI's native format, or None if the CLI
        discovers the server from argv alone."""
        return None

    def stdin_input(self, env: Environment) -> str | None:
        """Text to pipe to the CLI's stdin, or None to send nothing. Override
        for a CLI that takes its instruction that way (Claude Code) rather than
        as a flag (OpenClaw's `--message`)."""
        return None

    @classmethod
    def preflight_checks(cls, **params: Any) -> list[str]:
        """Whether the CLI this adapter drives is even findable, without
        running it. `binary` may come from adapter params (an explicit
        override) or the class default (e.g. `openclaw`, `claude`)."""
        import shutil

        binary = params.get("binary") or cls.binary
        if binary and shutil.which(binary) is None:
            return [f"{binary!r} not found on PATH"]
        return []

    # ── internals ────────────────────────────────────────────────────────────

    def _compose_instruction(self, env: Environment, *, submit_tool: str = "submit_views") -> str:
        """The one message every CLI agent takes: mission, tools manual, output
        schema. Always includes the tools manual — a subprocess agent talks to
        its tools over MCP, so it is tool-calling by construction."""
        symbols = tuple(sorted(env.policy.decidable))
        return prompts.compose_instruction(
            mission=self.mission_text,
            decision_date=env.cell.decision_date.isoformat(),
            symbols=symbols,
            tools_manual=prompts.render_tools(env.tools.descriptors()),
            output_schema=self.output_schema_text or None,
            submit_tool=submit_tool,
        )

    def _write_bindings(self, env: Environment) -> None:
        """Store enough for the MCP server to rebuild the environment.

        Includes MarketConfig (cache_root, offline) so the subprocess hits the
        same cache the orchestrator populated. API keys are not written — the
        OpenClaw/Claude MCP env block carries them instead (session dirs must
        not hold secrets).
        """
        assert env.session is not None
        bindings_path = env.session.path / BINDINGS_FILE

        bindings = []
        for kind, source in env.tools.bound.items():
            bindings.append({"kind": kind, "source": source})
        payload: dict[str, Any] = {
            "bindings": bindings,
            "kinds": list(env.kinds),
            "universe": list(env.policy.decidable | env.policy.peers),
            "peers": bool(env.policy.peers),
        }
        if env.market_config is None:
            raise AgentError(
                "subprocess agents require env.market_config so the MCP server "
                "can rebuild against the same cache; pass market_config into "
                "build_environment"
            )
        payload["config"] = env.market_config.to_dict(secrets=False)
        bindings_path.write_text(json.dumps(payload, indent=2))

    def _mcp_server_cmd(self) -> list[str]:
        """The command that launches the fintel MCP server as a stdio subprocess."""
        return [self._python(), "-m", MCP_SERVER_MODULE]

    @staticmethod
    def _python() -> str:
        return sys.executable

    def materialize_mcp(self, env: Environment, mcp_server_cmd: list[str]) -> None:
        """Persist the MCP server config so the CLI can discover it.

        Default: merge `mcp_config()` into `CONFIG_PATH` if the subclass
        declared one (Claude Code); nothing if neither is set (a CLI that
        discovers its MCP server from argv alone). OpenClaw's config path is
        per-profile, so it overrides this method directly instead.
        """
        config = self.mcp_config(env, mcp_server_cmd)
        if config is None or self.CONFIG_PATH is None:
            return
        self._merge_write(self.CONFIG_PATH, config)

    def enforce_pit_policy(self, env: Environment, mcp_server_cmd: list[str]) -> None:
        """Strip PIT-threat native tools and isolate the fintel MCP server.

        Required for every `cli_deny` adapter. Default raises — a subclass that
        forgets to implement this cannot silently run with web/fs still on.
        """
        raise NotImplementedError(
            f"{type(self).__name__} declares pit_enforcement='cli_deny' but "
            "does not implement enforce_pit_policy"
        )

    def cleanup_cell(self, env: Environment) -> None:
        """Undo per-cell profile mutations (restore stashed MCP servers, …).

        Default: nothing. OpenClaw/Claude Code override to put the operator's
        other MCP servers back after the cell.
        """
        return None

    @staticmethod
    def _merge_write(path: Path, patch: dict) -> None:
        """One level of dict merge — enough to add our server under
        `mcpServers` without clobbering any the operator already configured."""
        try:
            data = json.loads(path.read_text()) if path.is_file() else {}
        except json.JSONDecodeError:
            data = {}
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(data.get(key), dict):
                data[key].update(value)
            else:
                data[key] = value
        path.parent.mkdir(parents=True, exist_ok=True)
        # Unique temp name: parallel runs/cells mutate the same shared config
        # concurrently; a fixed `.tmp` name collides. Pid + uuid keeps each
        # write's temp file distinct.
        import os
        import uuid

        tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)

    def _launch(self, env: Environment) -> tuple[list[str], dict[str, str]]:
        assert env.session is not None
        mcp_cmd = self._mcp_server_cmd()
        # PIT first: deny threat tools + isolate fintel MCP. Then any remaining
        # materialize_mcp work (Claude Code's CONFIG_PATH merge is folded into
        # enforce_pit_policy for that adapter).
        self.enforce_pit_policy(env, mcp_cmd)
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

        # Explicit abstain wins even if the model also stuffed a neutral score —
        # otherwise a "I couldn't get data" response reads as ok/0.0.
        if reason:
            return AgentResponse(views={}, outcome="abstained", detail=reason, trace=trace)
        outcome: Outcome = "ok" if views else "empty"
        detail = "; ".join(notes) if not views else ""
        return AgentResponse(views=views, outcome=outcome, detail=detail, trace=trace)
