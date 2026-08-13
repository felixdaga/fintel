"""OpenClaw adapter.

OpenClaw is a local agent harness that reads its MCP servers from the profile
config and spawns them as stdio subprocesses. The fintel MCP server is one of
those, so every tool the agent calls goes through the one clamped, recorded path.

Per cell this adapter:

  * merges the platform PIT deny list into ``tools.deny`` (web/fs/runtime/memory
    off; sub-agents and plan tools stay — those are ability, not egress);
  * isolates ``mcp.servers`` to only ``fintel``, stashing the operator's other
    servers and restoring them when the cell ends;
  * points the fintel server at this cell's session dir (openclaw does not
    inherit the parent env, so ``FINTEL_SESSION_DIR`` has to live in the
    server's env block).

The model/billing is the profile's concern — the adapter never sets a model.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fintel.agents.adapters.base import SubprocessAgent
from fintel.agents.pit_policy import (
    FINTEL_MCP_SERVER,
    OPENCLAW_DENY,
    apply_openclaw_tools,
    isolate_fintel_mcp,
    restore_mcp,
)
from fintel.environment import Environment


def _free_port() -> int:
    """Grab an ephemeral free TCP port for an isolated openclaw gateway."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class OpenClawAgent(SubprocessAgent):
    """Runs ``openclaw --profile <p> agent --json --agent <id>`` with the fintel MCP server."""

    profile: str = "delorean"
    agent_id: str = "main"
    model: str = ""  # blank = use the profile's configured primary
    thinking: str = ""  # e.g. "high"; blank = profile default
    binary: str = "openclaw"
    name: str = "openclaw"
    version: str = "1"
    timeout_s: float = 600.0
    massive_api_key: str = ""
    brave_api_key: str = ""
    fred_api_key: str = ""
    repo_root: str = ""
    # Accepted so build_agent can pass pack company_names.json without a
    # TypeError fallback that would drop mission_text / output_schema_text.
    # OpenClaw does not use display names in-process (the model sees symbols
    # via the instruction); stored for parity with llm/optimized adapters.
    company_names: dict[str, str] = field(default_factory=dict)
    _session_id: str = field(default="", init=False, repr=False)
    _mcp_stash: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    # Per-cell profile isolation. The operator's profile (e.g. "delorean") is
    # never mutated by a run: each cell forks the operator's config into a
    # throwaway ``fintel-<session_id>`` profile on a UNIQUE gateway port, so
    # openclaw starts its own gateway + fintel mcp server for that cell. This
    # is what makes parallel openclaw runs safe — they cannot share a gateway
    # that would serve a stale fintel mcp server pointed at another cell's
    # session dir (the cause of the intermittent "0 reads" failures).
    _operator_profile: str = field(default="", init=False, repr=False)
    _isolated_profile: str | None = field(default=None, init=False, repr=False)
    _isolated_port: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.repo_root:
            self.repo_root = str(Path(__file__).resolve().parents[3])
        if not self.massive_api_key:
            self.massive_api_key = os.environ.get("MASSIVE_API_KEY", "")
        if not self.brave_api_key:
            self.brave_api_key = os.environ.get("BRAVE_API_KEY", "")
        if not self.fred_api_key:
            self.fred_api_key = os.environ.get("FRED_API_KEY", "")
        port = self._read_gateway_port()
        if port:
            self.extra_env = {**self.extra_env, "OPENCLAW_GATEWAY_PORT": str(port)}

    def session_transcript_path(self, env: Environment) -> Path | None:
        """``fintel-<session_id>.jsonl`` under the profile's sessions dir."""
        base = Path.home() / f".openclaw-{self.profile}" / "agents" / self.agent_id / "sessions"
        return base / f"{self._session_id}.jsonl"

    def _prepare_session(self, env: Environment) -> None:
        # Minted here (not in build_command) so the catcher — constructed
        # before build_command runs — tails the file the CLI will write.
        if not self._session_id:
            self._session_id = f"fintel-{uuid.uuid4().hex[:12]}"
        # Fork the operator's profile into a throwaway per-cell profile on a
        # unique gateway port. openclaw then starts its own gateway (no shared
        # gateway to serve a stale fintel mcp server from another cell), and
        # the operator's openclaw.json is never mutated — parallel-safe.
        if self._isolated_profile is None:
            self._operator_profile = self.profile
            self._isolated_profile = f"fintel-{self._session_id}"
            self._isolated_port = _free_port()
            self._fork_profile(self._operator_profile, self._isolated_profile, self._isolated_port)
            self.profile = self._isolated_profile
            self.extra_env = {**self.extra_env, "OPENCLAW_GATEWAY_PORT": str(self._isolated_port)}

    def build_command(self, env: Environment, mcp_server_cmd: list[str]) -> list[str]:
        args = [
            self.binary,
            "--profile",
            self.profile,
            "agent",
            "--json",
            "--agent",
            self.agent_id,
            "--session-id",
            self._session_id,
            "--message",
            self._instruction(env),
        ]
        if self.model:
            args += ["--model", self.model]
        if self.thinking:
            args += ["--thinking", self.thinking]
        return args

    def enforce_pit_policy(self, env: Environment, mcp_server_cmd: list[str]) -> None:
        """Apply PIT denies + isolate fintel MCP on the profile for this cell."""
        assert env.session is not None
        config_path = self._config_path(self.profile)
        data = self._load(config_path)
        apply_openclaw_tools(data)

        servers = data.setdefault("mcp", {}).setdefault("servers", {})
        if not isinstance(servers, dict):
            servers = {}
            data.setdefault("mcp", {})["servers"] = servers

        server_env = {"FINTEL_SESSION_DIR": str(env.session.path)}
        if self.massive_api_key:
            server_env["MASSIVE_API_KEY"] = self.massive_api_key
        if self.brave_api_key:
            server_env["BRAVE_API_KEY"] = self.brave_api_key
        if self.fred_api_key:
            server_env["FRED_API_KEY"] = self.fred_api_key
        fintel_entry = {
            "command": mcp_server_cmd[0],
            "args": mcp_server_cmd[1:],
            "env": server_env,
            "cwd": self.repo_root,
        }
        self._mcp_stash = isolate_fintel_mcp(servers, fintel_entry)
        self._atomic_write(config_path, data)
        env.log.append(
            "pit_policy_applied",
            adapter=self.name,
            deny=list(OPENCLAW_DENY),
            stashed_mcp=sorted(self._mcp_stash),
            mcp_servers=[FINTEL_MCP_SERVER],
        )

    def cleanup_cell(self, env: Environment) -> None:
        """Drop the throwaway per-cell profile so the operator's config is
        untouched. The operator's profile was never mutated, so there is nothing
        to restore there."""
        if self._mcp_stash:
            config_path = self._config_path(self.profile)
            data = self._load(config_path)
            servers = data.setdefault("mcp", {}).setdefault("servers", {})
            if isinstance(servers, dict):
                restore_mcp(servers, self._mcp_stash)
                self._atomic_write(config_path, data)
            self._mcp_stash = {}
        if self._isolated_profile is not None:
            iso_dir = Path.home() / f".openclaw-{self._isolated_profile}"
            shutil.rmtree(iso_dir, ignore_errors=True)
            self._isolated_profile = None
            self._isolated_port = None
            if self._operator_profile:
                self.profile = self._operator_profile
                self._operator_profile = ""

    def materialize_mcp(self, env: Environment, mcp_server_cmd: list[str]) -> None:
        """Folded into ``enforce_pit_policy`` — kept so the base contract stays clear."""
        return None

    def _instruction(self, env: Environment) -> str:
        return self._compose_instruction(env)

    @classmethod
    def preflight_checks(cls, **params: Any) -> list[str]:
        problems = super().preflight_checks(**params)
        profile = params.get("profile", cls.profile)
        config_path = cls._config_path(profile)
        if not config_path.is_file():
            problems.append(f"openclaw profile {profile!r} has no config at {config_path}")
            return problems
        # Deny list is applied at launch; here we only need the profile to be
        # writable so enforce_pit_policy can merge. Unwritable → fail closed.
        if not os.access(config_path, os.W_OK):
            problems.append(f"openclaw profile config is not writable: {config_path}")
        # Surface any *current* gaps as warnings-as-problems only if the file
        # can't be patched — otherwise launch will merge them in. We still
        # require the tools block to be a dict if present.
        data = cls._load(config_path)
        tools = data.get("tools")
        if tools is not None and not isinstance(tools, dict):
            problems.append(f"openclaw tools block must be an object, got {type(tools).__name__}")
        return problems

    @staticmethod
    def _config_path(profile: str) -> Path:
        return Path.home() / f".openclaw-{profile}" / "openclaw.json"

    def _fork_profile(self, operator: str, isolated: str, gateway_port: int) -> None:
        """Copy just enough of the operator's profile for a cell to run under
        ``--profile <isolated>``: ``openclaw.json`` (mcp/gateway/model) and the
        agent's model/auth config (``agents/<id>/agent/``). The gateway port is
        rewritten to ``gateway_port`` so the isolated profile starts its own
        gateway — the operator's shared gateway is never contacted. The heavy
        ``sessions/`` history is skipped; openclaw writes a fresh transcript."""
        src = Path.home() / f".openclaw-{operator}"
        dst = Path.home() / f".openclaw-{isolated}"
        dst.mkdir(parents=True, exist_ok=True)
        cfg = src / "openclaw.json"
        if cfg.is_file():
            data = self._load(cfg)
            gw = data.setdefault("gateway", {})
            if isinstance(gw, dict):
                gw["port"] = gateway_port
            self._atomic_write(dst / "openclaw.json", data)
        agent_cfg = src / "agents" / self.agent_id / "agent"
        if agent_cfg.is_dir():
            dst_agent = dst / "agents" / self.agent_id / "agent"
            dst_agent.mkdir(parents=True, exist_ok=True)
            for f in agent_cfg.iterdir():
                if f.is_file():
                    shutil.copy2(f, dst_agent / f.name)

    def _read_gateway_port(self) -> int | None:
        data = self._load(self._config_path(self.profile))
        port = (data.get("gateway") or {}).get("port")
        return int(port) if port else None

    @staticmethod
    def _load(path: Path) -> dict:
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _atomic_write(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Unique temp name: two parallel runs mutate the same shared profile
        # (`~/.openclaw-<profile>/openclaw.json`) concurrently, and a fixed
        # `.json.tmp` name collides — one run's `replace` consumes the other's
        # tmp, raising FileNotFoundError. Pid + uuid makes each write's tmp
        # distinct, so they never race on the temp file.
        import os
        import uuid

        tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)
