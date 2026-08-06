"""PIT tool policy: deny threat channels, keep agent ability, fail closed."""

from __future__ import annotations

import json

import pytest

from fintel import agents
from fintel.agents.pit_policy import (
    CLAUDE_CODE_DENY,
    FINTEL_MCP_SERVER,
    OPENCLAW_DENY,
    apply_openclaw_tools,
    isolate_fintel_mcp,
    merge_deny,
    missing_deny,
    openclaw_deny_ok,
    restore_mcp,
)


def test_merge_deny_is_additive_and_order_preserving():
    assert merge_deny(["group:web"], OPENCLAW_DENY)[0] == "group:web"
    assert set(OPENCLAW_DENY) <= set(merge_deny(["group:web"], OPENCLAW_DENY))


def test_missing_deny_lists_only_gaps():
    assert missing_deny(list(OPENCLAW_DENY), OPENCLAW_DENY) == []
    assert missing_deny(["group:web"], OPENCLAW_DENY) == [
        m for m in OPENCLAW_DENY if m != "group:web"
    ]


def test_apply_openclaw_tools_merges_without_wiping_existing():
    data = {"tools": {"deny": ["custom_tool"], "allow": ["sessions_spawn"]}}
    apply_openclaw_tools(data)
    assert "custom_tool" in data["tools"]["deny"]
    assert set(OPENCLAW_DENY) <= set(data["tools"]["deny"])
    assert data["tools"]["allow"] == ["sessions_spawn"]  # ability untouched


def test_isolate_fintel_mcp_stashes_others_and_restore_puts_them_back():
    servers = {
        "delorean": {"command": "old"},
        "github": {"command": "gh"},
        FINTEL_MCP_SERVER: {"command": "stale"},
    }
    stashed = isolate_fintel_mcp(servers, {"command": "new-fintel"})
    assert list(servers) == [FINTEL_MCP_SERVER]
    assert servers[FINTEL_MCP_SERVER]["command"] == "new-fintel"
    assert set(stashed) == {"delorean", "github"}
    restore_mcp(servers, stashed)
    assert servers["delorean"]["command"] == "old"
    assert servers["github"]["command"] == "gh"
    assert servers[FINTEL_MCP_SERVER]["command"] == "new-fintel"


def test_openclaw_deny_does_not_include_subagent_tools():
    """Sub-agents are ability, not egress — must stay available."""
    for kept in ("sessions_spawn", "sessions_yield", "session_status", "update_plan"):
        assert kept not in OPENCLAW_DENY


def test_claude_deny_does_not_include_subagent_tools():
    for kept in ("Agent", "Task"):
        assert kept not in CLAUDE_CODE_DENY


@pytest.mark.parametrize("name", sorted(agents.names()))
def test_every_registered_agent_declares_pit_enforcement(name):
    kwargs: dict = {}
    if name == "llm":
        kwargs = {
            "llm": type("L", (), {"model": "x", "complete": lambda self, *a, **k: None})(),
            "channel": "pack",
        }
    agent = agents.build(name, **kwargs)
    assert getattr(type(agent), "pit_enforcement", None) in ("access", "cli_deny")


def test_preflight_rejects_an_adapter_without_pit_enforcement():
    class Bare:
        name = "bare"
        version = "1"

        def decide(self, env):
            raise NotImplementedError

    import sys

    sys.modules["__pit_bare__"] = type(sys)("__pit_bare__")
    sys.modules["__pit_bare__"].Bare = Bare  # type: ignore[attr-defined]
    problems = agents.preflight("__pit_bare__:Bare")
    assert any("pit_enforcement" in p for p in problems)


def test_preflight_rejects_cli_deny_without_enforce_override():
    from fintel.agents.adapters.base import SubprocessAgent

    class Incomplete(SubprocessAgent):
        name = "incomplete"
        binary = "true"

        def build_command(self, env, mcp_server_cmd):
            return [self.binary]

    import sys

    sys.modules["__pit_incomplete__"] = type(sys)("__pit_incomplete__")
    sys.modules["__pit_incomplete__"].Incomplete = Incomplete  # type: ignore[attr-defined]
    problems = agents.preflight("__pit_incomplete__:Incomplete")
    assert any("enforce_pit_policy" in p for p in problems)


def test_openclaw_enforce_applies_deny_and_isolates_mcp(tmp_path, monkeypatch):
    from fintel.agents.adapters.openclaw import OpenClawAgent
    from fintel.environment import Cell, RuntimeConfig, build_environment
    from fintel.market.factory import build_data_sources
    from fintel.market.settings import MarketConfig
    from fintel.models.market import DataBinding
    from tests import fixtures
    from tests.test_environment import DAY, UNIVERSE

    profile_dir = tmp_path / ".openclaw-testpit"
    profile_dir.mkdir()
    config_path = profile_dir / "openclaw.json"
    config_path.write_text(
        json.dumps(
            {
                "tools": {"deny": [], "allow": ["sessions_spawn"]},
                "mcp": {
                    "servers": {
                        "delorean": {"command": "old"},
                        "other": {"command": "x"},
                    }
                },
            }
        )
    )
    monkeypatch.setattr(OpenClawAgent, "_config_path", staticmethod(lambda profile: config_path))

    fixtures.register_all()
    env = build_environment(
        cell=Cell(run_id="t-r1", decision_date=DAY, symbols=("AAPL",)),
        sources=build_data_sources(
            [DataBinding(kind="prices", source="flat_prices")],
            config=MarketConfig(cache_root=tmp_path / "cache", offline=True),
        ),
        universe=list(UNIVERSE),
        kinds=("prices",),
        runtime=RuntimeConfig(session_root=tmp_path / "sessions"),
    )
    agent = OpenClawAgent(profile="testpit", repo_root=str(tmp_path))
    agent.enforce_pit_policy(env, ["python", "-m", "fintel.environment.mcp_server"])

    data = json.loads(config_path.read_text())
    assert set(OPENCLAW_DENY) <= set(data["tools"]["deny"])
    assert data["tools"]["allow"] == ["sessions_spawn"]
    assert list(data["mcp"]["servers"]) == [FINTEL_MCP_SERVER]
    assert data["mcp"]["servers"][FINTEL_MCP_SERVER]["env"]["FINTEL_SESSION_DIR"] == str(
        env.session.path
    )
    assert set(agent._mcp_stash) == {"delorean", "other"}

    agent.cleanup_cell(env)
    data = json.loads(config_path.read_text())
    assert "delorean" in data["mcp"]["servers"]
    assert "other" in data["mcp"]["servers"]
    assert FINTEL_MCP_SERVER in data["mcp"]["servers"]


def test_claude_build_command_passes_disallowed_tools(tmp_path):
    from fintel.agents.adapters.claude_code import ClaudeCodeAgent
    from fintel.environment import Cell, RuntimeConfig, build_environment
    from fintel.market.factory import build_data_sources
    from fintel.market.settings import MarketConfig
    from fintel.models.market import DataBinding
    from tests import fixtures
    from tests.test_environment import DAY, UNIVERSE

    fixtures.register_all()
    env = build_environment(
        cell=Cell(run_id="t-r1", decision_date=DAY, symbols=("AAPL",)),
        sources=build_data_sources(
            [DataBinding(kind="prices", source="flat_prices")],
            config=MarketConfig(cache_root=tmp_path / "cache", offline=True),
        ),
        universe=list(UNIVERSE),
        kinds=("prices",),
        runtime=RuntimeConfig(session_root=tmp_path / "sessions"),
    )
    agent = ClaudeCodeAgent()
    cmd = agent.build_command(env, ["python", "-m", "fintel.environment.mcp_server"])
    for tool in CLAUDE_CODE_DENY:
        assert tool in cmd
        assert cmd[cmd.index(tool) - 1] == "--disallowedTools"


def test_openclaw_deny_ok_reports_gaps():
    assert openclaw_deny_ok({"tools": {"deny": list(OPENCLAW_DENY)}}) == []
    gaps = openclaw_deny_ok({"tools": {"deny": []}})
    assert len(gaps) == len(OPENCLAW_DENY)
