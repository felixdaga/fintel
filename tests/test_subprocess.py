"""The subprocess host: spawn a CLI, read back the answer.

Uses a fake CLI script so the full path — writing bindings, spawning, collecting
the result, classifying failures — runs on every commit without OpenClaw or
Claude Code installed.
"""

from __future__ import annotations

import json
import stat
import textwrap

import pytest

from fintel import agents
from fintel.agents.fingerprint import fingerprint
from fintel.agents.installed import (
    AgentError,
    RateLimited,
    SafetyRefusal,
    SubprocessAgent,
    classify_exit,
)
from fintel.environment import Cell, RuntimeConfig, build_environment
from fintel.market.factory import build_data_sources
from fintel.market.settings import MarketConfig
from fintel.models.market import DataBinding
from tests.test_environment import DAY, UNIVERSE

KINDS = ("prices",)


def make_env(tmp_path, *, symbols=("AAPL",)):
    from tests import fixtures

    fixtures.register_all()
    bindings = [DataBinding(kind="prices", source="flat_prices")]
    sources = build_data_sources(
        bindings, config=MarketConfig(cache_root=tmp_path / "cache", offline=True)
    )
    return build_environment(
        cell=Cell(run_id="test-r1", decision_date=DAY, symbols=symbols),
        sources=sources,
        universe=list(UNIVERSE),
        kinds=KINDS,
        runtime=RuntimeConfig(session_root=tmp_path / "sessions"),
    )


def fake_cli(tmp_path, body: str) -> str:
    script = tmp_path / "fake_agent.sh"
    script.write_text("#!/bin/bash\n" + body)
    script.chmod(stat.S_IRWXU)
    return str(script)


class _Script(SubprocessAgent):
    """Runs a shell script. Used by every test below."""

    name = "fake"
    version = "1"

    def build_command(self, env, mcp_server_cmd):
        return [self.binary]


def test_a_cli_that_writes_a_result_is_collected(tmp_path):
    """The happy path: CLI exits 0 and wrote result.json with views."""
    script = fake_cli(
        tmp_path,
        textwrap.dedent("""\
        cat > "$FINTEL_SESSION_DIR/result.json" <<'JSON'
        {"views": [{"symbol": "AAPL", "score": 0.6, "rationale": "cheap"}]}
        JSON
        """),
    )
    response = agents.invoke(_Script(binary=script, timeout_s=10), make_env(tmp_path))
    assert response.outcome == "ok"
    assert "AAPL" in response.views
    assert response.views["AAPL"].score == pytest.approx(0.6)


def test_the_session_dir_carries_the_cell_identity(tmp_path):
    """The CLI reads cell.json from $FINTEL_SESSION_DIR, not by guessing."""
    script = fake_cli(
        tmp_path,
        'python3 -c "import json,os; d=os.environ[\'FINTEL_SESSION_DIR\']; '
        'c=json.load(open(d+\'/cell.json\')); print(c[\'symbols\'])" >&2',
    )
    agents.invoke(_Script(binary=script, timeout_s=10), make_env(tmp_path))
    # No result.json → empty, but the cell was readable.


def test_a_cli_that_exits_nonzero_is_classified(tmp_path):
    script = fake_cli(tmp_path, 'echo "rate limit exceeded" >&2; exit 1')
    response = agents.invoke(_Script(binary=script, timeout_s=10), make_env(tmp_path))
    assert response.outcome == "rate_limited"
    assert response.retryable


def test_a_safety_refusal_is_not_retried(tmp_path):
    script = fake_cli(tmp_path, 'echo "blocked by content filter" >&2; exit 1')
    response = agents.invoke(_Script(binary=script, timeout_s=10), make_env(tmp_path))
    assert response.outcome == "refused"
    assert not response.retryable


def test_a_context_overflow_is_a_config_bug(tmp_path):
    script = fake_cli(tmp_path, 'echo "context length exceeded" >&2; exit 1')
    response = agents.invoke(_Script(binary=script, timeout_s=10), make_env(tmp_path))
    assert response.outcome == "context_overflow"
    assert not response.retryable


def test_a_timeout_keeps_its_name(tmp_path):
    script = fake_cli(tmp_path, "sleep 5; exit 0")
    response = agents.invoke(_Script(binary=script, timeout_s=0.5), make_env(tmp_path))
    assert response.outcome == "timeout"


def test_a_cli_that_writes_no_result_is_empty(tmp_path):
    script = fake_cli(tmp_path, "exit 0")
    response = agents.invoke(_Script(binary=script, timeout_s=10), make_env(tmp_path))
    assert response.outcome == "empty"
    assert "no result.json" in response.detail


def test_an_abstention_is_recorded(tmp_path):
    script = fake_cli(
        tmp_path,
        textwrap.dedent("""\
        cat > "$FINTEL_SESSION_DIR/result.json" <<'JSON'
        {"views": [], "abstain": true, "abstain_reason": "no edge"}
        JSON
        """),
    )
    response = agents.invoke(_Script(binary=script, timeout_s=10), make_env(tmp_path))
    assert response.outcome == "abstained"
    assert response.detail == "no edge"
    assert not response.retryable


def test_bindings_are_written_for_the_mcp_server(tmp_path):
    """The MCP server rebuilds the environment from bindings.json."""
    env = make_env(tmp_path)
    agents.invoke(_Script(binary=fake_cli(tmp_path, "exit 0"), timeout_s=10), env)
    bindings = json.loads((env.session.path / "bindings.json").read_text())
    assert any(b["kind"] == "prices" for b in bindings["bindings"])
    assert "AAPL" in bindings["universe"]


def test_a_failure_is_recorded_not_raised(tmp_path):
    """One bad cell must not abort the date."""
    env = make_env(tmp_path)
    agents.invoke(_Script(binary=fake_cli(tmp_path, "exit 42"), timeout_s=10), env)
    events = [e["event"] for e in env.log.events]
    assert "agent_failed" in events


# ── error classification unit tests ──────────────────────────────────────────


def test_classify_exit_finds_the_last_match():
    exc = classify_exit("cmd", 1, "starting up...\nrate limit\n", "")
    assert isinstance(exc, RateLimited)


def test_classify_exit_prefers_the_later_match():
    exc = classify_exit("cmd", 1, "rate limit\nthen: blocked by safety filter\n", "")
    assert isinstance(exc, SafetyRefusal)


def test_classify_exit_unknown_is_a_crash():
    exc = classify_exit("cmd", 1, "something weird", "")
    assert isinstance(exc, AgentError)
    assert exc.outcome == "crashed"


# ── fingerprint ──────────────────────────────────────────────────────────────


def test_two_identical_runs_have_the_same_digest():
    a = fingerprint(agent_name="llm", agent_version="1", model="m", channel="pack",
                    prompt="score it", data_kinds=("prices",))
    b = fingerprint(agent_name="llm", agent_version="1", model="m", channel="pack",
                    prompt="score it", data_kinds=("prices",))
    assert a.digest == b.digest


def test_a_different_prompt_changes_the_digest():
    a = fingerprint(agent_name="llm", agent_version="1", model="m", channel="pack",
                    prompt="score it", data_kinds=("prices",))
    b = fingerprint(agent_name="llm", agent_version="1", model="m", channel="pack",
                    prompt="score it differently", data_kinds=("prices",))
    assert a.digest != b.digest


def test_a_different_model_changes_the_digest():
    a = fingerprint(agent_name="llm", agent_version="1", model="a", channel="pack",
                    prompt="x", data_kinds=("prices",))
    b = fingerprint(agent_name="llm", agent_version="1", model="b", channel="pack",
                    prompt="x", data_kinds=("prices",))
    assert a.digest != b.digest


def test_a_different_channel_changes_the_digest():
    """The channel is part of the fingerprint, so a channel ablation is a
    different run — not the same run that happened to use a different path."""
    a = fingerprint(agent_name="llm", agent_version="1", model="m", channel="pack",
                    prompt="x", data_kinds=("prices",))
    b = fingerprint(agent_name="llm", agent_version="1", model="m", channel="tools",
                    prompt="x", data_kinds=("prices",))
    assert a.digest != b.digest


def test_the_fingerprint_is_serializable():
    fp = fingerprint(agent_name="llm", agent_version="1", model="m", channel="pack",
                      prompt="x", data_kinds=("prices",), adapter_params={"max_rounds": 5})
    d = fp.to_dict()
    assert d["digest"] == fp.digest
    assert d["adapter_params"]["max_rounds"] == 5
    json.dumps(d)
