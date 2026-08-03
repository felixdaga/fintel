"""The agent contract, and the conformance suite every adapter must pass.

The platform's purpose is running one strategy against several agents and
believing the comparison. That only holds if every adapter is held to the same
terms, so the checks below are parametrised over the registry rather than
written per adapter — a new agent inherits them by existing.
"""

from __future__ import annotations

import pytest

from fintel import agents
from fintel.agents import (
    Abstained,
    AgentTimeout,
    ContextOverflow,
    MalformedOutput,
    RateLimited,
    SafetyRefusal,
)
from fintel.agents.base import AgentError
from fintel.environment import Cell
from fintel.models.common import RETRYABLE
from fintel.models.decision import AgentResponse, View
from fintel.models.trace import Usage, total
from tests.test_environment import DAY, an_environment


class AlwaysSubmits:
    """A model that answers immediately, so the LLM adapter can be held to the
    same contract as the others without a network or a key."""

    model = "fake/model"

    def complete(self, messages, *, tools=(), force_tool=None, max_tokens=None):
        from fintel.agents.emit import SUBMIT_TOOL
        from fintel.agents.llm import Completion, ToolCall

        payload = {"views": [{"symbol": "AAPL", "score": 0.2, "rationale": "fine"}]}
        return Completion(
            tool_calls=(ToolCall(id="c1", name=SUBMIT_TOOL, arguments=payload),),
            model=self.model,
            finish_reason="tool_calls",
        )


# Every adapter the factory can build, with kwargs cheap enough to construct.
# A new adapter added to the registry and not to this map fails the suite, which
# is the point: conformance is inherited by existing, not opted into.
CONFORMANT: dict[str, dict] = {
    "constant": {},
    "scripted": {},
    "llm": {"llm": AlwaysSubmits(), "channel": "pack"},
}

# CLI adapters (OpenClaw, Claude Code) can't join CONFORMANT: invoking them for
# real needs the actual binary and, for OpenClaw, patches a real
# `~/.openclaw-<profile>/openclaw.json` — not something a unit test should do.
# They get their own lighter, non-invoking checks below.
SUBPROCESS_ADAPTERS: frozenset[str] = frozenset({"openclaw", "claude-code"})


def envir(tmp_path, **kw):
    return an_environment(tmp_path, **kw)


# ── the contract ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(CONFORMANT))
def test_every_adapter_satisfies_the_protocol(name):
    agent = agents.build(name, **CONFORMANT[name])
    assert isinstance(agent, agents.Agent)
    assert agent.name and agent.version


@pytest.mark.parametrize("name", sorted(CONFORMANT))
def test_every_adapter_returns_a_response_it_does_not_submit(tmp_path, name):
    """A return value can't be forgotten the way a callback can."""
    agent = agents.build(name, **CONFORMANT[name])
    response = agents.invoke(agent, envir(tmp_path / name))
    assert isinstance(response, AgentResponse)


@pytest.mark.parametrize("name", sorted(CONFORMANT))
def test_every_adapter_decides_only_on_symbols_it_may_decide(tmp_path, name):
    env = envir(tmp_path / name)
    response = agents.invoke(agents.build(name, **CONFORMANT[name]), env)
    assert set(response.views) <= set(env.policy.decidable)


@pytest.mark.parametrize("name", sorted(CONFORMANT))
def test_every_adapter_reads_only_through_the_recorded_path(tmp_path, name):
    """An adapter that fetched its own data would leave no trace here, and its
    scores would not be comparable with one that didn't."""
    env = envir(tmp_path / name)
    agents.invoke(agents.build(name, **CONFORMANT[name]), env)
    for read in env.log.reads:
        assert read["kind"] in env.kinds


@pytest.mark.parametrize("name", sorted(CONFORMANT))
def test_no_adapter_can_claim_ok_with_nothing_to_show(tmp_path, name):
    response = agents.invoke(agents.build(name, **CONFORMANT[name]), envir(tmp_path / name))
    assert (response.outcome == "ok") == bool(response.views)


# ── outcomes are distinguishable ─────────────────────────────────────────────


def test_an_abstention_is_not_a_crash(tmp_path):
    """The distinction the old platform lost: both produced zero views."""
    agent = agents.build("scripted", raises=Abstained("nothing compelling"))
    response = agents.invoke(agent, envir(tmp_path))
    assert response.outcome == "abstained"
    assert response.views == {}
    assert "nothing compelling" in response.detail


def test_a_crash_is_not_an_abstention(tmp_path):
    agent = agents.build("scripted", raises=ZeroDivisionError("bug"))
    response = agents.invoke(agent, envir(tmp_path))
    assert response.outcome == "crashed"
    assert "ZeroDivisionError" in response.detail


@pytest.mark.parametrize(
    ("exc", "outcome"),
    [
        (AgentTimeout("slow"), "timeout"),
        (RateLimited("429"), "rate_limited"),
        (SafetyRefusal("blocked"), "refused"),
        (ContextOverflow("too long"), "context_overflow"),
        (MalformedOutput("not json"), "parse_error"),
        (Abstained("pass"), "abstained"),
        (TimeoutError("builtin"), "timeout"),
        (RuntimeError("who knows"), "crashed"),
    ],
)
def test_each_failure_keeps_its_own_name(tmp_path, exc, outcome):
    agent = agents.build("scripted", raises=exc)
    assert agents.invoke(agent, envir(tmp_path)).outcome == outcome


def test_a_refusal_is_never_retried(tmp_path):
    """Re-rolling a refusal doesn't recover an answer, it invents one."""
    for i, exc in enumerate((SafetyRefusal("blocked"), Abstained("pass"), ContextOverflow("big"))):
        env = envir(tmp_path / str(i))
        response = agents.invoke(agents.build("scripted", raises=exc), env)
        assert not response.retryable, response.outcome


def test_a_rate_limit_is_retried(tmp_path):
    response = agents.invoke(agents.build("scripted", raises=RateLimited("429")), envir(tmp_path))
    assert response.retryable


def test_the_retryable_set_and_the_outcomes_agree():
    """Guards against adding an outcome and forgetting to rule on retrying it."""
    from typing import get_args

    from fintel.models.common import Outcome

    assert RETRYABLE <= set(get_args(Outcome))


def test_a_failure_is_recorded_not_raised(tmp_path):
    """One bad cell must not abort the date and erase every other symbol."""
    env = envir(tmp_path)
    agents.invoke(agents.build("scripted", raises=RuntimeError("boom")), env)
    events = [e["event"] for e in env.log.events]
    assert "agent_failed" in events
    assert "agent_returned" in events


def test_an_interrupt_still_stops_the_run(tmp_path):
    env = envir(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        agents.invoke(agents.build("scripted", raises=KeyboardInterrupt()), env)
    # ...but the cell's failure is on record first.
    assert "agent_failed" in [e["event"] for e in env.log.events]


# ── scope enforcement ────────────────────────────────────────────────────────


def test_a_view_outside_the_cell_is_dropped(tmp_path):
    """Otherwise a strategy scores an opinion it never asked for."""

    class Greedy:
        name, version = "greedy", "1"

        def decide(self, env):
            return AgentResponse(
                views={
                    "AAPL": View(symbol="AAPL", score=0.5),
                    "TSLA": View(symbol="TSLA", score=0.9),
                },
                outcome="ok",
            )

    env = envir(tmp_path)  # decides AAPL only
    response = agents.invoke(Greedy(), env)
    assert set(response.views) == {"AAPL"}
    assert "TSLA" in response.detail


def test_dropping_every_view_is_not_still_ok(tmp_path):
    class Wrong:
        name, version = "wrong", "1"

        def decide(self, env):
            return AgentResponse(views={"TSLA": View(symbol="TSLA", score=0.9)}, outcome="ok")

    response = agents.invoke(Wrong(), envir(tmp_path))
    assert response.views == {}
    assert response.outcome == "empty"


# ── channels are one path ────────────────────────────────────────────────────


@pytest.mark.parametrize("channel", ["direct", "tools", "pack"])
def test_every_channel_reaches_the_same_recorded_world(tmp_path, channel):
    """Tools, pack and direct access are presentations, not separate paths — so
    a channel ablation measures the channel, not a different data set."""
    env = envir(tmp_path / channel, kinds=("prices",))
    agent = agents.build("scripted", channel=channel, reads=("prices",))
    response = agents.invoke(agent, env)
    assert response.outcome == "ok"
    reads = env.log.reads
    assert reads and all(r["kind"] == "prices" for r in reads)


def test_the_channel_is_the_platforms_knob_not_the_agents(tmp_path):
    """Same adapter, same data, different delivery — which is the comparison the
    old code could only make inside one agent via evidence_mode."""
    seen = {}
    for channel in ("direct", "tools", "pack"):
        env = envir(tmp_path / channel, kinds=("prices",))
        agent = agents.build("scripted", channel=channel, reads=("prices",))
        agents.invoke(agent, env)
        seen[channel] = env.log.counts()
    assert all(c["ok"] >= 1 for c in seen.values())


def test_an_unknown_channel_is_an_error_not_a_silent_skip(tmp_path):
    agent = agents.build("scripted", channel="telepathy", reads=("prices",))
    response = agents.invoke(agent, envir(tmp_path, kinds=("prices",)))
    assert response.outcome == "crashed"


# ── cost comparability ───────────────────────────────────────────────────────


def test_a_reported_cost_and_an_estimated_one_do_not_sum_to_either(tmp_path):
    """The old rollup stamped the total `authoritative` if any leg reported a
    cost, so an estimate could masquerade as a measurement."""
    reported = Usage(n_llm_calls=1, cost_usd=0.10, basis="reported")
    estimated = Usage(n_llm_calls=1, cost_usd=0.05, basis="estimated")
    merged = reported.merge(estimated)
    assert merged.cost_usd == pytest.approx(0.15)
    assert merged.basis == "mixed"
    assert not merged.comparable


def test_like_bases_stay_comparable():
    a = Usage(n_llm_calls=1, cost_usd=0.10, basis="reported")
    b = Usage(n_llm_calls=2, cost_usd=0.20, basis="reported")
    merged = a.merge(b)
    assert merged.basis == "reported"
    assert merged.comparable
    assert merged.n_llm_calls == 3


def test_an_empty_usage_does_not_dilute_the_basis():
    """Summing over cells starts from a blank, which must not poison the label."""
    reported = Usage(n_llm_calls=1, cost_usd=0.10, basis="reported")
    assert total([Usage(), reported, Usage()]).basis == "reported"
    assert total([]).basis == "unknown"


def test_missing_cost_stays_none_rather_than_zero():
    merged = Usage(n_llm_calls=1, tokens_in=10).merge(Usage(n_llm_calls=1, tokens_in=5))
    assert merged.cost_usd is None
    assert not merged.comparable
    assert merged.tokens_in == 15


def test_an_agents_usage_survives_the_invocation(tmp_path):
    agent = agents.build("scripted", cost_basis="reported")
    response = agents.invoke(agent, envir(tmp_path))
    assert response.usage.basis == "reported"
    assert response.usage.comparable


# ── the registry cannot drift ────────────────────────────────────────────────


def test_the_valid_names_are_the_resolvable_ones():
    """Harbor keeps a name enum beside its map; two of its names validate and
    then fail to import. Deriving one from the other makes that impossible."""
    for name in agents.names():
        assert agents.build(name, **CONFORMANT.get(name, {})) is not None


def test_every_registered_agent_is_held_to_the_contract():
    """Guards the suite itself: registering an adapter without adding it here
    (or to `SUBPROCESS_ADAPTERS`) would let it skip every check below."""
    invocable = set(agents.names()) - SUBPROCESS_ADAPTERS
    assert invocable == set(CONFORMANT), (
        "adapters in the registry but not in CONFORMANT (or vice versa): "
        f"{invocable ^ set(CONFORMANT)}"
    )


@pytest.mark.parametrize("name", sorted(SUBPROCESS_ADAPTERS))
def test_subprocess_adapters_satisfy_the_protocol_without_invoking(name):
    """Built, not invoked — real invocation needs the CLI installed and, for
    OpenClaw, writes into the operator's real profile config."""
    agent = agents.build(name)
    assert isinstance(agent, agents.Agent)
    assert agent.name and agent.version


@pytest.mark.parametrize("name", sorted(SUBPROCESS_ADAPTERS))
def test_subprocess_adapters_carry_mission_and_schema(name):
    """The same platform-supplied context every other adapter accepts."""
    agent = agents.build(name, mission_text="score it", output_schema_text="{}")
    assert agent.mission_text == "score it"
    assert agent.output_schema_text == "{}"


@pytest.mark.parametrize("name", sorted(agents.names()))
def test_every_adapter_declares_pit_enforcement(name):
    """Standard requirement: access (in-process) or cli_deny (subprocess)."""
    kwargs = CONFORMANT.get(name, {})
    agent = agents.build(name, **kwargs)
    assert getattr(type(agent), "pit_enforcement", None) in ("access", "cli_deny")


@pytest.mark.parametrize("name", sorted(SUBPROCESS_ADAPTERS))
def test_subprocess_adapters_are_cli_deny_and_override_enforce(name):
    agent = agents.build(name)
    assert type(agent).pit_enforcement == "cli_deny"
    from fintel.agents.installed.base import SubprocessAgent

    assert type(agent).enforce_pit_policy is not SubprocessAgent.enforce_pit_policy


def test_a_custom_adapter_needs_no_registry_entry():
    agent = agents.build("fintel.agents.scripted:ConstantAgent", score=0.25)
    assert agent.score == 0.25


def test_an_unknown_name_lists_what_exists():
    with pytest.raises(ValueError, match="unknown agent 'nope'.*Available"):
        agents.build("nope")


def test_registering_over_a_name_needs_saying_so():
    with pytest.raises(ValueError, match="already registered"):
        agents.register("constant", "fintel.agents.scripted:ScriptedAgent")


# ── the baseline is real ─────────────────────────────────────────────────────


def test_the_constant_baseline_scores_every_symbol_it_may_decide(tmp_path):
    portfolio = Cell(
        run_id="job-r1",
        decision_date=DAY,
        symbols=("AAPL", "MSFT", "NVDA"),
        scope="portfolio",
    )
    env = envir(tmp_path, cell=portfolio)
    response = agents.invoke(agents.build("constant", score=0.3), env)
    assert set(response.views) == set(env.policy.decidable) == {"AAPL", "MSFT", "NVDA"}
    assert {v.score for v in response.views.values()} == {0.3}


def test_the_baseline_reads_nothing(tmp_path):
    """A floor that consulted data wouldn't be a floor."""
    env = envir(tmp_path)
    agents.invoke(agents.build("constant"), env)
    assert env.log.reads == []


def test_an_agent_error_carries_its_own_verdict():
    assert AgentError("x").outcome == "crashed"
    assert SafetyRefusal("x").outcome == "refused"
