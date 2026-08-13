"""The in-process LLM host: error naming, output parsing, and both channels.

Everything here runs against a scripted fake model, so the failure paths that
only appear when a provider misbehaves are exercised on every commit.
"""

from __future__ import annotations

import json

import pytest

from fintel import agents
from fintel.agents import emit
from fintel.agents.base import (
    AgentError,
    AgentTimeout,
    ContextOverflow,
    MalformedOutput,
    ProviderUnavailable,
    RateLimited,
    SafetyRefusal,
)
from fintel.agents.installed.llm_agent import LLMAgent
from fintel.agents.llm import (
    Completion,
    ToolCall,
    classify_status,
    parse_completion,
    usage_of,
)
from tests.test_environment import an_environment

DECIDABLE = frozenset({"AAPL"})


# ── a fake model ─────────────────────────────────────────────────────────────


class Fake:
    """Replays scripted turns. Records what it was asked."""

    model = "fake/model"

    def __init__(self, *turns: Completion | BaseException):
        self.turns = list(turns)
        self.calls: list[dict] = []

    def complete(self, messages, *, tools=(), force_tool=None, max_tokens=None):
        self.calls.append(
            {
                "messages": messages,
                "tools": [t["function"]["name"] for t in tools],
                "force_tool": force_tool,
            }
        )
        if not self.turns:
            raise AssertionError("fake model ran out of scripted turns")
        turn = self.turns.pop(0)
        if isinstance(turn, BaseException):
            raise turn
        return turn


def submitted(views: list[dict] | None = None, **extra) -> Completion:
    payload = {"views": views if views is not None else [_view()], **extra}
    return Completion(
        tool_calls=(ToolCall(id="c1", name=emit.SUBMIT_TOOL, arguments=payload),),
        model="fake/model",
        finish_reason="tool_calls",
    )


def _view(**kw) -> dict:
    return {"symbol": "AAPL", "score": 0.4, "rationale": "cheap", **kw}


def asked(name: str, **args) -> Completion:
    return Completion(
        tool_calls=(ToolCall(id="t1", name=name, arguments=args),),
        model="fake/model",
        finish_reason="tool_calls",
    )


# ── failures get their own names ─────────────────────────────────────────────


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (429, "slow down", RateLimited),
        (503, "unavailable", ProviderUnavailable),
        (500, "boom", ProviderUnavailable),
        (400, "maximum context length exceeded", ContextOverflow),
        (400, "prompt is too long for this model", ContextOverflow),
        (400, "blocked by content filter", SafetyRefusal),
        (403, "request flagged for safety", SafetyRefusal),
        (401, "no key", AgentError),
        (404, "no such model", AgentError),
    ],
)
def test_a_failed_response_is_named(status, body, expected):
    assert isinstance(classify_status(status, body), expected)


def test_a_context_overflow_is_not_treated_as_transient():
    """Retrying a prompt that doesn't fit reproduces it at cost."""
    from fintel.models.common import RETRYABLE

    assert classify_status(400, "maximum context").outcome not in RETRYABLE
    assert classify_status(429, "rate limit").outcome in RETRYABLE


def test_a_content_filter_finish_is_a_refusal_not_an_empty():
    payload = {"choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}]}
    with pytest.raises(SafetyRefusal):
        parse_completion(payload, model="m")


def test_model_prose_mentioning_safety_is_not_a_refusal():
    """Regression: BODY_PATTERNS must not scan successful completion text.
    3M qualitative writeups often say 'safety litigation' in the first 400
    chars and were mis-tagged as SafetyRefusal → cell outcome refused."""
    from fintel.agents.llm import parse_completion

    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        "# 3M Company (MMM) — Qualitative Franchise & Risk Assessment\n"
                        "3M faces material safety litigation and PFAS exposure risks.\n"
                    )
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }
    completion = parse_completion(payload, model="m")
    assert "safety litigation" in completion.text


def test_a_truncated_response_says_so():
    payload = {"choices": [{"message": {"content": "half a th"}, "finish_reason": "length"}]}
    with pytest.raises(ContextOverflow):
        parse_completion(payload, model="m")


def test_unparseable_tool_arguments_are_a_parse_error_not_a_crash():
    payload = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"id": "1", "function": {"name": "submit_views", "arguments": "{not json"}}
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    with pytest.raises(MalformedOutput):
        parse_completion(payload, model="m")


def test_an_empty_response_is_a_parse_error():
    with pytest.raises(MalformedOutput):
        parse_completion({"choices": []}, model="m")


def test_a_provider_error_in_a_200_body_is_treated_as_model_text():
    """A 200 with finish_reason 'stop' carrying 'rate limit' in the text is
    model prose, not a real rate limit (that arrives as HTTP 429). Scanning
    successful completion text for error needles false-positives on ordinary
    model output — e.g. a 3M qualitative report mentioning 'safety litigation'."""
    message = {"content": "API Error: rate limit"}
    payload = {"choices": [{"message": message, "finish_reason": "stop"}]}
    completion = parse_completion(payload, model="m")
    assert completion.text == "API Error: rate limit"
    assert completion.finish_reason == "stop"


# ── cost basis comes from the provider or not at all ─────────────────────────


def test_a_stated_cost_is_reported():
    raw = {"usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.004}}
    usage = usage_of(raw, model="m")
    assert usage.basis == "reported"
    assert usage.comparable
    assert usage.cost_usd == pytest.approx(0.004)


def test_an_unstated_cost_is_unknown_not_zero():
    """No rate card here, so we don't manufacture a number that would later be
    summed with a measured one."""
    usage = usage_of({"usage": {"prompt_tokens": 10, "completion_tokens": 2}}, model="m")
    assert usage.basis == "unknown"
    assert usage.cost_usd is None
    assert not usage.comparable
    assert usage.tokens_in == 10


# ── output parsing is one path ───────────────────────────────────────────────


def test_a_clean_payload_becomes_views():
    views, notes = emit.parse_views({"views": [_view()]}, decidable=DECIDABLE)
    assert views["AAPL"].score == pytest.approx(0.4)
    assert views["AAPL"].conviction is None
    assert views["AAPL"].time_horizon is None
    assert notes == []


def test_emit_does_not_invent_schema_absent_fields():
    """decision.json should follow the submit payload / strategy schema, not
    platform View defaults for unused keys."""
    views, _ = emit.parse_views({"views": [_view()]}, decidable=DECIDABLE)
    dumped = views["AAPL"].model_dump(mode="json", exclude_none=True)
    assert "conviction" not in dumped
    assert "time_horizon" not in dumped
    assert set(dumped) >= {"symbol", "score", "rationale"}


def test_an_overshooting_score_is_clamped_and_noted():
    """1.4 means 'very positive'; rejecting it would lose that."""
    views, notes = emit.parse_views({"views": [_view(score=1.4)]}, decidable=DECIDABLE)
    assert views["AAPL"].score == 1.0
    assert any("coerced" in n for n in notes)


def test_an_unscoreable_score_becomes_neutral_with_a_note():
    views, notes = emit.parse_views({"views": [_view(score="very good")]}, decidable=DECIDABLE)
    assert views["AAPL"].score == 0.0
    assert notes


def test_one_bad_entry_does_not_lose_the_others():
    payload = {"views": [_view(), "garbage", {"score": 0.1}, _view(symbol="MSFT")]}
    views, notes = emit.parse_views(payload, decidable=frozenset({"AAPL", "MSFT"}))
    assert set(views) == {"AAPL", "MSFT"}
    assert len(notes) == 2


def test_a_view_on_a_symbol_this_cell_cannot_decide_is_refused():
    views, notes = emit.parse_views({"views": [_view(symbol="TSLA")]}, decidable=DECIDABLE)
    assert views == {}
    assert any("not decidable" in n for n in notes)


def test_a_duplicate_symbol_keeps_the_first():
    payload = {"views": [_view(score=0.4), _view(score=-0.9)]}
    views, notes = emit.parse_views(payload, decidable=DECIDABLE)
    assert views["AAPL"].score == pytest.approx(0.4)
    assert any("twice" in n for n in notes)


def test_views_of_the_wrong_shape_are_reported_not_raised():
    views, notes = emit.parse_views({"views": "nope"}, decidable=DECIDABLE)
    assert views == {}
    assert "expected a list" in notes[0]


def test_the_schema_offers_an_explicit_way_to_decline():
    """Without one, 'no views' means both 'no opinion' and 'something broke'."""
    schema = emit.submit_schema(("AAPL",))
    assert "abstain" in schema["properties"]
    assert emit.abstained({"views": [], "abstain": True, "abstain_reason": "thin"}) == "thin"
    assert emit.abstained({"views": []}) is None


# ── the pack channel ─────────────────────────────────────────────────────────


def test_the_pack_channel_takes_one_call_and_returns_views(tmp_path):
    fake = Fake(submitted())
    env = an_environment(tmp_path, kinds=("prices",))
    response = agents.invoke(LLMAgent(llm=fake, channel="pack"), env)
    assert response.outcome == "ok"
    assert set(response.views) == {"AAPL"}
    assert len(fake.calls) == 1
    assert fake.calls[0]["force_tool"] == emit.SUBMIT_TOOL


def test_the_pack_channel_puts_the_evidence_in_the_prompt(tmp_path):
    fake = Fake(submitted())
    env = an_environment(tmp_path, kinds=("prices",))
    agents.invoke(LLMAgent(llm=fake, channel="pack"), env)
    prompt = fake.calls[0]["messages"][1]["content"]
    assert "## Prices" in prompt
    assert "2024-06-03" in prompt
    # And the reads went through the recorded path.
    assert env.log.counts()["ok"] >= 1


def test_the_pack_channel_offers_no_data_tools(tmp_path):
    """Its whole point is that the model cannot ask for more."""
    fake = Fake(submitted())
    agents.invoke(LLMAgent(llm=fake, channel="pack"), an_environment(tmp_path, kinds=("prices",)))
    assert fake.calls[0]["tools"] == [emit.SUBMIT_TOOL]


def test_a_forced_call_the_model_ignores_is_a_parse_error(tmp_path):
    fake = Fake(Completion(text="I'd rather chat", finish_reason="stop"))
    response = agents.invoke(LLMAgent(llm=fake, channel="pack"), an_environment(tmp_path))
    assert response.outcome == "parse_error"


# ── the tools channel ────────────────────────────────────────────────────────


def test_the_tools_channel_lets_the_model_ask_then_submit(tmp_path):
    fake = Fake(asked("get_prices", symbol="AAPL"), submitted())
    env = an_environment(tmp_path, kinds=("prices",))
    response = agents.invoke(LLMAgent(llm=fake, channel="tools"), env)
    assert response.outcome == "ok"
    assert len(fake.calls) == 2
    assert "get_prices" in fake.calls[0]["tools"]
    # The tool result came back through access, so it is in the record.
    assert [r["kind"] for r in env.log.reads] == ["prices"]


def test_a_tool_result_carries_its_status_into_the_conversation(tmp_path):
    """So the model can tell 'nothing there' from 'the lookup broke'."""
    fake = Fake(asked("get_prices", symbol="AAPL"), submitted())
    env = an_environment(tmp_path, kinds=("prices",))
    agents.invoke(LLMAgent(llm=fake, channel="tools"), env)
    tool_message = next(m for m in fake.calls[1]["messages"] if m["role"] == "tool")
    assert json.loads(tool_message["content"])["status"] == "ok"


def test_a_tool_the_run_does_not_have_is_refused_not_fatal(tmp_path):
    fake = Fake(asked("get_secrets", symbol="AAPL"), submitted())
    env = an_environment(tmp_path, kinds=("prices",))
    response = agents.invoke(LLMAgent(llm=fake, channel="tools"), env)
    assert response.outcome == "ok"
    tool_message = next(m for m in fake.calls[1]["messages"] if m["role"] == "tool")
    assert json.loads(tool_message["content"])["status"] == "denied"


def test_the_last_round_forces_an_answer(tmp_path):
    """A model that would browse forever still has to produce something."""
    fake = Fake(asked("get_prices", symbol="AAPL"), submitted())
    env = an_environment(tmp_path, kinds=("prices",))
    agents.invoke(LLMAgent(llm=fake, channel="tools", max_rounds=2), env)
    assert fake.calls[0]["force_tool"] is None
    assert fake.calls[1]["force_tool"] == emit.SUBMIT_TOOL


def test_a_model_that_never_submits_is_empty_not_ok(tmp_path):
    chatter = [Completion(text="thinking...", finish_reason="stop") for _ in range(3)]
    fake = Fake(*chatter)
    response = agents.invoke(
        LLMAgent(llm=fake, channel="tools", max_rounds=3), an_environment(tmp_path)
    )
    assert response.outcome == "empty"
    assert "3 rounds" in response.detail


# ── outcomes and usage survive the round trip ────────────────────────────────


def test_a_declared_abstention_is_recorded_as_one(tmp_path):
    fake = Fake(submitted(views=[], abstain=True, abstain_reason="no edge here"))
    response = agents.invoke(LLMAgent(llm=fake, channel="pack"), an_environment(tmp_path))
    assert response.outcome == "abstained"
    assert response.detail == "no edge here"
    assert not response.retryable


def test_an_empty_submission_without_a_reason_is_empty(tmp_path):
    fake = Fake(submitted(views=[]))
    response = agents.invoke(LLMAgent(llm=fake, channel="pack"), an_environment(tmp_path))
    assert response.outcome == "empty"
    assert response.retryable


def test_a_rate_limit_mid_run_keeps_its_name(tmp_path):
    fake = Fake(RateLimited("429 from provider"))
    response = agents.invoke(LLMAgent(llm=fake, channel="pack"), an_environment(tmp_path))
    assert response.outcome == "rate_limited"
    assert response.retryable


def test_a_timeout_mid_run_keeps_its_name(tmp_path):
    fake = Fake(AgentTimeout("gave up"))
    response = agents.invoke(LLMAgent(llm=fake, channel="pack"), an_environment(tmp_path))
    assert response.outcome == "timeout"


def test_usage_accumulates_across_rounds(tmp_path):
    from fintel.models.trace import Usage

    def priced(completion: Completion) -> Completion:
        return Completion(
            text=completion.text,
            tool_calls=completion.tool_calls,
            usage=Usage(
                n_llm_calls=1, tokens_in=100, tokens_out=20, cost_usd=0.01, basis="reported"
            ),
            model="fake/model",
            finish_reason=completion.finish_reason,
        )

    fake = Fake(priced(asked("get_prices", symbol="AAPL")), priced(submitted()))
    env = an_environment(tmp_path, kinds=("prices",))
    response = agents.invoke(LLMAgent(llm=fake, channel="tools"), env)
    assert response.usage.n_llm_calls == 2
    assert response.usage.cost_usd == pytest.approx(0.02)
    assert response.usage.basis == "reported"
    assert len(response.trace.steps) == 2


def test_coercions_are_kept_in_the_trace(tmp_path):
    fake = Fake(submitted(views=[_view(score=9.0)]))
    response = agents.invoke(LLMAgent(llm=fake, channel="pack"), an_environment(tmp_path))
    assert response.views["AAPL"].score == 1.0
    assert response.trace.metadata["coercions"]


# ── the channel is the platform's knob ───────────────────────────────────────


def test_the_same_agent_runs_over_either_channel(tmp_path):
    """The comparison the old code could only make inside one agent."""
    outcomes = {}
    for channel, turns in (
        ("pack", (submitted(),)),
        ("tools", (asked("get_prices", symbol="AAPL"), submitted())),
    ):
        env = an_environment(tmp_path / channel, kinds=("prices",))
        response = agents.invoke(LLMAgent(llm=Fake(*turns), channel=channel), env)
        outcomes[channel] = response
    assert all(r.outcome == "ok" for r in outcomes.values())
    assert outcomes["pack"].trace.metadata["channel"] == "pack"
    assert outcomes["tools"].trace.metadata["channel"] == "tools"
    # Same views, different route to them.
    assert set(outcomes["pack"].views) == set(outcomes["tools"].views)


def test_an_unknown_channel_is_refused_at_construction():
    with pytest.raises(ValueError, match="channel must be"):
        LLMAgent(llm=Fake(), channel="semaphore")


def test_the_agent_is_registered_and_conformant(tmp_path):
    agent = agents.build("llm", llm=Fake(submitted()), channel="pack")
    assert isinstance(agent, agents.Agent)
    assert agents.invoke(agent, an_environment(tmp_path)).outcome == "ok"


def test_the_prompt_tells_the_model_its_memory_postdates_the_date(tmp_path):
    """The one instruction that matters most for a backtest."""
    fake = Fake(submitted())
    agents.invoke(LLMAgent(llm=fake, channel="pack"), an_environment(tmp_path))
    system = fake.calls[0]["messages"][0]["content"]
    assert "postdates the decision date" in system
    assert "empty" in system and "failed" in system


# ── strategy pack mission + output schema ───────────────────────────────────


def test_a_pack_mission_replaces_the_generic_fallback(tmp_path):
    fake = Fake(submitted())
    agent = LLMAgent(llm=fake, channel="pack", mission_text="Be a momentum analyst.")
    agents.invoke(agent, an_environment(tmp_path))
    system = fake.calls[0]["messages"][0]["content"]
    assert "Be a momentum analyst." in system
    assert "equity analyst. Judge" not in system  # the generic MISSION fallback


def test_no_pack_mission_falls_back_to_the_generic_default(tmp_path):
    fake = Fake(submitted())
    agents.invoke(LLMAgent(llm=fake, channel="pack"), an_environment(tmp_path))
    system = fake.calls[0]["messages"][0]["content"]
    assert "equity analyst. Judge" in system


def test_the_output_schema_reaches_the_pack_channel_prompt(tmp_path):
    fake = Fake(submitted())
    agent = LLMAgent(llm=fake, channel="pack", output_schema_text='{"score": "..."}')
    agents.invoke(agent, an_environment(tmp_path))
    user = fake.calls[0]["messages"][1]["content"]
    assert "## Output schema" in user
    assert '"score"' in user


def test_the_output_schema_reaches_the_tools_channel_prompt(tmp_path):
    fake = Fake(submitted())
    agent = LLMAgent(llm=fake, channel="tools", output_schema_text="see mission.md")
    agents.invoke(agent, an_environment(tmp_path))
    user = fake.calls[0]["messages"][1]["content"]
    assert "## Output schema" in user
    assert "see mission.md" in user


def test_no_output_schema_means_no_schema_block(tmp_path):
    fake = Fake(submitted())
    agents.invoke(LLMAgent(llm=fake, channel="pack"), an_environment(tmp_path))
    user = fake.calls[0]["messages"][1]["content"]
    assert "## Output schema" not in user


def test_the_tools_channel_prompt_carries_a_rendered_tools_manual(tmp_path):
    """On top of the native `tools=` schemas, the tools channel gets a prose
    manual — the task-level framing a bare JSON schema doesn't carry."""
    fake = Fake(asked("get_prices", symbol="AAPL"), submitted())
    env = an_environment(tmp_path, kinds=("prices",))
    agents.invoke(LLMAgent(llm=fake, channel="tools"), env)
    user = fake.calls[0]["messages"][1]["content"]
    assert "## Tools" in user
    assert "get_prices" in user


def test_the_pack_channel_prompt_has_no_tools_manual(tmp_path):
    """It has no tools to call, so it shouldn't be told about any."""
    fake = Fake(submitted())
    agents.invoke(LLMAgent(llm=fake, channel="pack"), an_environment(tmp_path, kinds=("prices",)))
    user = fake.calls[0]["messages"][1]["content"]
    assert "## Tools" not in user


# ── preflight ─────────────────────────────────────────────────────────────────


def test_llm_preflight_passes_with_a_prebuilt_client():
    assert LLMAgent.preflight_checks(llm=Fake()) == []


def test_llm_preflight_flags_a_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    problems = LLMAgent.preflight_checks()
    assert problems and "OPENROUTER_API_KEY" in problems[0]


def test_llm_preflight_passes_with_an_api_key_set(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    assert LLMAgent.preflight_checks() == []
