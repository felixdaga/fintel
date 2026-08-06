"""An LLM deciding one cell, over either channel.

One adapter, not two. The old repo had a single-call agent that took a rendered
dossier and a separate tool-loop analyst, duplicating prompt assembly, output
parsing and usage collection so that the two could never be compared cleanly.
Here `channel` selects the delivery and nothing else changes, so switching it
measures the channel.

The agent does not fetch. Both channels resolve to `env.access`, so the same
question gets the same PIT-clamped answer and both appear in the same record.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar

from fintel.agents import emit, prompts
from fintel.agents.base import MalformedOutput
from fintel.agents.llm import LLM, Completion, as_tool_spec
from fintel.agents.pit_policy import PitEnforcement
from fintel.environment import Environment
from fintel.models.common import Outcome
from fintel.models.decision import AgentResponse
from fintel.models.trace import ReasoningTrace, TraceStep, Usage, total

MISSION = (
    "You are an equity analyst. Judge the securities you are given on the "
    "evidence available at the decision date, and score each one."
)

RULES = (
    "Rules:\n"
    "- Every number you cite must come from the evidence provided. Do not recall "
    "prices, results or events from memory; your memory postdates the decision date.\n"
    "- Absence of data is not bad news. A lookup marked 'empty' genuinely has "
    "nothing; one marked 'failed' broke, and tells you nothing either way.\n"
    f"- Finish by calling {emit.SUBMIT_TOOL} exactly once.\n"
    "- Declining is a real answer. If the evidence does not support a view, "
    "abstain and say why rather than scoring something you don't believe."
)


def now() -> datetime:
    return datetime.now(UTC)


@dataclass
class LLMAgent:
    """`channel='pack'` renders the evidence up front; `'tools'` lets the model ask."""

    llm: LLM | None = None
    model: str | None = None
    channel: str = "pack"
    mission: str = MISSION
    # The strategy pack's mission.md / output_schema.json, wired in by
    # `simulate.cell.build_agent`. `mission` above is the generic fallback used
    # only when no pack mission is supplied (e.g. bare unit tests).
    mission_text: str = ""
    output_schema_text: str = ""
    max_rounds: int = 6
    max_tokens: int | None = 4000
    name: str = "llm"
    version: str = "1"
    # Standard adapter requirement: no native tool surface; PIT is access.
    pit_enforcement: ClassVar[PitEnforcement] = "access"
    steps: list[TraceStep] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.llm is None:
            from fintel.agents.llm import OpenRouter

            self.llm = OpenRouter.from_env(self.model)
        if self.channel not in ("pack", "tools"):
            raise ValueError(f"channel must be 'pack' or 'tools', not {self.channel!r}")

    @staticmethod
    def preflight_checks(**params: Any) -> list[str]:
        """No key needed if the caller supplies a pre-built `llm` client
        (tests, or a custom provider) — only `OpenRouter.from_env` needs one."""
        import os

        if params.get("llm") is not None:
            return []
        if not os.environ.get("OPENROUTER_API_KEY"):
            return ["OPENROUTER_API_KEY is not set; the llm agent cannot call OpenRouter"]
        return []

    def decide(self, env: Environment) -> AgentResponse:
        self.steps = []
        symbols = tuple(sorted(env.policy.decidable))
        submit = as_tool_spec(
            emit.SUBMIT_TOOL, emit.submit_description(symbols), emit.submit_schema(symbols)
        )
        if self.channel == "pack":
            payload, note = self._one_shot(env, symbols, submit)
        else:
            payload, note = self._tool_loop(env, symbols, submit)
        return self._respond(env, payload, note)

    # ── channels ────────────────────────────────────────────────────────────

    def _system_prompt(self) -> str:
        """The strategy's own mission if the pack supplied one, else the
        generic fallback (bare unit tests, or a package with no mission.md)."""
        body = self.mission_text.strip() or self.mission
        return f"{body}\n\n{RULES}"

    def _schema_block(self) -> str:
        if not self.output_schema_text.strip():
            return ""
        return f"\n\n## Output schema (from the strategy pack)\n{self.output_schema_text.strip()}"

    def _one_shot(
        self, env: Environment, symbols: tuple[str, ...], submit: dict
    ) -> tuple[dict | None, str]:
        """Everything up front, one forced call. The floor for a reading agent.

        No tools manual here: the pack channel has no tools to call, so listing
        them would promise a capability this delivery doesn't have.
        """
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Decision date: {env.cell.decision_date.isoformat()}.\n"
                    f"Score: {', '.join(symbols)}.\n\n"
                    f"{env.evidence(symbol=symbols[0] if len(symbols) == 1 else None)}"
                    f"{self._schema_block()}"
                ),
            },
        ]
        completion = self._call(messages, tools=(submit,), force_tool=emit.SUBMIT_TOOL)
        if env.nerve is not None:
            env.nerve.emit(
                "agent_stage",
                cell=env.cell.name,
                decision_date=env.cell.decision_date.isoformat(),
                stage="one_shot",
                text="",
            )
        call = completion.call_named(emit.SUBMIT_TOOL)
        if call is None:
            raise MalformedOutput(
                f"model was required to call {emit.SUBMIT_TOOL} and did not "
                f"(finish_reason={completion.finish_reason!r})"
            )
        return call.arguments, ""

    def _tool_loop(
        self, env: Environment, symbols: tuple[str, ...], submit: dict
    ) -> tuple[dict | None, str]:
        """The model asks for what it wants, within a bounded number of rounds.

        The tools are also passed natively via `tools=` below, which is what
        actually governs what the model can call; the rendered manual in the
        user message adds the task-level "how to use these together" framing
        that a bare JSON schema per tool doesn't carry.
        """
        data_tools = [
            as_tool_spec(spec.name, spec.description, spec.schema)
            for spec in env.tools.descriptors()
        ]
        tools = (*data_tools, submit)
        manual = prompts.render_tools(env.tools.descriptors())
        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Decision date: {env.cell.decision_date.isoformat()}.\n"
                    f"Score: {', '.join(symbols)}.\n\n"
                    f"## Tools\n{manual}\n\n"
                    f"Gather the evidence you need with the tools, then submit."
                    f"{self._schema_block()}"
                ),
            },
        ]

        for round_index in range(self.max_rounds):
            last_round = round_index == self.max_rounds - 1
            completion = self._call(
                messages,
                tools=tools,
                # On the final round, stop offering data and require an answer,
                # so a model that would browse forever still returns something.
                force_tool=emit.SUBMIT_TOOL if last_round else None,
            )
            if env.nerve is not None:
                env.nerve.emit(
                    "agent_stage",
                    cell=env.cell.name,
                    decision_date=env.cell.decision_date.isoformat(),
                    stage="round",
                    round=round_index + 1,
                    text=(completion.text or "")[:80],
                )
            submitted = completion.call_named(emit.SUBMIT_TOOL)
            if submitted is not None:
                return submitted.arguments, ""
            if not completion.tool_calls:
                messages.append({"role": "assistant", "content": completion.text})
                messages.append(
                    {"role": "user", "content": f"Call {emit.SUBMIT_TOOL} now with your answer."}
                )
                continue
            messages.append(self._assistant_turn(completion))
            for call in completion.tool_calls:
                if env.nerve is not None:
                    env.nerve.emit(
                        "agent_tool_call",
                        cell=env.cell.name,
                        decision_date=env.cell.decision_date.isoformat(),
                        tool=call.name,
                        args=str(call.arguments)[:120],
                    )
                payload = env.tools.call(call.name, call.arguments)
                if env.nerve is not None:
                    env.nerve.emit(
                        "agent_tool_result",
                        cell=env.cell.name,
                        decision_date=env.cell.decision_date.isoformat(),
                        tool=call.name,
                        ok=(payload.get("status") != "failed"),
                        text=(payload.get("detail") or "")[:80]
                        if payload.get("status") == "failed"
                        else "",
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": self._render(payload),
                    }
                )
        return None, f"no answer after {self.max_rounds} rounds"

    # ── plumbing ────────────────────────────────────────────────────────────

    def _call(self, messages: list[dict], **kw: Any) -> Completion:
        started = now()
        assert self.llm is not None
        completion = self.llm.complete(messages, max_tokens=self.max_tokens, **kw)
        self.steps.append(
            TraceStep(
                step_id=uuid.uuid4().hex[:12],
                kind="llm_call",
                started_at=started,
                duration_ms=int((now() - started).total_seconds() * 1000),
                model=completion.model,
                tokens_in=completion.usage.tokens_in or None,
                tokens_out=completion.usage.tokens_out or None,
                cost_usd=completion.usage.cost_usd,
                payload={
                    "finish_reason": completion.finish_reason,
                    "tool_calls": [c.name for c in completion.tool_calls],
                    "cost_basis": completion.usage.basis,
                },
            )
        )
        return completion

    @staticmethod
    def _assistant_turn(completion: Completion) -> dict:
        return {
            "role": "assistant",
            "content": completion.text or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": _dumps(call.arguments)},
                }
                for call in completion.tool_calls
            ],
        }

    @staticmethod
    def _render(payload: dict) -> str:
        """Hand back the status too, so 'nothing there' can't read as a failure."""
        return _dumps(payload)[:20000]

    def _usage(self) -> Usage:
        return total(
            Usage(
                n_llm_calls=1,
                tokens_in=step.tokens_in or 0,
                tokens_out=step.tokens_out or 0,
                cost_usd=step.cost_usd,
                basis=step.payload.get("cost_basis", "unknown"),
            )
            for step in self.steps
        )

    def _respond(self, env: Environment, payload: dict | None, note: str) -> AgentResponse:
        usage = self._usage()
        trace = ReasoningTrace(steps=list(self.steps), metadata={"channel": self.channel})

        if payload is None:
            return AgentResponse(views={}, outcome="empty", detail=note, usage=usage, trace=trace)

        reason = emit.abstained(payload)
        views, notes = emit.parse_views(payload, decidable=env.policy.decidable)
        if notes:
            trace.metadata["coercions"] = notes

        if reason is not None and not views:
            return AgentResponse(
                views={}, outcome="abstained", detail=reason, usage=usage, trace=trace
            )
        outcome: Outcome = "ok" if views else "empty"
        detail = "; ".join(notes) if not views else ""
        trace.final_explanation = next(iter(views.values())).rationale if views else (reason or "")
        return AgentResponse(views=views, outcome=outcome, detail=detail, usage=usage, trace=trace)


def _dumps(value: Any) -> str:
    import json

    return json.dumps(value, default=str)
