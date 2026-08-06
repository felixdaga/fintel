"""Standalone four-call specialist pipeline — platform-agnostic.

This module has **no fintel imports**. It is the reusable agent: a quantitative
specialist and a qualitative specialist run in parallel over pre-rendered
evidence packs, an optional independent verifier audits their reports, and a
final Portfolio Manager synthesises them into one structured decision.

The host (a fintel adapter, or any other backtesting platform) supplies:

* an ``llm`` client satisfying :class:`LLMClient` (one method, ``complete``),
* a ``submit_tool`` spec (name + description + JSON schema) defining the
  output contract the PM must emit via a forced tool call,
* pre-rendered evidence strings (``quant_evidence``, ``qual_evidence``),
* optional ``mission_text`` / ``output_schema_text`` and an ``on_stage``
  listener for live progress.

The pipeline returns :class:`AgentResult` — the PM's raw ``submit`` arguments
plus the specialist/verifier reports and one :class:`CallRecord` per LLM call.
The host parses ``submit_args`` into its own view model; this module never
imports a platform's data classes, so it is portable across strategies and
backtesting hosts.

Pipeline::

    quantitative specialist -+
                        +-> independent verifier -> final PM (submit)
    qualitative specialist -+
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# ── Host-supplied dependencies (structural protocols, no platform import) ──


class LLMClient(Protocol):
    """A minimal chat-completion client. The host injects its own (e.g. an
    OpenRouter wrapper). Only ``complete`` is required."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: tuple[dict, ...] = (),
        force_tool: str | None = None,
        max_tokens: int | None = None,
    ) -> Any: ...


class Completion(Protocol):
    """What ``LLMClient.complete`` returns. Read structurally — the host's
    completion type need only expose these attributes."""

    text: str
    finish_reason: str | None
    model: str | None
    usage: Any
    tool_calls: list[Any]

    def call_named(self, name: str) -> Any: ...


StageListener = Callable[[str, str], None]
"""``(stage, symbol)`` — called once at the start of each pipeline stage."""


# ── Prompts ────────────────────────────────────────────────────────────────

QUANTITATIVE_SPECIALIST_SYSTEM = """\
You are the quantitative analyst in a fundamental-attractiveness pipeline.
Your job: judge where the *business* is going from the numbers — earnings
trajectory and durability, margins, cash generation, balance-sheet resilience,
and valuation versus the company's own history. Do not momentum-chase.

DATA SOURCES THE AGENT USES (all point-in-time; the user prompt lists which are in this pack):
  fundamentals     — income / balance / cash-flow filings and trajectories
  ratios           — daily trailing valuation & profitability vs own history
  prices           — level, range, and returns (context, not the thesis)
  macro            — FRED regime (rates, vol, oil, credit, …)
  news_sentiment   — daily sentiment score/count series derived from news

SOURCE WEIGHT (insightful fundamental analyst — higher = more thesis weight):
  1. fundamentals  — primary. The business itself.
  2. ratios        — primary co-anchor. Is the market paying more or less for
                     the same earnings power than it used to? Valuation is
                     conditional on earnings quality in fundamentals; a low
                     multiple alone is not "cheap."
  3. prices        — secondary check only: has price already moved with (or
                     against) the fundamental path? Never the reason for the
                     rating.
  4. macro         — regime overlay that may qualify valuation or durability
                     (e.g. rates vs earnings yield, oil shock). Not the driver.
  5. news_sentiment — weakest quant input. Mood thermometer; do not let a
                     sentiment trend substitute for fundamentals.

Write a concise report (aim for ~600-1000 words; do not exceed ~1200 words — the
Portfolio Manager reads the structure and the key-evidence citations, not
length, and an overlong report risks truncation) with:
1. revenue, earnings, margin, cash-flow, and balance-sheet trajectory
   (fundamentals-first);
2. earnings quality and balance-sheet resilience from the numbers;
3. valuation versus the company's own supplied history (ratios);
4. price level / return context versus that fundamental path (prices as check);
5. macro / sentiment overlays only where they materially qualify the view;
6. the strongest quantitative upside and downside;
7. a provisional quantitative conclusion on fundamental attractiveness.

Every numerical claim must quote an exact supplied value. Mark unavailable
evidence explicitly. If the growth panel says TTM is incomplete, do not invent
or cite a trailing-twelve-month total — use QoQ/YoY and per-quarter figures
only. Prefer operating_cash_flow when free_cash_flow is n/a; missing FCF is not
missing cash generation. Do not invent business narrative, news events, or
filing text. Do not discuss portfolio sizing, stops, or entry timing.

End your report with a `## key evidence` section listing the specific data
points that materially moved your conclusion (typically 3-8), one per line in
the format:
  source_type | source_id | excerpt
where source_type is one of: fundamentals, ratios, prices, macro,
news_sentiment; source_id is the date or identifier (filing date /
bar date / observation date); excerpt is the exact value (e.g.
pe_diluted=36.2x, close=177.19, revenue=215.9B, 10y_yield=4.12%). Prefer
fundamentals and ratios citations for thesis-driving points. These are what
the Portfolio Manager lifts into sources_cited — make them precise and
traceable to a single supplied datum."""


QUALITATIVE_SPECIALIST_SYSTEM = """\
You are the qualitative analyst in a fundamental-attractiveness pipeline.
Your job: assess franchise quality, competitive position, structural growth,
and material risks/offsets from text — so the Portfolio Manager can qualify
the quantitative fundamental view. Recent headlines sharpen or qualify; they
are not the driver. A good quarter or a headline does not make a bad business
attractive, and a bad headline does not undo a durable franchise.

DATA SOURCES THE AGENT USES (all point-in-time; the user prompt lists which are in this pack):
  web_search  — two tiers:
                  structural (business / competitive / risk / strategy) ~30d
                  updates (latest developments / earnings) ~7d
  news        — per-ticker articles, recent window only (~14d)
  filing_text — 10-K / 10-Q / 8-K sections (when bound by the strategy)

SOURCE WEIGHT (insightful fundamental analyst — higher = more thesis weight):
  1. web_search structural — primary qualitative. Business model, segments,
     competitive position, structural risks, capital-allocation strategy.
     This is where franchise and structural growth live.
  2. news — secondary. Dated developments (guidance, regulation, M&A,
     leadership, product) that sharpen or qualify the franchise read.
     Supporting context, not the thesis driver.
  3. web_search updates — tertiary. Freshest catalysts; highest noise /
     momentum contamination. Use to date events; do not let a single update
     headline outweigh structural franchise evidence.

Write a concise report (aim for ~600-1000 words; do not exceed ~1200 words — the
Portfolio Manager reads the structure and the key-evidence citations, not
length, and an overlong report risks truncation) with:
1. business / franchise read (prefer structural web_search);
2. dated recent developments from news + updates (guidance, competition,
   regulation, corporate actions, product, capital allocation);
3. material qualitative risks and offsets;
4. a concise positive / negative / neutral qualitative assessment of how the
   narrative supports or challenges fundamental attractiveness.

PRECISION RULES — quote or omit:
- Separate confirmed evidence from inference; label inferences as such.
- Cite the supplied date for every event.
- Never upgrade vague text into a precise number (e.g. "share gains" must not
  become "4pp"; "strong cash flow" must not become "$54B FCF").
- Do not invent numbers, multiples, or financial facts that are not verbatim
  (or clearly numeric) in the supplied text. If you use a web/news number,
  put that exact phrasing in `## key evidence`.
- Do not relabel operating cash flow as free cash flow (or vice versa).
- Do not discuss portfolio sizing, stops, or entry timing.

End your report with a `## key evidence` section listing the specific data
points that materially moved your conclusion (typically 3-8), one per line in
the format:
  source_type | source_id | excerpt
where source_type is news or web_search; source_id is the published date or
URL/host; excerpt is a short verbatim quote or exact fact. Prefer structural
web_search citations for franchise claims and news citations for dated events.
These are what the Portfolio Manager lifts into sources_cited — make them
precise and traceable to a single supplied datum."""


VERIFIER_SYSTEM = """\
You are an independent adversarial verifier in a centralized investment
pipeline. Audit the specialist reports against the raw evidence. You are not a
second research analyst and must not rewrite a full thesis or invent facts.

STRICT EVIDENCE BOUNDARY — use ONLY the supplied specialist reports and raw
evidence blocks in this prompt. Do not import outside knowledge (macro events,
one-time charges, lawsuits, accounting adjustments, product launches, or any
"common knowledge" about the company) unless that fact appears explicitly in
the supplied text. If you suspect missing context, say so as "not stated in
supplied evidence" — never fill the gap from memory. unsupported_claims must
quote a report phrase and show it conflicts with or is absent from the raw
packs; do not add off-pack facts as "context the PM should know."

Check exactly:
1. unsupported or numerically inconsistent claims vs the raw packs;
2. omitted evidence that is present in the packs and could change direction
   or magnitude;
3. valuation conclusions that ignore earnings quality or balance-sheet risk
   shown in the packs;
4. excessive reliance on recent narrative without fundamental support in the
   packs;
5. internal contradictions between the quantitative and qualitative reports
   (especially cash: OCF present in quant pack vs "cash unavailable", or qual
   citing FCF/OCF figures that conflict with quant);
6. specialists citing evidence outside their pack (quant using news text, or
   qual inventing financial figures);
7. ECONOMIC IDENTITY — do not rubber-stamp "matches pack":
   - If the pack marks TTM incomplete / non-contiguous, any specialist TTM
     total is unsupported;
   - Over-precise claims (e.g. "4pp") must appear in the cited excerpt, not
     merely in a paraphrase;
   - Any content dated on or after the decision date in the packs is a PIT
     leak: severity at least major; recommend LOWER or NEUTRALIZE if used.

verified_points must state an economic check (e.g. "YoY EPS +18.3% equals
2.84/2.40"), not "matches raw data" alone.

Emit submit_verification exactly once. Prefer recommendation=KEEP with
severity=none and corrections=[] when issues are minor or stylistic. Escalate
severity when TTM is misused, quant↔qual cash conflicts, over-precise uncited
claims, or PIT leaks appear. Use LOWER/RAISE/NEUTRALIZE when direction or
magnitude should change, and list at most 3 concrete advisory notes (ids C1..).
These notes guide the Portfolio Manager; they are not a hard gate."""


# Hard lead directive, placed FIRST in the PM system message so it has the
# highest precedence. mimo-v2.5-pro has no hidden reasoning channel, so any
# free text before the forced tool call is a pure preamble — it is discarded
# by parse_completion and only risks truncating the call. This directive and
# the matching rule in _PM_RULES make the model emit submit_views directly.
# Thesis / score scale / horizon / lane weights live in the strategy pack's
# mission.md (wired into pm_system as mission_text). The agent only owns
# pipeline mechanics below — do not restate package policy here.
_PM_LEAD = (
    "RESPOND WITH ONLY THE submit_views TOOL CALL. Do not write any prose, "
    "essay, reasoning, or narrative before the tool call — that text is "
    "discarded and only risks truncating the call. Your entire thesis goes "
    "inside the view's rationale and key_factors fields.\n\n"
)

_PM_RULES = (
    "Rules (pipeline mechanics — the strategy mission owns the thesis):\n"
    "- Every number you cite must come from the specialist reports / evidence "
    "provided. Do not recall prices, results or events from memory; your "
    "memory postdates the decision date.\n"
    "- Every material claim in rationale must also appear in key_factors and "
    "sources_cited (copy from each specialist's `## key evidence` block).\n"
    "- Absence of data is not bad news. A lookup marked 'empty' genuinely has "
    "nothing; one marked 'failed' broke, and tells you nothing either way. "
    "Labels like TTM incomplete mean do not invent a trailing total.\n"
    "- Declining is a real answer. If the evidence does not support a view, "
    "abstain and say why rather than scoring something you don't believe.\n"
    "- OUTPUT SHAPE: respond with ONLY the submit_views tool call. Do NOT write "
    "any free text, essay, or narrative before the tool call — it is discarded, "
    "wastes the output budget, and risks truncating the call itself. Your "
    "entire thesis goes inside the view's `rationale` and `key_factors` fields; "
    "the tool call IS the answer, not a preamble followed by it.\n"
)


def _quantitative_specialist_prompt(
    *, ticker: str, trade_date: str, evidence: str, kinds: tuple[str, ...] = ()
) -> str:
    kind_list = ", ".join(kinds) if kinds else "(none)"
    return f"""The instrument is {ticker}. The decision date is {trade_date}; all
evidence is strictly point-in-time before that date.

## Quantitative evidence
Kinds in this pack: {kind_list}.
{evidence}

Produce the quantitative report now."""


def _qualitative_specialist_prompt(
    *, ticker: str, trade_date: str, evidence: str, kinds: tuple[str, ...] = ()
) -> str:
    kind_list = ", ".join(kinds) if kinds else "(none)"
    return f"""The instrument is {ticker}. The decision date is {trade_date}; all
evidence is strictly point-in-time before that date.

## Qualitative evidence
Kinds in this pack: {kind_list}.
{evidence}

Produce the qualitative report now."""


def _verifier_prompt(
    *, ticker: str, trade_date: str, quantitative_evidence: str,
    qualitative_evidence: str, quantitative_report: str, qualitative_report: str,
) -> str:
    return f"""Audit the specialist reports for {ticker} as of {trade_date}.

Call submit_verification exactly once.

## Quantitative specialist report
{quantitative_report}

## Qualitative specialist report
{qualitative_report}

## Raw quantitative evidence
{quantitative_evidence}

## Raw qualitative evidence
{qualitative_evidence}
"""


def _render_verification_compact(verification_obj: dict[str, Any] | None) -> str:
    obj = verification_obj or {}
    lines = [
        f"recommendation={obj.get('recommendation', 'n/a')}",
        f"severity={obj.get('severity', 'n/a')}",
    ]
    verified = obj.get("verified_points") or []
    if verified:
        lines.append("verified: " + "; ".join(str(v) for v in verified[:6]))
    unsupported = obj.get("unsupported_claims") or []
    if unsupported:
        lines.append("unsupported: " + "; ".join(str(v) for v in unsupported[:6]))
    corrections = obj.get("corrections") or []
    if corrections:
        lines.append("advisory notes:")
        for item in corrections:
            if not isinstance(item, dict):
                lines.append(f"- {item}")
                continue
            lines.append(
                f"- [{item.get('id', '?')}] {item.get('issue', '')} "
                f"-> {item.get('required_action', '')}"
            )
    else:
        lines.append("advisory notes: (none)")
    return "\n".join(lines)


def _final_pm_prompt(
    *, ticker: str, trade_date: str, quantitative_report: str,
    qualitative_report: str, verification_obj: dict[str, Any] | None,
    include_verification: bool,
) -> str:
    """PM user prompt: reports (+ optional verification). Thesis is in system mission."""
    if include_verification:
        verify_instr = (
            "Synthesize the specialist reports under the strategy mission in your "
            "system prompt. Treat the verification block as advisory — weigh it, "
            "override it when the reports clearly justify otherwise, and explain "
            "briefly in the thesis.\n"
            "Emit submit_views right away (no prose preamble). Every key factor "
            "needs a specific supplied value or dated event from the reports. "
            "Populate sources_cited by copying from each specialist's "
            "`## key evidence` block. Leave addressed_corrections empty unless "
            "you optionally note how an advisory id influenced the call."
        )
        verify_block = (
            "\n\n## Structured verification (advisory)\n"
            f"{_render_verification_compact(verification_obj)}"
        )
    else:
        verify_instr = (
            "Synthesize the specialist reports under the strategy mission in your "
            "system prompt into a single decision.\n"
            "Emit submit_views right away (no prose preamble). Every key factor "
            "needs a specific supplied value or dated event from the reports. "
            "Populate sources_cited by copying from each specialist's "
            "`## key evidence` block. Leave addressed_corrections empty."
        )
        verify_block = ""
    return f"""You are the final Portfolio Manager (orchestrator) for {ticker} as of {trade_date}.

{verify_instr}

## Quantitative specialist report
{quantitative_report}

## Qualitative specialist report
{qualitative_report}{verify_block}
"""


# ── Tool specs ─────────────────────────────────────────────────────────────

# The verifier's audit contract is owned by the agent (it is an internal stage,
# not the host's output). Mirrors the Delorean VerificationReport model.
SUBMIT_VERIFICATION = "submit_verification"
_VERIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "recommendation": {
            "type": "string",
            "enum": ["KEEP", "LOWER", "RAISE", "NEUTRALIZE"],
            "description": "KEEP if reports sound; LOWER/RAISE if magnitude/direction should move; NEUTRALIZE if the evidence does not support a non-zero lean.",
        },
        "severity": {"type": "string", "enum": ["none", "minor", "major"]},
        "verified_points": {"type": "array", "items": {"type": "string"}},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "issue": {"type": "string"},
                    "required_action": {"type": "string"},
                },
                "required": ["id", "issue", "required_action"],
            },
        },
    },
    "required": ["recommendation"],
}


def _as_tool_spec(name: str, description: str, schema: dict) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": schema}}


# A generic default submit schema (a rating on [-1,+1] with rationale, key
# factors, and cited sources). Hosts that want a different output contract
# pass their own ``submit_tool`` to :meth:`OptimizedAgent.decide_one`.
_DEFAULT_SUBMIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "views": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "score": {"type": "number", "minimum": -1, "maximum": 1},
                    "rationale": {"type": "string"},
                    "key_factors": {"type": "array", "items": {"type": "string"}},
                    "sources_cited": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_type": {"type": "string"},
                                "source_id": {"type": "string"},
                                "excerpt": {"type": "string"},
                            },
                        },
                    },
                },
                "required": ["symbol", "score", "rationale"],
            },
        },
        "abstain": {"type": "boolean"},
        "abstain_reason": {"type": "string"},
    },
    "required": ["views"],
}


# Used only when the host did not wire a strategy mission (unit tests / bare
# runs). Real jobs always pass package mission.md via mission_text.
_DEFAULT_MISSION = (
    "You are a systematic equity research analyst. Rate each company on [-1,+1] "
    "using only the supplied point-in-time evidence. Prefer milder scores when "
    "evidence is thin or conflicting. The score is a rating, not a trade."
)


# ── Records (platform-neutral) ─────────────────────────────────────────────


@dataclass
class CallRecord:
    """One LLM call's outcome + bookkeeping, in platform-neutral terms. The
    host turns these into its own trace/usage types."""

    stage: str
    text: str
    finish_reason: str | None
    model: str | None
    tool_name: str | None  # the forced tool's name, if any
    tool_args: dict[str, Any] | None  # that tool's arguments (None if not called)
    tokens_in: int
    tokens_out: int
    cost_usd: float | None
    started_at: datetime
    duration_ms: int


@dataclass
class AgentResult:
    """What :meth:`OptimizedAgent.decide_one` returns.

    ``submit_args`` is the raw arguments of the PM's forced submit tool call
    (``None`` if the PM did not call it). The host parses this into its own
    view model — the agent does not import any platform's data classes.
    ``verification_flag`` is set when the verifier stage ran but didn't call
    its tool — a flagged tool failure, not a cell-level error.
    """

    symbol: str
    submit_args: dict[str, Any] | None
    quant_report: str
    qual_report: str
    verification: dict[str, Any] | None
    verification_flag: str | None
    decision_md: str
    error: str | None
    calls: list[CallRecord]


@dataclass
class OptimizedAgent:
    """The standalone four-call specialist pipeline.

    Stateless across symbols: the host constructs one instance, injects an
    :class:`LLMClient` and an optional :class:`StageListener`, and calls
    :meth:`decide_one` per symbol. All platform coupling (evidence rendering,
    output schema, view parsing, nerve/trace bookkeeping) lives in the host
    adapter; this class only runs the LLM pipeline.
    """

    model: str = "xiaomi/mimo-v2.5-pro"
    temperature: float = 0.0
    # Specialists emit free-text reports (not a forced tool call), so a cap is
    # appropriate — but 6000 truncated CAT's qualitative report (lots of news +
    # web evidence) mid-stream -> finish_reason="length" -> ContextOverflow.
    # 16000 is generous for a "concise" report + key-evidence block while still
    # bounding a runaway specialist short of the 131k model max.
    specialist_max_tokens: int | None = 16000
    # The PM is uncapped (max_tokens omitted). It writes a reasoning preamble
    # before the forced submit_views call; any cap truncates that preamble
    # mid-stream -> finish_reason="length" with no tool call -> ContextOverflow
    # (this is what crashed AMZN/CAT at 6000 and again at 12000). Delorean's PM
    # ran uncapped for the same reason — reasoning tokens count against the
    # budget, so capping is dangerous. The forced tool call still bounds the
    # output *shape*; only its verbosity is unbounded.
    synthesis_max_tokens: int | None = None
    enable_verification: bool = True

    # Strategy-pack context (wired by the host; optional).
    mission_text: str = ""
    output_schema_text: str = ""

    # The PM's output contract. The host normally overrides ``submit_tool`` on
    # the call so the schema can list the exact symbol(s) being decided.
    submit_tool_name: str = "submit_views"
    submit_tool_description: str = "Submit your final answer. Call this exactly once."
    submit_tool_schema: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_SUBMIT_SCHEMA))

    # Host-injected dependencies.
    llm: LLMClient | None = None
    on_stage: StageListener | None = None

    def _emit_stage(self, name: str, symbol: str) -> None:
        if self.on_stage is not None:
            try:
                self.on_stage(name, symbol)
            except Exception:  # noqa: BLE001 - a listener must not break the pipeline
                logger.debug("on_stage listener raised", exc_info=True)

    def _call(
        self, *, stage: str, system: str, user: str,
        tools: tuple[dict, ...] = (), force_tool: str | None = None,
        max_tokens: int | None = None,
    ) -> CallRecord:
        if self.llm is None:
            raise RuntimeError(
                "OptimizedAgent.llm is not set; the host must inject an LLMClient"
            )
        started = datetime.now(UTC)
        t0 = time.monotonic()
        completion = self.llm.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tools=tools, force_tool=force_tool, max_tokens=max_tokens,
        )
        duration_ms = int((time.monotonic() - t0) * 1000)
        usage = getattr(completion, "usage", None)
        tokens_in = int(getattr(usage, "tokens_in", 0) or 0) if usage is not None else 0
        tokens_out = int(getattr(usage, "tokens_out", 0) or 0) if usage is not None else 0
        cost_usd = getattr(usage, "cost_usd", None) if usage is not None else None

        tool_args: dict[str, Any] | None = None
        if force_tool is not None and hasattr(completion, "call_named"):
            tc = completion.call_named(force_tool)
            if tc is not None:
                args = getattr(tc, "arguments", None)
                tool_args = args if isinstance(args, dict) else None

        return CallRecord(
            stage=stage,
            text=getattr(completion, "text", "") or "",
            finish_reason=getattr(completion, "finish_reason", None),
            model=getattr(completion, "model", None),
            tool_name=force_tool,
            tool_args=tool_args,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            started_at=started,
            duration_ms=duration_ms,
        )

    def decide_one(
        self, *, symbol: str, trade_date: str,
        quant_evidence: str, qual_evidence: str,
        submit_tool: dict | None = None,
        quant_kinds: tuple[str, ...] = (),
        qual_kinds: tuple[str, ...] = (),
    ) -> AgentResult:
        """Run specialists -> (verifier) -> final PM for one symbol.

        ``submit_tool`` overrides the instance's default submit tool spec, so a
        host can hand the PM a schema pinned to the exact symbol being decided.
        ``quant_kinds``/``qual_kinds`` list the data kinds actually present in
        each evidence pack, so the specialist user prompt declares what's in
        the pack rather than hardcoding a specific strategy's bindings.
        Returns the raw PM ``submit`` arguments; the host parses them.
        """
        calls: list[CallRecord] = []

        # 1 + 2: specialists in parallel (text reports, no tools). Emit both
        # stages up front (thread-safe) so a live dashboard tracks both while
        # they run.
        q_user = _quantitative_specialist_prompt(
            ticker=symbol, trade_date=trade_date, evidence=quant_evidence, kinds=quant_kinds,
        )
        l_user = _qualitative_specialist_prompt(
            ticker=symbol, trade_date=trade_date, evidence=qual_evidence, kinds=qual_kinds,
        )
        self._emit_stage("quantitative_specialist", symbol)
        self._emit_stage("qualitative_specialist", symbol)
        with ThreadPoolExecutor(max_workers=2) as ex:
            q_fut = ex.submit(
                self._call, stage="quantitative_specialist",
                system=QUANTITATIVE_SPECIALIST_SYSTEM, user=q_user,
                max_tokens=self.specialist_max_tokens,
            )
            l_fut = ex.submit(
                self._call, stage="qualitative_specialist",
                system=QUALITATIVE_SPECIALIST_SYSTEM, user=l_user,
                max_tokens=self.specialist_max_tokens,
            )
            q_call = q_fut.result()
            l_call = l_fut.result()
        calls.extend([q_call, l_call])
        quant_report = q_call.text
        qual_report = l_call.text

        # 3: independent verifier (optional) — structured submit_verification.
        verification_obj: dict[str, Any] | None = None
        verification_flag: str | None = None
        if self.enable_verification:
            self._emit_stage("independent_verifier", symbol)
            v_user = _verifier_prompt(
                ticker=symbol, trade_date=trade_date,
                quantitative_evidence=quant_evidence, qualitative_evidence=qual_evidence,
                quantitative_report=quant_report, qualitative_report=qual_report,
            )
            v_tool = _as_tool_spec(
                SUBMIT_VERIFICATION,
                "Submit your verification audit. Call this exactly once.",
                _VERIFICATION_SCHEMA,
            )
            v_call = self._call(
                stage="independent_verifier", system=VERIFIER_SYSTEM, user=v_user,
                tools=(v_tool,), force_tool=SUBMIT_VERIFICATION,
                max_tokens=self.synthesis_max_tokens,
            )
            calls.append(v_call)
            if v_call.tool_args is not None:
                verification_obj = v_call.tool_args
            else:
                # Verifier didn't call the tool — flag it, but don't kill
                # the cell. The PM still runs; it just gets a "n/a"
                # verification block instead of a real audit.
                verification_flag = (
                    f"verifier did not call {SUBMIT_VERIFICATION} "
                    f"(finish_reason={v_call.finish_reason!r})"
                )
                logger.warning("optimized: %s for %s", verification_flag, symbol)

        # 4: final PM — forced submit -> raw arguments (host parses).
        self._emit_stage("final_pm", symbol)
        pm_system = _PM_LEAD + (self.mission_text.strip() or _DEFAULT_MISSION) + "\n\n" + _PM_RULES
        pm_user = _final_pm_prompt(
            ticker=symbol, trade_date=trade_date,
            quantitative_report=quant_report, qualitative_report=qual_report,
            verification_obj=verification_obj, include_verification=self.enable_verification,
        )
        if self.output_schema_text.strip():
            pm_user += (
                "\n\n## Output schema (from the strategy pack)\n"
                + self.output_schema_text.strip()
            )
        if submit_tool is None:
            submit_tool = _as_tool_spec(
                self.submit_tool_name, self.submit_tool_description, self.submit_tool_schema
            )
        submit_name = submit_tool["function"]["name"]
        pm_call = self._call(
            stage="final_pm", system=pm_system, user=pm_user,
            tools=(submit_tool,), force_tool=submit_name,
            max_tokens=self.synthesis_max_tokens,
        )
        calls.append(pm_call)

        decision_md = pm_call.text
        error: str | None = None
        submit_args = pm_call.tool_args
        if submit_args is None:
            error = (
                f"PM did not call {submit_name} "
                f"(finish_reason={pm_call.finish_reason!r})"
            )

        return AgentResult(
            symbol=symbol,
            submit_args=submit_args,
            quant_report=quant_report,
            qual_report=qual_report,
            verification=verification_obj,
            verification_flag=verification_flag,
            decision_md=decision_md,
            error=error,
            calls=calls,
        )


__all__ = [
    "AgentResult",
    "CallRecord",
    "Completion",
    "LLMClient",
    "OptimizedAgent",
    "StageListener",
]



