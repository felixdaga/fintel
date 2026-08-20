"""`strategy.toml` — the package's mission, its data needs, and how it's judged."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fintel.models.common import DecisionScope
from fintel.models.market import DataBinding, ScheduleRef, UniverseRef

MANIFEST_NAME = "strategy.toml"
LOCK_NAME = "strategy.lock"


class DecisionSpec(BaseModel):
    """`scope` decides what one agent invocation owns, hence the fan-out."""

    model_config = ConfigDict(extra="forbid")

    scope: DecisionScope = "single_name"
    schedule: ScheduleRef


class ScoringSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kpi: str  # builtin name or module:Class
    # The signal: how the decision's metrics become THE signal. Resolved by the
    # evaluation layer (`evaluate/signals.py`); the platform owns the mechanics
    # around it (transform, ensemble, holdings), the strategy owns what the
    # signal *is*. Defaults to the view's score (single-name).
    signal: str = "single_name"
    params: dict = Field(default_factory=dict)
    transform: str = "rank_range"
    horizons: list[int] = Field(default_factory=lambda: [1, 2, 3])
    metric_key: str = "icir"


class EvalSpec(BaseModel):
    """Agent-on-agent evaluation: a rating agent with current knowledge
    reviews finished run outputs.

    Declared in ``[eval]`` in ``strategy.toml``. When present, ``fintel report``
    deploys a minimal agent (the ``rating_agent``) with access only to the
    tools listed in ``rating_tools`` (no PIT cutoff — the rater has current
    knowledge) to rate specific output fields from each cell.

    The platform provides the plumbing (agent adapters, MCP, cell execution,
    invoke); the pack specifies *what* to rate (``rating_prompt_file``),
    *what shape* the rating takes (``rating_schema_file``), and *which agent*
    plays the rater. This mirrors the simulation contract: the pack owns the
    strategy, the platform owns the mechanics.
    """

    model_config = ConfigDict(extra="forbid")

    rating_agent: str = "llm"
    rating_agent_opt: dict = Field(default_factory=dict)
    # Tool kinds the rater gets. Must be a subset of the pack's declared data
    # kinds — the rater reuses the same MCP sources but without PIT cutoff.
    # Empty (default) = rater gets no tools, pure LLM judgment with its own
    # knowledge. Add kinds (e.g. ["event_timeline"]) to feed the rater extra
    # context — the full, un-clamped source is served (decision date is set
    # to today so the PIT cutoff doesn't clip historical data).
    rating_tools: list[str] = Field(default_factory=list)
    rating_prompt_file: str = "rating_prompt.md"
    rating_schema_file: str = "rating_schema.json"
    # Which run to rate. "best" = the first run (r1); "all" = every run;
    # "worst" = the last run. Default "best" keeps cost down for large K.
    rating_run: str = "best"


class AblationSpec(BaseModel):
    """Strategy-pack-controlled ablation knobs read by participating agents.

    Declared in ``[ablation]`` in ``strategy.toml``. Unlike ``[eval]`` (which
    runs post-hoc), this section changes what a participating agent *sees* during
    the simulation itself. The canonical use is the search-only ablation: feed
    an LLM agent a single predetermined web search instead of the full curated
    timeline + exploratory tools, to measure how much the curated surface
    mattered.

    The platform only carries the values; the agent adapter decides what to do
    with them. An agent that does not understand a knob ignores it (the
    ``llm`` agent reads ``search_query``; ``openclaw`` does not). This keeps the
    pack in control of *what* to feed without the platform knowing *how*.
    """

    model_config = ConfigDict(extra="forbid")

    # When set, the ``llm`` agent's pack channel runs this single web search
    # (via the existing ``web_search`` source, same PIT/cache path) and feeds
    # ONLY that result to the model — no curated timeline, no exploratory
    # tools. Empty (default) = normal pack-channel behaviour (render all
    # declared kinds). The query is strategy-owned, not agent-chosen.
    search_query: str = ""
    # Lookback window for the predetermined search (days). Defaults to the
    # ``web_search`` binding's lookback when 0/absent.
    search_lookback_days: int = 0


class StrategyManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    name: str
    description: str = ""

    universe: UniverseRef
    decision: DecisionSpec
    data: list[DataBinding] = Field(default_factory=list)
    scoring: ScoringSpec
    eval: EvalSpec | None = None
    ablation: AblationSpec | None = None

    mission_file: str = "mission.md"
    output_schema_file: str = "output_schema.json"
    # Optional standing thesis + dated research notes. Missing files are
    # empty, not an error — a pack without an alpha view is the default.
    alpha_view_file: str = "alpha_view.md"
    alpha_views_dir: str = "alpha_views"
    cache_dir: str = "cache"

    @model_validator(mode="after")
    def _unique_kinds(self) -> StrategyManifest:
        kinds = [b.kind for b in self.data]
        dupes = {k for k in kinds if kinds.count(k) > 1}
        if dupes:
            raise ValueError(f"duplicate data kinds: {sorted(dupes)}")
        return self

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(b.kind for b in self.data)


@dataclass(frozen=True)
class StrategyPaths:
    root: Path
    manifest: StrategyManifest

    @property
    def manifest_file(self) -> Path:
        return self.root / MANIFEST_NAME

    @property
    def mission(self) -> Path:
        return self.root / self.manifest.mission_file

    @property
    def alpha_view(self) -> Path:
        """Optional standing thesis (``alpha_view.md``). Missing = no view."""
        return self.root / self.manifest.alpha_view_file

    @property
    def alpha_views_dir(self) -> Path:
        """Optional dated notes (``alpha_views/YYYY-MM-DD.md``). Missing = none."""
        return self.root / self.manifest.alpha_views_dir

    @property
    def output_schema(self) -> Path:
        return self.root / self.manifest.output_schema_file

    @property
    def company_names(self) -> Path:
        """Optional universe-name fallback (JSON dict symbol→name). Loaded by
        the simulate layer and passed to agents that need it (e.g. the optimized
        agent's evidence config for web-search query naming)."""
        return self.root / "company_names.json"

    @property
    def cache_dir(self) -> Path:
        return self.root / self.manifest.cache_dir

    @property
    def lock(self) -> Path:
        return self.root / LOCK_NAME

    @property
    def rating_prompt(self) -> Path:
        return self.root / self.manifest.eval.rating_prompt_file

    @property
    def rating_schema(self) -> Path:
        return self.root / self.manifest.eval.rating_schema_file
