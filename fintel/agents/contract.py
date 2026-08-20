"""The pack surface the platform injects into every adapter.

``PACK_CONTEXT_FIELDS`` is the central list: conformance
(``test_every_adapter_accepts_pack_context``) and ``fintel check`` both
read it. A new strategy-pack field that adapters must accept is added
here, declared on every builtin, and forwarded by multi-call agents that
spawn sub-agents.

``OPTIONAL_INJECT_FIELDS`` are kwargs the platform may also pass, but
only when the pack actually has the corresponding content (and, for
ablation, only to agents that understand the knob).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

PACK_CONTEXT_FIELDS: tuple[str, ...] = (
    "mission_text",
    "output_schema_text",
    "alpha_view_text",
)

# Injected when the pack has the matching content. ``search_query`` is
# currently wired only into the ``llm`` agent (see simulate/job.py).
OPTIONAL_INJECT_FIELDS: tuple[str, ...] = (
    "company_names",
    "search_query",
    "search_lookback_days",
)

# Ablation knobs the platform injects only for this agent name.
ABLATION_AGENT = "llm"


def init_param_names(cls: type) -> frozenset[str] | None:
    """Constructor kwargs an adapter will accept.

    Returns ``None`` when the constructor takes ``**kwargs`` (accepts anything).
    """
    if is_dataclass(cls):
        return frozenset(f.name for f in fields(cls) if f.init and not f.name.startswith("_"))
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return frozenset()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return None
    return frozenset(name for name in sig.parameters if name != "self")


@dataclass(frozen=True)
class AgentSurface:
    """What one adapter declares about pack context, without invoking it."""

    name: str
    known: bool
    pit_enforcement: str | None
    subagents: bool
    accepts: dict[str, bool]
    preflight: tuple[str, ...]
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "known": self.known,
            "pit_enforcement": self.pit_enforcement,
            "subagents": self.subagents,
            "accepts": dict(self.accepts),
            "preflight": list(self.preflight),
            "issues": list(self.issues),
        }


def inspect_agent(name: str, *, options: dict[str, Any] | None = None) -> AgentSurface:
    """Resolve an adapter class and report pack-context coverage.

    Does not call ``decide``. Agent ``preflight_checks`` may still flag a
    missing API key or binary — that is the point of the report.
    """
    from fintel.agents.factory import AGENTS, preflight
    from fintel.utils.import_path import resolve

    target = AGENTS.get(name, name if ":" in name else None)
    if target is None:
        return AgentSurface(
            name=name,
            known=False,
            pit_enforcement=None,
            subagents=False,
            accepts={f: False for f in PACK_CONTEXT_FIELDS + OPTIONAL_INJECT_FIELDS},
            preflight=(),
            issues=(f"unknown agent {name!r}",),
        )

    cls = resolve(target)
    params = init_param_names(cls)
    wanted = PACK_CONTEXT_FIELDS + OPTIONAL_INJECT_FIELDS
    if params is None:
        accepts = {field: True for field in wanted}
    else:
        accepts = {field: field in params for field in wanted}
    pit = getattr(cls, "pit_enforcement", None)
    pit_s = str(pit) if pit is not None else None
    subagents = bool(getattr(cls, "subagents", False))

    issues: list[str] = []
    missing_required = [f for f in PACK_CONTEXT_FIELDS if not accepts[f]]
    if missing_required:
        issues.append(
            f"does not declare {missing_required}; "
            "conformance requires every adapter to accept the pack-context fields"
        )
    if subagents and not accepts.get("alpha_view_text"):
        issues.append(
            "declares sub-agents but does not accept alpha_view_text; "
            "specialists that never see mission.md will miss the thesis"
        )

    pf = tuple(preflight(name, **(options or {})))
    return AgentSurface(
        name=name,
        known=True,
        pit_enforcement=pit_s,
        subagents=subagents,
        accepts=accepts,
        preflight=pf,
        issues=tuple(issues),
    )
