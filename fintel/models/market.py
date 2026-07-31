"""References to the world a job runs against.

Each is `kind`/`source` (resolved by the matching factory) plus free-form
extras collected as `params`, so a new universe/schedule/source needs no
schema change here.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator


class _Spec(BaseModel):
    model_config = ConfigDict(extra="allow")

    @property
    def params(self) -> dict:
        return dict(self.model_extra or {})


class UniverseRef(_Spec):
    preset: str | None = None
    symbols: list[str] | None = None
    source: str | None = None  # module:Class for a custom universe

    @model_validator(mode="after")
    def _one_of(self) -> UniverseRef:
        if not (self.preset or self.symbols or self.source):
            raise ValueError("universe needs one of: preset, symbols, source")
        return self


class ScheduleRef(_Spec):
    kind: str  # factory name (single_point | quarterly | custom_dates) or module:Class


class DataBinding(_Spec):
    """One declared data kind and how to serve it.

    `kind` is what the agent asks for. `source` is a builtin name or
    `module:Class` — a package supplying its own source gets the platform's
    caching, PIT clamp, tool, and evidence rendering for free.
    """

    kind: str
    source: str
