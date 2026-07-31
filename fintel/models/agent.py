from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelSpec(BaseModel):
    """The 'who is playing' freeze. All-None means leave the agent's own
    defaults alone, which makes the run honestly non-replicable rather than
    silently so."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    provider: list[str] | None = None
    allow_fallbacks: bool = False
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    thinking: str | None = None
    extra_params: dict[str, Any] = Field(default_factory=dict)

    def to_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for key in ("temperature", "top_p", "seed"):
            val = getattr(self, key)
            if val is not None:
                params[key] = val
        if self.provider is not None:
            params["provider"] = {
                "order": list(self.provider),
                "allow_fallbacks": self.allow_fallbacks,
            }
        params.update(self.extra_params)
        return params

    def is_noop(self) -> bool:
        return not self.to_params() and self.id is None and self.thinking is None


class AgentSpec(BaseModel):
    """Which agent plays, and how it's frozen for one job.

    `options` is opaque to the platform — it comes from `--agent-opt k=v` and is
    interpreted only by that agent's adapter, which declares what it accepts.
    """

    model_config = ConfigDict(extra="forbid")

    name: str  # builtin name or module:factory_fn
    model: ModelSpec = Field(default_factory=ModelSpec)
    timeout_seconds: int | None = None
    allow_sub_agents: bool = False
    options: dict[str, Any] = Field(default_factory=dict)
