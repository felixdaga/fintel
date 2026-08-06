"""Preflight: every reason a package cannot run, found before any money is spent.

A backtest that fails on the third decision date because a data source needs a
key that was never set, or a computed kind's upstream was never bound, has
already paid for two dates of LLM calls. Preflight runs once, returns *all*
findings rather than raising on the first, so one command tells you everything
to fix — the same shape as `catalog.check_bindings`, extended to the parts the
catalog cannot see: files, env, and the scoring KPI.

What preflight does **not** do is execute the package. It checks that the
declared world is resolvable; whether it actually returns data is a runtime
question, and answering it here would mean fetching, which is exactly the cost
preflight exists to avoid.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from fintel.market import catalog
from fintel.models.strategy import StrategyManifest, StrategyPaths


@dataclass(frozen=True)
class PreflightResult:
    """All findings, plus whether the package may run. `ok` is `not problems`."""

    ok: bool
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    required_env: list[str] = field(default_factory=list)

    def raise_if_not_ok(self) -> None:
        if not self.ok:
            raise PreflightError(self.problems)


class PreflightError(ValueError):
    """Raised by `raise_if_not_ok`. Carries the full finding list."""


def preflight(paths: StrategyPaths, *, env: dict[str, str] | None = None) -> PreflightResult:
    """Validate a loaded package against the platform. Never raises on findings.

    `env` defaults to `os.environ`; pass a dict in tests to assert without
    touching the real environment.
    """
    env = env if env is not None else dict(os.environ)
    manifest = paths.manifest
    problems: list[str] = []
    warnings: list[str] = []

    # 1. Data bindings vs catalog — the catalog's own check, plus env it needs.
    problems.extend(catalog.check_bindings(manifest.data))
    required = catalog.required_env(manifest.data)
    missing_env = [k for k in required if not env.get(k)]
    for k in missing_env:
        problems.append(f"env var {k!r} is required by a declared data source but not set")

    # 2. Files the manifest points at. The mission is the agent's instructions;
    #    a missing one is fatal. The output schema is optional metadata, so its
    #    absence is a warning, not a stop.
    if not paths.mission.is_file():
        problems.append(f"mission file {paths.mission.name!r} not found in package root")
    if not paths.output_schema.is_file():
        warnings.append(f"output schema {paths.output_schema.name!r} not found; schema-less run")

    # 3. Universe ref — resolvable by name shape, not by fetching.
    problems.extend(_check_universe(manifest))

    # 4. Schedule ref — known kind or a `module:Callable`.
    problems.extend(_check_schedule(manifest))

    # 5. Scoring KPI — builtin name or `module:Class`. We do not import it here:
    #    a KPI module with side effects at import time would make preflight
    #    non-hermetic. Format only.
    problems.extend(_check_kpi(manifest))

    return PreflightResult(
        ok=not problems,
        problems=problems,
        warnings=warnings,
        required_env=required,
    )


def _check_universe(manifest: StrategyManifest) -> list[str]:
    ref = manifest.universe
    if ref.symbols:
        if not all(isinstance(s, str) and s.strip() for s in ref.symbols):
            return ["universe.symbols contains an empty or non-string entry"]
        return []
    if ref.preset:
        names = {u.name for u in catalog.universes()}
        if ref.preset not in names and ":" not in ref.preset:
            return [f"unknown universe preset {ref.preset!r}; available: {sorted(names)}"]
        return []
    if ref.source and ":" in ref.source:
        return []
    if ref.source:
        return [f"universe.source {ref.source!r} is not a builtin preset and not {ref.source!r} a module:Callable"]
    return ["universe needs one of: preset, symbols, source"]


def _check_schedule(manifest: StrategyManifest) -> list[str]:
    from fintel.market.factory import SCHEDULES

    kind = manifest.decision.schedule.kind
    if kind in SCHEDULES or ":" in kind:
        return []
    return [
        f"unknown schedule kind {kind!r}; builtins: {sorted(SCHEDULES)}, or module:Class"
    ]


def _check_kpi(manifest: StrategyManifest) -> list[str]:
    """The scoring layer isn't built yet, so we can't validate a builtin KPI
    name against a registry. We check the shape only: a non-empty string that is
    either a `module:Class` (resolvable later) or a bare name (assumed builtin,
    validated against the scoring registry when that layer lands). A bare name
    that turns out not to be a builtin will fail at scoring build time, not here.
    """
    kpi = manifest.scoring.kpi
    if not kpi or not kpi.strip():
        return ["scoring.kpi is empty"]
    return []
