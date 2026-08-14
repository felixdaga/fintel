"""Deploy config: one file that drives the full rebalance pipeline.

Live F1 config lives at ``f1_deploy/f1.toml`` (private). A research job
may still keep a colocated ``<job_dir>/deploy/*.toml``. Switching
strategy, agent, holdings rule, or capital = edit the live config, not
the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from pathlib import Path
from typing import Any

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass(frozen=True)
class HoldingsConfig:
    rule: str = "ew_long_threshold"
    threshold: float = 0.3
    cost_bps: float = 5.0
    active_budget: float = 0.5
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SiteConfig:
    """Public live-strategy page. Frozen across hops unless inception changes."""

    start: Date | None = None  # clip + rebase NAV/scores here (F1 went live 2026-04-24)


@dataclass(frozen=True)
class DeployConfig:
    job_id: str
    output_root: str = "runs"
    strategy_package: str = ""
    agent: str = ""
    model: str = ""
    k_repeats: int = 1
    holdings: HoldingsConfig = field(default_factory=HoldingsConfig)
    capital: float = 10000.0
    schedule_override: dict[str, Any] = field(default_factory=dict)
    site: SiteConfig = field(default_factory=SiteConfig)
    # Absolute path to the TOML; used to locate job_dir / deploy_dir.
    config_path: Path | None = None


def load_deploy_config(path: str | Path) -> DeployConfig:
    """Load a TOML deploy config file.

    Preferred live location: ``f1_deploy/f1.toml`` (explicit ``job_id``).
    Legacy: ``runs/<job_id>/deploy/f1.toml`` — ``job_id`` / ``output_root``
    may be omitted and inferred from the path.
    """
    path = Path(path).expanduser().resolve()
    with path.open("rb") as f:
        raw = tomllib.load(f)

    holdings_raw = raw.get("holdings", {})
    holdings = HoldingsConfig(
        rule=holdings_raw.get("rule", "ew_long_threshold"),
        threshold=float(holdings_raw.get("threshold", 0.3)),
        cost_bps=float(holdings_raw.get("cost_bps", 5.0)),
        active_budget=float(holdings_raw.get("active_budget", 0.5)),
        params={
            k: v
            for k, v in holdings_raw.items()
            if k not in ("rule", "threshold", "cost_bps", "active_budget")
        },
    )

    schedule_raw = raw.get("schedule", {})

    site_raw = raw.get("site", {})
    site = SiteConfig(start=_as_date(site_raw.get("start")))

    capital_raw = raw.get("capital", 10000.0)
    if isinstance(capital_raw, dict):
        capital_raw = capital_raw.get("starting", 10000.0)

    job_id, output_root = _resolve_job_location(path, raw)

    return DeployConfig(
        job_id=job_id,
        output_root=output_root,
        strategy_package=raw.get("strategy_package", ""),
        agent=raw.get("agent", ""),
        model=raw.get("model", ""),
        k_repeats=int(raw.get("k_repeats", 1)),
        holdings=holdings,
        capital=float(capital_raw),
        schedule_override=schedule_raw,
        site=site,
        config_path=path,
    )


def job_dir(config: DeployConfig) -> Path:
    """Research job directory (agent decisions), not the live fund home.

    Legacy colocated toml (``<job>/deploy/*.toml``) infers the job folder
    from the path. Live toml under ``f1_deploy/`` uses ``output_root/job_id``.
    """
    path = config.config_path
    if path is not None and path.parent.name == "deploy":
        job_folder = path.parent.parent
        if job_folder.name != "f1_deploy":
            return job_folder
    root = _resolve_output_root(config.output_root, path)
    return root / config.job_id


def deploy_dir(config: DeployConfig) -> Path:
    """Where live artifacts live (fund ledger, proposals, sleeve).

    The directory containing the toml. For a colocated research-job config
    that is ``<job>/deploy/``; for live F1 it is ``f1_deploy/``.
    """
    if config.config_path is not None:
        return config.config_path.parent
    return job_dir(config) / "deploy"


def _resolve_output_root(output_root: str, config_path: Path | None) -> Path:
    root = Path(output_root)
    if root.is_absolute():
        return root
    if config_path is not None:
        for p in [config_path.parent, *config_path.parents]:
            candidate = p / root
            if candidate.is_dir() and (p / "fintel").is_dir():
                return candidate
            if p.name == "f1_deploy" and (p.parent / root).is_dir():
                return p.parent / root
    return Path.cwd() / root


def _resolve_job_location(path: Path, raw: dict[str, Any]) -> tuple[str, str]:
    """Return (job_id, output_root). Infer only for ``<job>/deploy/*.toml``."""
    if path.parent.name == "deploy" and path.parent.parent.name != "f1_deploy":
        inferred_job = path.parent.parent.name
        inferred_root = str(path.parent.parent.parent)
        return (
            str(raw.get("job_id", inferred_job)),
            str(raw.get("output_root", inferred_root)),
        )
    if "job_id" not in raw:
        raise KeyError(
            f"deploy config {path} is missing job_id — live configs "
            f"(not under <job_dir>/deploy/) must set job_id explicitly"
        )
    return str(raw["job_id"]), str(raw.get("output_root", "runs"))


def cadence_label(schedule: dict[str, Any]) -> str:
    """Public cadence word from ``[schedule].kind``."""
    kind = str(schedule.get("kind") or "").lower()
    if "biweekly" in kind:
        return "biweekly"
    if "weekly" in kind:
        return "weekly"
    if "month" in kind:
        return "monthly"
    return kind or "weekly"


def _as_date(value: Any) -> Date | None:
    if value is None or value == "":
        return None
    if isinstance(value, Date):
        return value
    return Date.fromisoformat(str(value)[:10])
