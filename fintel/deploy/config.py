"""Deploy config: one file that drives the full rebalance pipeline.

The TOML lives with the run at ``<job_dir>/deploy/f1.toml``. Pipeline
scripts live in ``scripts/``. Switching strategy, agent, holdings rule,
or capital = edit the config, not the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    # Absolute path to the TOML; used to locate the job dir when the config
    # lives at <job_dir>/deploy/*.toml
    config_path: Path | None = None


def load_deploy_config(path: str | Path) -> DeployConfig:
    """Load a TOML deploy config file.

    Preferred location: ``runs/<job_id>/deploy/f1.toml``. When the config
    sits under a ``deploy/`` folder, ``job_id`` / ``output_root`` may be
    omitted and are inferred from the path.
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
        config_path=path,
    )


def job_dir(config: DeployConfig) -> Path:
    """Job directory for this deploy config.

    If the TOML lives at ``<job_dir>/deploy/*.toml``, that parent wins.
    Otherwise falls back to ``<output_root>/<job_id>``.
    """
    if config.config_path is not None and config.config_path.parent.name == "deploy":
        return config.config_path.parent.parent
    return Path(config.output_root) / config.job_id


def _resolve_job_location(path: Path, raw: dict[str, Any]) -> tuple[str, str]:
    """Return (job_id, output_root), preferring path inference when under deploy/."""
    if path.parent.name == "deploy":
        inferred_job = path.parent.parent.name
        inferred_root = str(path.parent.parent.parent)
        return (
            str(raw.get("job_id", inferred_job)),
            str(raw.get("output_root", inferred_root)),
        )
    if "job_id" not in raw:
        raise KeyError(
            f"deploy config {path} is missing job_id and is not under "
            f"<job_dir>/deploy/"
        )
    return str(raw["job_id"]), str(raw.get("output_root", "runs"))
