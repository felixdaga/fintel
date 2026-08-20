"""What a strategy pack wires vs what it leaves at platform defaults.

``PACK_FEATURES`` is the central list of optional pack surfaces. ``fintel check``
and its tests iterate it so a new strategy feature cannot land in docs-only.
Required files (mission) live here too so the report is one table, not two.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fintel.models.strategy import StrategyManifest, StrategyPaths
from fintel.strategy.load import load
from fintel.strategy.preflight import preflight
from fintel.strategy.views import AlphaViewLibrary

Status = Literal["wired", "default", "missing"]


@dataclass(frozen=True)
class PackFeature:
    """One optional (or required) pack surface ``fintel check`` knows about."""

    id: str
    kind: Literal["file", "dir", "section"]
    required: bool
    # StrategyPaths attribute for file/dir features.
    path_attr: str = ""
    # Top-level strategy.toml key for section features.
    toml_key: str = ""
    label: str = ""


# Adding a pack feature: declare it here, load it in strategy/, inject it in
# simulate, accept it on adapters (see fintel.agents.contract.PACK_CONTEXT_FIELDS).
PACK_FEATURES: tuple[PackFeature, ...] = (
    PackFeature(
        "mission",
        "file",
        required=True,
        path_attr="mission",
        label="scoring rubric",
    ),
    PackFeature(
        "output_schema",
        "file",
        required=False,
        path_attr="output_schema",
        label="View contract",
    ),
    PackFeature(
        "alpha_view",
        "file",
        required=False,
        path_attr="alpha_view",
        label="standing thesis",
    ),
    PackFeature(
        "alpha_views",
        "dir",
        required=False,
        path_attr="alpha_views_dir",
        label="dated research notes (PIT)",
    ),
    PackFeature(
        "company_names",
        "file",
        required=False,
        path_attr="company_names",
        label="display names for evidence",
    ),
    PackFeature(
        "eval",
        "section",
        required=False,
        toml_key="eval",
        label="agent-on-agent rating",
    ),
    PackFeature(
        "ablation",
        "section",
        required=False,
        toml_key="ablation",
        label="in-sim ablation knobs",
    ),
)


@dataclass(frozen=True)
class FeatureState:
    id: str
    status: Status
    detail: str
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "detail": self.detail,
            "label": self.label,
        }


@dataclass(frozen=True)
class PackReport:
    name: str
    path: str
    features: tuple[FeatureState, ...]
    kinds: tuple[str, ...]
    scoring_defaults: tuple[FeatureState, ...]
    problems: tuple[str, ...]
    warnings: tuple[str, ...]
    alpha_view_notes: tuple[str, ...]
    ablation_search_query: str
    has_company_names: bool
    has_alpha_view: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "features": [f.to_dict() for f in self.features],
            "kinds": list(self.kinds),
            "scoring_defaults": [f.to_dict() for f in self.scoring_defaults],
            "problems": list(self.problems),
            "warnings": list(self.warnings),
            "alpha_view_notes": list(self.alpha_view_notes),
            "ablation_search_query": self.ablation_search_query,
            "has_company_names": self.has_company_names,
            "has_alpha_view": self.has_alpha_view,
        }


_SCORING_OPTIONAL = ("signal", "transform", "horizons", "metric_key")


def inspect_pack(
    package_dir: str | Path,
    *,
    env: dict[str, str] | None = None,
) -> PackReport:
    """Load a pack and report wired vs default surfaces. Never writes a lock."""
    paths = load(package_dir)
    raw = tomllib.loads(paths.manifest_file.read_text())
    pf = preflight(paths, env=env)
    library = AlphaViewLibrary.load(paths)

    features = tuple(_feature_state(feat, paths, raw, library) for feat in PACK_FEATURES)
    notes = tuple(n.as_of.isoformat() for n in library.notes)
    has_alpha = bool(library.standing.strip() or library.notes)
    has_names = paths.company_names.is_file()
    query = ""
    if paths.manifest.ablation is not None:
        query = paths.manifest.ablation.search_query.strip()

    return PackReport(
        name=paths.manifest.name,
        path=str(paths.root),
        features=features,
        kinds=paths.manifest.kinds,
        scoring_defaults=_scoring_defaults(raw, paths.manifest),
        problems=tuple(pf.problems),
        warnings=tuple(pf.warnings),
        alpha_view_notes=notes,
        ablation_search_query=query,
        has_company_names=has_names,
        has_alpha_view=has_alpha,
    )


def _feature_state(
    feat: PackFeature,
    paths: StrategyPaths,
    raw: dict[str, Any],
    library: AlphaViewLibrary,
) -> FeatureState:
    if feat.kind == "file":
        path: Path = getattr(paths, feat.path_attr)
        if path.is_file():
            size = path.stat().st_size
            return FeatureState(feat.id, "wired", f"{path.name} ({size} bytes)", feat.label)
        if feat.required:
            return FeatureState(feat.id, "missing", f"{path.name} not found", feat.label)
        return FeatureState(feat.id, "default", f"{path.name} omitted", feat.label)

    if feat.kind == "dir":
        path = getattr(paths, feat.path_attr)
        n = len(library.notes)
        if n:
            dates = ", ".join(note.as_of.isoformat() for note in library.notes)
            return FeatureState(feat.id, "wired", f"{n} dated note(s): {dates}", feat.label)
        return FeatureState(feat.id, "default", "no dated notes", feat.label)

    # section
    if feat.toml_key in raw:
        extra = _section_detail(feat.id, paths)
        return FeatureState(feat.id, "wired", extra or f"[{feat.toml_key}] present", feat.label)
    return FeatureState(feat.id, "default", f"[{feat.toml_key}] omitted", feat.label)


def _section_detail(feat_id: str, paths: StrategyPaths) -> str:
    manifest = paths.manifest
    if feat_id == "eval" and manifest.eval is not None:
        prompt = paths.rating_prompt
        schema = paths.rating_schema
        bits = [f"rating_agent={manifest.eval.rating_agent}"]
        if not prompt.is_file():
            bits.append(f"missing {prompt.name}")
        if not schema.is_file():
            bits.append(f"missing {schema.name}")
        return "; ".join(bits)
    if feat_id == "ablation" and manifest.ablation is not None:
        q = manifest.ablation.search_query.strip()
        if q:
            return f"search_query set ({len(q)} chars)"
        return "search_query empty (section present, knob unused)"
    return ""


def _scoring_defaults(raw: dict[str, Any], manifest: StrategyManifest) -> tuple[FeatureState, ...]:
    scoring_raw = raw.get("scoring") if isinstance(raw.get("scoring"), dict) else {}
    out: list[FeatureState] = []
    fields = type(manifest.scoring).model_fields
    for name in _SCORING_OPTIONAL:
        field = fields[name]
        value = getattr(manifest.scoring, name)
        if name in scoring_raw:
            out.append(FeatureState(f"scoring.{name}", "wired", repr(value), name))
        else:
            default = field.get_default(call_default_factory=True)
            out.append(FeatureState(f"scoring.{name}", "default", f"default {default!r}", name))
    return tuple(out)
