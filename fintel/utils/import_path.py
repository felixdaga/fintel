"""`module.path:Name` resolution — shared by every factory."""

from __future__ import annotations

import importlib
from typing import Any

FORMAT = "module.path:Name"


def resolve(target: str) -> Any:
    if ":" not in target:
        raise ValueError(f"expected {FORMAT!r}, got {target!r}")
    module_name, _, attr = target.partition(":")
    if not module_name or not attr:
        raise ValueError(f"expected {FORMAT!r}, got {target!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"cannot import {module_name!r} from {target!r}: {exc}") from exc
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(f"{module_name!r} has no attribute {attr!r}") from exc


def build(target: str, builtins: dict[str, str], *, label: str, **kwargs: Any) -> Any:
    """Resolve `target` as a builtin name or an import path, then call it.

    The one resolution rule, used by every `factory.py`.
    """
    spec = builtins.get(target, target)
    if ":" not in spec:
        raise ValueError(
            f"unknown {label} {target!r}; expected one of {sorted(builtins)} or {FORMAT!r}"
        )
    return resolve(spec)(**kwargs)
