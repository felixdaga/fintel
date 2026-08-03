"""Artifact I/O. Atomic writes, so a reader never sees a half-written result.

Every level writes the same trio — config, lock, result — and a reader (the
report layer, a re-run, a human) must never observe a truncated file. The old
repo wrote results in place, so a crash mid-write left a result that parsed as
"empty" rather than "absent", which is a different and misleading failure.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, default=str, indent=2)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_model(path: Path, model: BaseModel) -> None:
    write_json(path, model.model_dump(mode="json"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_model(path: Path, cls: type[BaseModel]) -> BaseModel:
    return cls.model_validate(read_json(path))


def append_line(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, default=str) + "\n")
