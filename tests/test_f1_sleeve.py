"""F1 sleeve screen: never target-only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

f1_deploy = pytest.importorskip("f1_deploy")

from f1_deploy.fund import f1_sleeve, held_symbols, load_sleeve  # noqa: E402


def test_sleeve_unions_allowlist_held_and_targets(tmp_path: Path):
    records = tmp_path / "fund" / "records"
    records.mkdir(parents=True)
    (records / "sleeve.json").write_text(json.dumps({"symbols": ["AAPL", "MSFT"]}) + "\n")
    (records / "positions.json").write_text(
        json.dumps({"positions": [{"symbol": "CRM"}, {"symbol": "GS"}]}) + "\n"
    )
    got = f1_sleeve(tmp_path, target_symbols=["NVDA", "MSFT"])
    assert got == {"AAPL", "MSFT", "CRM", "GS", "NVDA"}
    assert "CRM" in held_symbols(tmp_path)
    assert load_sleeve(tmp_path) == {"AAPL", "MSFT"}


def test_missing_sleeve_file_is_empty_not_whole_djia(tmp_path: Path):
    assert load_sleeve(tmp_path) == set()
    got = f1_sleeve(tmp_path, target_symbols=["NVDA"])
    assert got == {"NVDA"}
