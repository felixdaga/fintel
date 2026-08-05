"""Secrets bootstrap reads fintel's local ``.env/`` directory."""

from __future__ import annotations

import os
from pathlib import Path

from fintel.utils.secrets import bootstrap_env, load_dotenv, load_env_dir


def test_load_dotenv_parses_keys_and_comments(tmp_path: Path):
    path = tmp_path / "keys.env"
    path.write_text(
        "# comment\n"
        "OPENROUTER_API_KEY=sk-test\n"
        "export MASSIVE_API_KEY='mass-1'\n"
        "BRAVE_API_KEY=\"brave-2\"\n"
        "\n"
        "IGNORED\n",
        encoding="utf-8",
    )
    assert load_dotenv(path) == {
        "OPENROUTER_API_KEY": "sk-test",
        "MASSIVE_API_KEY": "mass-1",
        "BRAVE_API_KEY": "brave-2",
    }


def test_load_env_dir_merges_and_keys_env_wins(tmp_path: Path):
    (tmp_path / "extra.env").write_text("FOO=from-extra\nBAR=1\n", encoding="utf-8")
    (tmp_path / "keys.env").write_text("FOO=from-keys\nBAZ=2\n", encoding="utf-8")
    assert load_env_dir(tmp_path) == {"FOO": "from-keys", "BAR": "1", "BAZ": "2"}


def test_bootstrap_env_sets_missing_only(tmp_path: Path, monkeypatch):
    (tmp_path / "keys.env").write_text(
        "OPENROUTER_API_KEY=from-file\nMASSIVE_API_KEY=mass\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("MASSIVE_API_KEY", "already")
    populated = bootstrap_env(env_dir=tmp_path)
    assert populated == {"OPENROUTER_API_KEY": "from-file"}
    assert os.environ["OPENROUTER_API_KEY"] == "from-file"
    assert os.environ["MASSIVE_API_KEY"] == "already"
