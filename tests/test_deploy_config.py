"""Live vs colocated deploy config: job_dir vs deploy_dir."""

from __future__ import annotations

from pathlib import Path

from fintel.deploy.config import deploy_dir, job_dir, load_deploy_config


def test_live_toml_uses_explicit_job_and_toml_folder(tmp_path: Path):
    (tmp_path / "fintel").mkdir()
    (tmp_path / "runs" / "job-b").mkdir(parents=True)
    live = tmp_path / "f1_deploy"
    live.mkdir()
    (live / "f1.toml").write_text(
        'job_id = "job-b"\noutput_root = "runs"\n'
        "[holdings]\nrule = \"score_weighted_long\"\nthreshold = 0.0\n"
    )
    cfg = load_deploy_config(live / "f1.toml")
    assert cfg.job_id == "job-b"
    assert cfg.holdings.rule == "score_weighted_long"
    assert job_dir(cfg) == tmp_path / "runs" / "job-b"
    assert deploy_dir(cfg) == live


def test_site_start_and_cadence_label(tmp_path: Path):
    from datetime import date as Date

    from fintel.deploy.config import cadence_label

    (tmp_path / "fintel").mkdir()
    (tmp_path / "runs" / "job-b").mkdir(parents=True)
    live = tmp_path / "f1_deploy"
    live.mkdir()
    (live / "f1.toml").write_text(
        'job_id = "job-b"\noutput_root = "runs"\n'
        "[schedule]\nkind = \"biweekly_fridays\"\n"
        "[site]\nstart = \"2026-04-24\"\n"
    )
    cfg = load_deploy_config(live / "f1.toml")
    assert cfg.site.start == Date(2026, 4, 24)
    assert cadence_label(cfg.schedule_override) == "biweekly"


def test_colocated_toml_still_infers_job_folder(tmp_path: Path):
    job = tmp_path / "runs" / "job-w"
    (job / "deploy").mkdir(parents=True)
    (job / "deploy" / "f1.toml").write_text("[holdings]\nrule = \"ew_long_threshold\"\n")
    cfg = load_deploy_config(job / "deploy" / "f1.toml")
    assert cfg.job_id == "job-w"
    assert job_dir(cfg) == job
    assert deploy_dir(cfg) == job / "deploy"
