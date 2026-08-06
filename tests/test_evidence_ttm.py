"""TTM labeling must require contiguous quarters (AAPL smoke failure mode)."""

from __future__ import annotations

from fintel.agents.evidence import EvidenceConfig, FintelEvidence, quarters_contiguous


def _q(period_end: str, fiscal_period: str, **nums: float) -> dict:
    return {
        "timeframe": "quarterly",
        "period_end": period_end,
        "fiscal_period": fiscal_period,
        "filing_date": period_end,
        **nums,
    }


def test_quarters_contiguous_by_fiscal_period():
    ok, detail = quarters_contiguous(
        [
            _q("2025-12-27", "Q1"),
            _q("2025-09-27", "Q4"),
            _q("2025-06-28", "Q3"),
            _q("2025-03-29", "Q2"),
        ]
    )
    assert ok
    assert "Q1→Q4→Q3→Q2" == detail


def test_quarters_contiguous_detects_missing_q4():
    """Newest-available four with a skipped Q4 must not count as TTM."""
    ok, detail = quarters_contiguous(
        [
            _q("2025-12-27", "Q1"),
            _q("2025-06-28", "Q3"),  # Q4 missing
            _q("2025-03-29", "Q2"),
            _q("2024-12-28", "Q1"),
        ]
    )
    assert not ok
    assert "fiscal gap" in detail


class _Policy:
    lookback_cap_map = {"fundamentals": 720}


class _Access:
    policy = _Policy()

    def read(self, kind: str, **_: object) -> object:  # pragma: no cover
        raise AssertionError(f"unexpected read {kind}")


def _evidence() -> FintelEvidence:
    from datetime import date

    return FintelEvidence(
        symbol="AAPL",
        decision_date=date(2026, 4, 24),
        access=_Access(),
        config=EvidenceConfig(),
    )


def test_growth_block_refuses_gapped_ttm_sum():
    """Do not print a summed TTM when the last 4 available quarters have a gap."""
    panel = [
        _q(
            "2025-12-27",
            "Q1",
            revenue=143.0,
            net_income=42.0,
            operating_income=50.0,
            gross_profit=69.0,
            operating_cash_flow=54.0,
            free_cash_flow=None,
            eps_diluted=2.84,
        ),
        _q(
            "2025-06-28",
            "Q3",
            revenue=94.0,
            net_income=23.0,
            operating_income=28.0,
            gross_profit=44.0,
            operating_cash_flow=28.0,
            eps_diluted=1.5,
        ),
        _q(
            "2025-03-29",
            "Q2",
            revenue=95.0,
            net_income=24.0,
            operating_income=29.0,
            gross_profit=45.0,
            operating_cash_flow=24.0,
            eps_diluted=1.6,
        ),
        _q(
            "2024-12-28",
            "Q1",
            revenue=124.0,
            net_income=36.0,
            operating_income=43.0,
            gross_profit=58.0,
            operating_cash_flow=30.0,
            eps_diluted=2.4,
        ),
    ]
    text = _evidence()._growth_block(panel)
    assert "TTM: incomplete" in text
    assert "contiguous" in text
    assert "457" not in text  # no false summed TTM
    assert "operating_cash_flow" in text  # QoQ/YoY still surface OCF


def test_growth_block_emits_contiguous_ttm():
    panel = [
        _q(
            "2025-12-27",
            "Q1",
            revenue=100.0,
            net_income=10.0,
            operating_income=12.0,
            gross_profit=40.0,
            operating_cash_flow=15.0,
            free_cash_flow=11.0,
            eps_diluted=1.0,
        ),
        _q(
            "2025-09-27",
            "Q4",
            revenue=90.0,
            net_income=9.0,
            operating_income=11.0,
            gross_profit=36.0,
            operating_cash_flow=14.0,
            free_cash_flow=10.0,
            eps_diluted=0.9,
        ),
        _q(
            "2025-06-28",
            "Q3",
            revenue=80.0,
            net_income=8.0,
            operating_income=10.0,
            gross_profit=32.0,
            operating_cash_flow=13.0,
            free_cash_flow=9.0,
            eps_diluted=0.8,
        ),
        _q(
            "2025-03-29",
            "Q2",
            revenue=70.0,
            net_income=7.0,
            operating_income=9.0,
            gross_profit=28.0,
            operating_cash_flow=12.0,
            free_cash_flow=8.0,
            eps_diluted=0.7,
        ),
    ]
    text = _evidence()._growth_block(panel)
    assert "TTM (sum of 4 contiguous quarters" in text
    assert "revenue=340" in text
    assert "operating_cash_flow=54" in text
