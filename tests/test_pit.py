from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from fintel.pit import Cutoff

CUT = Cutoff(date(2022, 7, 1))


def test_strictly_before_the_decision_date():
    assert CUT.allows("2022-06-30")
    assert CUT.allows("2022-06-30 23:59:59")
    assert not CUT.allows("2022-07-01")
    assert not CUT.allows("2022-07-01 00:00:01")
    assert not CUT.allows("2022-07-02")


def test_undated_is_never_allowed():
    assert not CUT.allows(None)
    assert not CUT.allows("")
    assert not CUT.allows("not a date")
    assert not CUT.allows(pd.NaT)


def test_tz_aware_is_compared_in_utc():
    # 2022-07-01 09:00 +10:00 is 2022-06-30 23:00 UTC — still historical.
    assert CUT.allows(pd.Timestamp("2022-07-01 09:00", tz="Australia/Sydney"))
    assert not CUT.allows(pd.Timestamp("2022-07-01 12:00", tz="UTC"))


def test_datetime_decision_date_is_narrowed_to_a_date():
    assert Cutoff(datetime(2022, 7, 1, 15, 30)).decision_date == date(2022, 7, 1)


def test_rejects_non_date():
    with pytest.raises(TypeError):
        Cutoff("2022-07-01")  # type: ignore[arg-type]


# ── frames ───────────────────────────────────────────────────────────────────


def test_clamp_frame_on_index():
    df = pd.DataFrame(
        {"close": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(["2022-06-29", "2022-06-30", "2022-07-01"]),
    )
    out = CUT.clamp_frame(df)
    assert list(out["close"]) == [1.0, 2.0]


def test_clamp_frame_on_named_column():
    df = pd.DataFrame({"filed": ["2022-05-15", "2022-08-15"], "eps": [1.0, 2.0]})
    out = CUT.clamp_frame(df, "filed")
    assert list(out["eps"]) == [1.0]


def test_publication_lag_beats_period_end():
    """A quarter that ended before the decision but was filed after is invisible."""
    df = pd.DataFrame(
        {"period_end": ["2022-03-31", "2022-06-30"], "filed": ["2022-05-15", "2022-08-12"]}
    )
    assert len(CUT.clamp_frame(df, "filed")) == 1
    # Clamping on the wrong column is the classic leak, and it shows up here.
    assert len(CUT.clamp_frame(df, "period_end")) == 2


def test_clamp_frame_drops_unparseable_dates():
    df = pd.DataFrame({"d": ["2022-06-01", "garbage", None], "v": [1, 2, 3]})
    assert list(CUT.clamp_frame(df, "d")["v"]) == [1]


def test_clamp_frame_empty_and_missing_column():
    assert CUT.clamp_frame(pd.DataFrame()).empty
    with pytest.raises(KeyError):
        CUT.clamp_frame(pd.DataFrame({"a": [1]}), "missing")


def test_clamp_frame_does_not_mutate_input():
    df = pd.DataFrame({"d": ["2022-06-01", "2022-08-01"], "v": [1, 2]})
    CUT.clamp_frame(df, "d")
    assert len(df) == 2


# ── records ──────────────────────────────────────────────────────────────────


def test_clamp_records_filters_and_orders():
    news = [
        {"published": "2022-07-05", "id": "future"},
        {"published": "2022-06-20", "id": "b"},
        {"published": "2022-01-04", "id": "a"},
        {"published": None, "id": "undated"},
    ]
    assert [r["id"] for r in CUT.clamp_records(news, "published")] == ["a", "b"]


def test_violations_names_what_leaked():
    news = [{"published": "2022-06-20", "id": "ok"}, {"published": "2022-07-09", "id": "leak"}]
    assert [r["id"] for r in CUT.violations(news, "published")] == ["leak"]
