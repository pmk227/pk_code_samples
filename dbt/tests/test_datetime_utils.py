import math
import datetime as dt
import pandas as pd
import pytest

from src.utilities.datetime_utils import (
    convert_datetime_to_dateid,
    convert_strdate_to_dateid,
    convert_date_id_to_date,
    convert_rel_date_to_date_id,
    convert_date_id_to_target_type,
    to_datetime_utc,
    Interval,
    compute_date_slices,
)

# ------------------------
# Conversion helpers
# ------------------------

def test_convert_datetime_to_dateid_scalar_date():
    d = dt.date(2025, 8, 15)
    assert convert_datetime_to_dateid(d) == 20250815


def test_convert_datetime_to_dateid_scalar_datetime():
    d = dt.datetime(2023, 1, 2, 12, 30, 0)
    assert convert_datetime_to_dateid(d) == 20230102


def test_convert_datetime_to_dateid_series():
    s = pd.to_datetime(pd.Series(["2024-01-01", "2024-12-31"]))
    out = convert_datetime_to_dateid(s)
    assert isinstance(out, pd.Series)
    assert out.tolist() == [20240101, 20241231]


def test_convert_strdate_to_dateid_scalar_default_pattern():
    assert convert_strdate_to_dateid("2024-07-04") == 20240704


def test_convert_strdate_to_dateid_scalar_custom_pattern():
    assert convert_strdate_to_dateid("07/04/2024", pattern="%m/%d/%Y") == 20240704


def test_convert_strdate_to_dateid_series():
    s = pd.Series(["2023-03-01", "2023-03-02"])
    out = convert_strdate_to_dateid(s)
    assert out.tolist() == [20230301, 20230302]


def test_convert_date_id_to_date_roundtrip():
    did = 19991231
    d = convert_date_id_to_date(did)
    assert d == dt.date(1999, 12, 31)


def test_convert_rel_date_to_date_id_passthrough_when_already_date_id():
    assert convert_rel_date_to_date_id(20250817) == 20250817


def test_convert_rel_date_to_date_id_relative_zero_is_today():
    today = dt.date.today()
    expected = today.year * 10000 + today.month * 100 + today.day
    assert convert_rel_date_to_date_id(0) == expected


def test_convert_date_id_to_target_type_date_id_passthrough():
    assert convert_date_id_to_target_type(20240131, "date_id") == 20240131


def test_convert_date_id_to_target_type_strftime():
    assert convert_date_id_to_target_type(20240131, "%Y-%m-%d") == "2024-01-31"


def test_convert_date_id_to_target_type_unix_variants():
    did = 19700102  # Jan 2, 1970
    u = convert_date_id_to_target_type(did, "unix")
    assert isinstance(u, int) and u >= 86400  # ≥ one day after epoch
    assert convert_date_id_to_target_type(did, "unix_ms") == u * 1000
    assert convert_date_id_to_target_type(did, "unix_us") == u * 1_000_000
    assert convert_date_id_to_target_type(did, "unix_ns") == u * 1_000_000_000


def test_to_datetime_utc_from_string_and_datetime():
    # from string
    s = to_datetime_utc("2024-06-01")
    assert s.tzinfo is not None and s.utcoffset() == dt.timedelta(0)
    assert (s.year, s.month, s.day) == (2024, 6, 1)

    # from naive datetime -> assume UTC
    d0 = dt.datetime(2024, 6, 1, 12, 0, 0)
    d1 = to_datetime_utc(d0)
    assert d1.tzinfo is not None and d1.utcoffset() == dt.timedelta(0)
    assert d1.hour == 12

    # from aware datetime -> converted to UTC
    aware = dt.datetime(2024, 6, 1, 7, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=-5)))
    d2 = to_datetime_utc(aware)
    assert d2.hour == 12 and d2.utcoffset() == dt.timedelta(0)


def test_convert_date_id_to_target_type_invalid_unix_spec_raises():
    with pytest.raises(ValueError):
        convert_date_id_to_target_type(20240101, "unix_min")


# ------------------------
# Interval parsing (parameterized) & slicing
# ------------------------

@pytest.mark.parametrize(
    "text,expected_minutes",
    [
        ("1m", 1),
        ("5h", 300),
        ("2d", 2880),
        ("60m", 60),
        ("1h", 60),
        ("1440m", 1440),
        ("1d", 1440),
    ],
)
def test_interval_parse_and_minutes_param(text, expected_minutes):
    assert Interval.parse(text).minutes == expected_minutes


@pytest.mark.parametrize("bad_text", ["", "0m", "10x", "m", "999z"])
def test_interval_parse_invalid_param(bad_text):
    with pytest.raises(ValueError):
        Interval.parse(bad_text)


def test_compute_date_slices_single_window_default_rule():
    s = compute_date_slices(
        start="2024-01-01",
        end="2024-04-30",
        interval="1m",
        days_per_interval_minute=120,  # 120 days per 1-minute interval
    )
    assert len(s.datetime_ranges) == 1
    a, b = s.datetime_ranges[0]
    assert a == dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    assert b == dt.datetime(2024, 4, 30, tzinfo=dt.timezone.utc)
    au, bu = s.unix_ranges[0]
    assert isinstance(au, int) and isinstance(bu, int)
    assert bu > au


def test_compute_date_slices_multiple_windows_by_lowering_allowance():
    # ~120 days total; allowance 30 days per 1m interval -> ~4 windows
    s = compute_date_slices(
        start="2024-01-01",
        end="2024-04-30",
        interval="1m",
        days_per_interval_minute=30,
    )
    assert len(s.datetime_ranges) >= 3
    # Coverage
    assert s.datetime_ranges[-1][0] == dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    assert s.datetime_ranges[0][1] == dt.datetime(2024, 4, 30, tzinfo=dt.timezone.utc)
    # Contiguity (built backward)
    prev_start = None
    for a, b in s.datetime_ranges:
        assert a <= b
        if prev_start is not None:
            assert b == prev_start
        prev_start = a


@pytest.mark.parametrize(
    "ivl_a, ivl_b",
    [
        ("60m", "1h"),
        ("120m", "2h"),
        ("1440m", "1d"),
    ],
)
def test_equivalent_intervals_produce_same_windows(ivl_a, ivl_b):
    start = "2024-01-01"
    end = "2024-02-15"  # ~45 days
    allowance = 10  # days per 1-minute interval
    s_a = compute_date_slices(start=start, end=end, interval=ivl_a, days_per_interval_minute=allowance)
    s_b = compute_date_slices(start=start, end=end, interval=ivl_b, days_per_interval_minute=allowance)
    assert s_a.datetime_ranges == s_b.datetime_ranges
    assert s_a.unix_ranges == s_b.unix_ranges


def test_leap_day_is_included_and_contiguous():
    # allowance = 1 day per 1m interval -> window = 1 day
    s = compute_date_slices(
        start="2024-02-28",
        end="2024-03-01",
        interval="1m",
        days_per_interval_minute=1,
    )
    # Coverage
    assert s.datetime_ranges[-1][0] == dt.datetime(2024, 2, 28, tzinfo=dt.timezone.utc)
    assert s.datetime_ranges[0][1] == dt.datetime(2024, 3, 1, tzinfo=dt.timezone.utc)
    # Leap-day included
    covered_days = set()
    for a, b in s.datetime_ranges:
        day = a.date()
        while day <= b.date():
            covered_days.add(day)
            day += dt.timedelta(days=1)
    assert dt.date(2024, 2, 29) in covered_days


def test_large_span_window_count_matches_formula():
    start = "2020-01-01"
    end = "2025-01-01"
    interval = "5m"
    allowance = 30  # window size = 5 * 30 = 150 days
    s = compute_date_slices(start=start, end=end, interval=interval, days_per_interval_minute=allowance)
    start_dt = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)
    end_dt = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
    total_days = (end_dt - start_dt).days
    expected_windows = math.ceil(total_days / 150)
    assert len(s.datetime_ranges) == expected_windows


def test_compute_date_slices_invalid_range_raises():
    with pytest.raises(ValueError):
        compute_date_slices(start="2024-02-01", end="2024-01-31", interval="1m")
