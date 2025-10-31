import math
import pytest

from src.io.throttle_unit_builders import (
    enumerate_day_units,
    date_range_units,
    list_chunk_units,
)
from src.utilities.datetime_utils import compute_date_slices


def test_enumerate_day_units_inclusive_and_order():
    units = enumerate_day_units(start="2024-01-05", end="2024-01-08")
    # 05, 06, 07, 08 -> 4 units
    assert len(units) == 4
    assert [u.kind for u in units] == ["date"] * 4
    assert [u.meta["date"] for u in units] == [
        "2024-01-05",
        "2024-01-06",
        "2024-01-07",
        "2024-01-08",
    ]


def test_enumerate_day_units_raises_on_start_after_end():
    with pytest.raises(ValueError):
        enumerate_day_units(start="2024-01-08", end="2024-01-05")


def test_date_range_units_aligns_with_compute_date_slices():
    # Force multiple windows by lowering allowance
    start, end = "2024-01-01", "2024-04-30"
    interval = "1m"
    allowance = 30  # days per 1-minute interval

    # Expected windows from the canonical slicer
    slices = compute_date_slices(
        start=start,
        end=end,
        interval=interval,
        days_per_interval_minute=allowance,
        date_format="%Y-%m-%d",
    )
    expected = [
        (a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d"))
        for a, b in slices.datetime_ranges
    ]

    # Builder output
    units = date_range_units(
        start=start,
        end=end,
        interval=interval,
        days_per_interval_minute=allowance,
        date_format="%Y-%m-%d",
    )

    assert len(units) == len(expected)
    assert all(u.kind == "date_range" for u in units)
    got = [(u.meta["from_date"], u.meta["to_date"]) for u in units]
    assert got == expected


def test_list_chunk_units_exact_division_and_remainder():
    # 500 items, chunk_size 128 => 3 full (128*3=384) + 1 remainder (116) = 4 chunks
    items = list(range(500))
    units = list_chunk_units(items, chunk_size=128)

    assert len(units) == math.ceil(500 / 128) == 4
    assert all(u.kind == "list" for u in units)

    lengths = [len(u.meta["items"]) for u in units]
    assert lengths[:3] == [128, 128, 128]
    assert lengths[3] == 500 - 128 * 3 == 116

    # Items preserved in order within chunks
    rebuilt = [x for u in units for x in u.meta["items"]]
    assert rebuilt == items


def test_list_chunk_units_rejects_nonpositive_chunk_size():
    with pytest.raises(ValueError):
        list_chunk_units(list(range(10)), chunk_size=0)
    with pytest.raises(ValueError):
        list_chunk_units(list(range(10)), chunk_size=-5)


def test_enumerate_day_units_single_day_inclusive():
    units = enumerate_day_units(start="2024-03-10", end="2024-03-10")
    assert len(units) == 1
    assert units[0].kind == "date"
    assert units[0].meta["date"] == "2024-03-10"


def test_enumerate_day_units_custom_format():
    units = enumerate_day_units(start="03/01/2024", end="03/03/2024", date_format="%m/%d/%Y")
    assert [u.meta["date"] for u in units] == ["03/01/2024", "03/02/2024", "03/03/2024"]


def test_enumerate_day_units_leap_day_present():
    units = enumerate_day_units(start="2024-02-28", end="2024-03-01")
    dates = [u.meta["date"] for u in units]
    assert "2024-02-29" in dates
    assert dates == ["2024-02-28", "2024-02-29", "2024-03-01"]


def test_date_range_units_single_window_when_allowance_large():
    # Allowance big enough to keep it one slice
    units = date_range_units(
        start="2024-01-01",
        end="2024-04-30",
        interval="1m",
        days_per_interval_minute=10_000,  # huge -> single slice
    )
    assert len(units) == 1
    assert units[0].kind == "date_range"
    assert units[0].meta["from_date"] == "2024-01-01"
    assert units[0].meta["to_date"] == "2024-04-30"


def test_date_range_units_invalid_interval_raises():
    with pytest.raises(ValueError):
        date_range_units(
            start="2024-01-01",
            end="2024-01-10",
            interval="10x",  # bad unit
        )


def test_date_range_units_start_after_end_raises():
    with pytest.raises(ValueError):
        date_range_units(
            start="2024-05-01",
            end="2024-04-01",
            interval="1m",
        )


def test_list_chunk_units_chunk_size_ge_len_returns_single_unit():
    items = list(range(10))
    units = list_chunk_units(items, chunk_size=50)
    assert len(units) == 1
    assert units[0].kind == "list"
    assert units[0].meta["items"] == items  # preserves order and all items


def test_list_chunk_units_chunk_size_one_returns_one_item_per_unit():
    items = ["a", "b", "c"]
    units = list_chunk_units(items, chunk_size=1)
    assert len(units) == 3
    assert [u.meta["items"] for u in units] == [["a"], ["b"], ["c"]]


def test_list_chunk_units_empty_input_returns_empty():
    units = list_chunk_units([], chunk_size=10)
    assert units == []
