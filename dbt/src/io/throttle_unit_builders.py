# src/io/throttle_unit_builders.py
from typing import Any, List, Sequence
import pandas as pd
import numpy as np

from src.io.throttler import ThrottleUnit
from src.utilities.datetime_utils import compute_date_slices


def date_range_units(*, start: str | pd.Timestamp, end: str | pd.Timestamp, interval: str, days_per_interval_minute: int = 120,
                     date_format: str = "%Y-%m-%d",) -> List[ThrottleUnit]:
    """
    Build range-based units that align to vendor constraints derived from `interval`.
    Example payload handed to your hook:
        {"kind": "date_range", "from_date": "YYYY-MM-DD", "to_date": "YYYY-MM-DD"}


    :param start:       Start date
    :param end:         End date
    :param interval:    1m, 5m, 1h, etc (accepts m, h, d)
    :param days_per_interval_minute: (optional) - eodhd uses 120
    :param date_format: (optional) - eodhd uses YYYY-MM-DD
    :return:
    """
    slices = compute_date_slices(
        start=start,
        end=end,
        interval=interval,
        days_per_interval_minute=days_per_interval_minute,
        date_format=date_format,
    )
    units: List[ThrottleUnit] = []
    for a, b in slices.datetime_ranges:
        units.append(
            ThrottleUnit(
                kind="date_range",
                meta={"from_date": a.strftime(date_format), "to_date": b.strftime(date_format)},
            )
        )
    return units


def enumerate_day_units(*, start: str, end: str, date_format: str = "%Y-%m-%d",) -> List[ThrottleUnit]:
    """
    Build one unit per calendar day (no vendor-alignment; pure daily fan-out).
    Example payload to hook: {"kind": "date", "date": "YYYY-MM-DD"}
    """
    units: List[ThrottleUnit] = []
    from datetime import datetime, timedelta

    a = datetime.strptime(start, date_format)
    b = datetime.strptime(end, date_format)
    if a > b:
        raise ValueError("start must be <= end")

    cur = a
    one_day = timedelta(days=1)
    while cur <= b:
        units.append(ThrottleUnit(kind="date", meta={"date": cur.strftime(date_format)}))
        cur += one_day
    return units


def list_chunk_units(
    items: Sequence[Any], *, chunk_size: int,) -> List[ThrottleUnit]:
    """
    Generic list chunker. Produces units like:
        {"kind": "list", "items": [ ... up to chunk_size ... ]}
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    units: List[ThrottleUnit] = []
    for i in range(0, len(items), chunk_size):
        units.append(ThrottleUnit(kind="list", meta={"items": list(items[i : i + chunk_size])}))
    return units


def dataframe_chunk_units(df: pd.DataFrame, *, chunk_size: int=1, num_chunks: int=None, return_rows:bool=False) -> List[ThrottleUnit]:
    """
    Create ThrottleUnits that describe iloc ranges into `df`.
    Each unit: {"kind": "df_iloc", "start": i, "stop": j}

    :param df:          dataframe whose indexes to chunk
    :param chunk_size:  (optional) chunk size for each unit
    :param num_chunks:  (optional) number of chunks to return
    :param return_rows: (optional) return entire rows instead of row indices
    :return:
    """

    if chunk_size is None and num_chunks is None:
        raise ValueError("Either chunk_size or num_chunks must be specified")

    if isinstance(num_chunks, int):
        chunks = _balanced_group_sizes(len(df), num_chunks)
    elif isinstance(num_chunks, float):
        raise ValueError("num_chunks must be an int")
    else:
        chunks = [chunk_size]*int(np.ceil((len(df)/chunk_size)))

    units: List[ThrottleUnit] = []

    if not return_rows:
        total_rows = len(df)
        for start in range(0, total_rows, chunk_size):
            stop = min(start + chunk_size, total_rows)
            units.append(ThrottleUnit(kind="df_iloc", meta={"start": start, "stop": stop}))
    else:

        indices = np.cumsum(chunks)[:-1]  # drop last, since that's the end of df
        dfs = np.split(df, indices)

        for chunk in dfs:
            units.append(ThrottleUnit(kind="DataFrame", meta={"df_chunk": chunk}))

    return units


def _balanced_group_sizes(items: int, groups: int) -> List[int]:
    """
    Split n items into k groups where each group size differs by at most 1.

    Args:
        items: Total number of items
        groups: Number of groups

    Returns:
        List of group sizes of length k.
    """
    if items <= groups:
        # if fewer items than groups, groups are size 1
        return [1] * items

    quotient, remainder = divmod(items, groups)  # quotient = base size, remainder = how many get +1
    return [quotient + 1] * remainder + [quotient] * (groups - remainder)

def dataframe_single_unit(df: pd.DataFrame, *, copy_df: bool = True) -> List[ThrottleUnit]:
    """
    Wrap an entire DataFrame in a single ThrottleUnit so call sites can treat it
    like any other unit stream (even when no throttling/fan-out is needed).

    Unit shape mirrors dataframe_chunk_units(return_rows=True):
        {"kind": "DataFrame", "df_chunk": <pd.DataFrame>}

    Args:
        df:       The DataFrame to pass through.
        copy_df:  If True, deep-copy the DataFrame before embedding (avoids mutation
                  surprises downstream; costs memory/time).

    Returns:
        A single-element list containing one ThrottleUnit.

    Raises:
        TypeError: If df is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    df_payload = df.copy(deep=True) if copy_df else df
    unit = [ThrottleUnit(kind="DataFrame", meta={"df_chunk": df_payload})]
    return unit
