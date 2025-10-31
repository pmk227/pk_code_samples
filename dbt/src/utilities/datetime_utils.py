import datetime as _dt
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Union

import pandas as pd


# ------------------------
# Basic conversions
# ------------------------

def convert_datetime_to_dateid(date: Union[_dt.date, _dt.datetime, pd.Series]) -> Union[int, pd.Series]:
    if isinstance(date, pd.Series):
        return date.dt.year * 10000 + date.dt.month * 100 + date.dt.day
    else:
        return date.year * 10000 + date.month * 100 + date.day


def convert_strdate_to_dateid(date: Union[str, pd.Series], pattern: str = "%Y-%m-%d") -> Union[int, pd.Series]:
    """
    Convert a string (or series of strings) to date_id using a format pattern.
    """
    if isinstance(date, str):
        # preserve scalar return type
        ts = pd.to_datetime(pd.Series([date]), format=pattern)
        return int(convert_datetime_to_dateid(ts.iloc[0]))
    else:
        ts = pd.to_datetime(date, format=pattern)
        return convert_datetime_to_dateid(ts)


def convert_date_id_to_date(date: int) -> _dt.date:
    year = date // 10000
    month = (date // 100) % 100
    day = date % 100
    return _dt.date(year, month, day)


def convert_rel_date_to_date_id(relative_date: int) -> int:
    """Converts a relative date_id (0, -5, -25) to an absolute date id (20240815)."""
    if relative_date > 200000:
        return relative_date  # already a date_id
    today = _dt.date.today() + _dt.timedelta(days=relative_date)
    return convert_datetime_to_dateid(today)


def convert_date_id_to_target_type(date_id: int, date_format: str) -> Union[str, int]:
    """Converts date_id to target type. Use 'date_id' to keep as int or 'unix'/'unix_ms'/'unix_us'/'unix_ns'."""
    if not date_format:
        raise ValueError("date_format is required.")

    if date_format.lower() == "date_id":
        return date_id

    if date_format[:4].lower() == "unix":
        date_variable = convert_date_id_to_date(date_id)
        dt = _dt.datetime.combine(date_variable, _dt.datetime.min.time(), tzinfo=_dt.timezone.utc)
        seconds = int(dt.timestamp())
        match date_format:
            case "unix":
                return seconds
            case "unix_ms":
                return seconds * 1_000
            case "unix_us":
                return seconds * 1_000_000
            case "unix_ns":
                return seconds * 1_000_000_000
            case _:
                raise ValueError(f"Unsupported unix format: {date_format}")

    date_variable = convert_date_id_to_date(date_id)
    return _dt.datetime.combine(date_variable, _dt.datetime.min.time()).strftime(date_format)


def convert_any_to_datetime(value, in_tz="UTC", out_tz="UTC"):
    """
    Convert str/int/datetime into tz-aware pandas Timestamp.

    - int YYYYMMDD (date_id)
    - int relative (0, -10, etc.)
    - string ("YYYY-MM-DD" or with time)
    - datetime-like
    """

    def check_n_convert_cst(tz):
        if isinstance(tz, str) and tz.upper() == "CST":
            return "America/Chicago"
        else:
            return tz

    in_tz = check_n_convert_cst(in_tz)
    out_tz = check_n_convert_cst(out_tz)

    if isinstance(value, int):
        if value > 20000000:  # treat as YYYYMMDD
            ts = pd.to_datetime(str(value), format="%Y%m%d")
        else:  # relative offset
            ts = pd.Timestamp.today().normalize() + pd.Timedelta(days=value)
    else:
        ts = pd.to_datetime(value, errors="raise")

    # enforce tz aware input
    ts = ts.tz_localize(in_tz) if ts.tzinfo is None else ts.tz_convert(in_tz)

    return ts.tz_convert(out_tz) # convert for output



# ------------------------------------
# Interval parsing & date slice logic
# ------------------------------------

DATE_FMT = "%Y-%m-%d"


class IntervalUnit(str, Enum):
    MINUTE = "m"
    HOUR = "h"
    DAY = "d"


@dataclass(frozen=True)
class Interval:
    """Parsed interval like '1m', '2h', '1d'."""
    value: int
    unit: IntervalUnit

    @staticmethod
    def parse(text: str) -> "Interval":
        if not text or not isinstance(text, str):
            raise ValueError("Interval string is required, e.g., '1m', '2h', '1d'.")
        text = text.strip()
        unit_char = text[-1].lower()
        if unit_char not in {u.value for u in IntervalUnit}:
            raise ValueError(f"Unsupported interval unit in '{text}'. Use one of: m/h/d.")
        try:
            value = int(text[:-1])
        except ValueError:
            raise ValueError(f"Interval must start with an integer: got '{text}'.")
        if value <= 0:
            raise ValueError("Interval value must be > 0.")
        return Interval(value=value, unit=IntervalUnit(unit_char))

    @property
    def minutes(self) -> int:
        match self.unit:
            case IntervalUnit.MINUTE:
                return self.value
            case IntervalUnit.HOUR:
                return self.value * 60
            case IntervalUnit.DAY:
                return self.value * 60 * 24


def to_datetime_utc(d: Union[str, _dt.datetime], pattern: str = DATE_FMT) -> _dt.datetime:
    if isinstance(d, str):
        dt = _dt.datetime.strptime(d, pattern)
        return dt.replace(tzinfo=_dt.timezone.utc)
    if isinstance(d, _dt.datetime):
        return d.astimezone(_dt.timezone.utc) if d.tzinfo else d.replace(tzinfo=_dt.timezone.utc)
    raise TypeError("start/end must be 'YYYY-MM-DD' string or datetime.")


@dataclass(frozen=True)
class DateSlices:
    datetime_ranges: List[Tuple[_dt.datetime, _dt.datetime]]
    unix_ranges: List[Tuple[int, int]]  # seconds since epoch


def compute_date_slices(
    *,
    start: Union[str, _dt.datetime],
    end: Union[str, _dt.datetime],
    interval: Union[str, Interval],
    days_per_interval_minute: int = 120,
    date_format: str = DATE_FMT,  # used when start/end are strings
) -> DateSlices:
    """
    Split [start, end] into closed intervals that respect a vendor max-span rule:
      max span (days) = interval.minutes * days_per_interval_minute

    Returns ranges in both datetime(UTC) and unix seconds.
    """
    start_dt = to_datetime_utc(start, pattern=date_format) if isinstance(start, str) else to_datetime_utc(start)
    end_dt = to_datetime_utc(end, pattern=date_format) if isinstance(end, str) else to_datetime_utc(end)
    if start_dt > end_dt:
        raise ValueError("start must be <= end.")

    ivl = Interval.parse(interval) if isinstance(interval, str) else interval
    max_days = ivl.minutes * days_per_interval_minute
    if max_days <= 0:
        raise ValueError("Computed max days per request must be > 0.")
    max_window = _dt.timedelta(days=max_days)

    ranges_dt: List[Tuple[_dt.datetime, _dt.datetime]] = []
    cur_end = end_dt
    while True:
        cur_start = max(start_dt, cur_end - max_window)
        ranges_dt.append((cur_start, cur_end))
        if cur_start == start_dt:
            break
        cur_end = cur_start

    ranges_unix: List[Tuple[int, int]] = [(int(a.timestamp()), int(b.timestamp())) for a, b in ranges_dt]
    return DateSlices(datetime_ranges=ranges_dt, unix_ranges=ranges_unix)
