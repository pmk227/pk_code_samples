# tests/test_throttlers.py
import pytest
from typing import Any, Iterable
from threading import Event
from multiprocessing import Manager

from src.io.throttler import (
    ThrottleUnit,
    SerialThrottler,
    MultiThreadThrottler,
    MultiProcessThrottler,
)
import src.io.throttler as th_mod  # for monkeypatching module-level names like time.sleep / os.cpu_count


# ============================================================
# SerialThrottler
# ============================================================

def test_serial_accumulate_outputs_and_payload_order():
    calls: list[dict[str, Any]] = []

    def hook(*, endpoint: str, throttle: dict, **_):
        calls.append({"endpoint": endpoint, "throttle": throttle})
        return f"{endpoint}:{throttle['kind']}"

    units = [
        ThrottleUnit(kind="date_range", meta={"from": "2024-01-01", "to": "2024-01-10"}),
        ThrottleUnit(kind="date_range", meta={"from": "2024-01-11", "to": "2024-01-20"}),
    ]

    t = SerialThrottler(hook=hook, base_kwargs={"endpoint": "intraday"})
    results = t.throttle(units=units, accumulate_results=True)

    assert results == ["intraday:date_range", "intraday:date_range"]
    assert calls[0]["throttle"] == {"kind": "date_range", "from": "2024-01-01", "to": "2024-01-10"}
    assert calls[1]["throttle"] == {"kind": "date_range", "from": "2024-01-11", "to": "2024-01-20"}


def test_serial_accepts_plain_dict_units_equivalently():
    calls: list[dict[str, Any]] = []

    def hook(*, throttle: dict, **_):
        calls.append(throttle)
        return throttle["kind"]

    dict_units = [
        {"kind": "list", "items": ["AAPL", "MSFT"]},
        {"kind": "list", "items": ["GOOG"]},
    ]

    t = SerialThrottler(hook=hook)
    results = t.throttle(units=dict_units, accumulate_results=True)

    assert results == ["list", "list"]
    assert calls == [
        {"kind": "list", "items": ["AAPL", "MSFT"]},
        {"kind": "list", "items": ["GOOG"]},
    ]


def test_serial_sleep_invoked_per_unit(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(th_mod.time, "sleep", lambda s: calls.__setitem__("n", calls["n"] + 1))

    def hook(*, throttle: dict, **_):
        return None

    units = [
        ThrottleUnit(kind="date", meta={"date": "2024-01-05"}),
        ThrottleUnit(kind="date", meta={"date": "2024-01-06"}),
    ]

    SerialThrottler(hook=hook, sleep_s=0.1).throttle(units=units)
    assert calls["n"] == 2


def test_serial_empty_units_returns_none_and_calls_nothing():
    def hook(**_):
        raise AssertionError("hook should not be called")

    out = SerialThrottler(hook=hook, base_kwargs={"x": 1}).throttle(units=[])
    assert out is None


def test_serial_empty_units_accumulate_true_returns_empty_list():
    def hook(**_):
        raise AssertionError("hook should not be called")

    out = SerialThrottler(hook=hook).throttle(units=[], accumulate_results=True)
    assert out == []


def test_serial_hook_exception_propagates_and_stops_iteration():
    seen = []

    def hook(*, throttle: dict, **_):
        seen.append(throttle)
        if throttle["kind"] == "boom":
            raise RuntimeError("boom")
        return "ok"

    units = [
        ThrottleUnit(kind="date", meta={"date": "2024-01-01"}),
        ThrottleUnit(kind="boom", meta={}),
        ThrottleUnit(kind="date", meta={"date": "2024-01-02"}),  # should not run
    ]

    with pytest.raises(RuntimeError, match="boom"):
        SerialThrottler(hook=hook).throttle(units=units, accumulate_results=True)

    assert [u["kind"] for u in seen] == ["date", "boom"]


def test_serial_dict_unit_missing_kind_raises_key_error():
    bad_units = [{"from": "2024-01-01", "to": "2024-01-02"}]

    def hook(**_):
        return None

    with pytest.raises((KeyError, ValueError)):
        SerialThrottler(hook=hook).throttle(units=bad_units)


def test_serial_preserves_units_without_expansion():
    calls = []

    def hook(*, throttle: dict, **_):
        calls.append(throttle)
        return throttle

    units = [
        ThrottleUnit(kind="date", meta={"from_date": "2024-01-05", "to_date": "2024-01-08"}),
        ThrottleUnit(kind="date", meta={"from_date": "2024-01-09", "to_date": "2024-01-12"}),
    ]

    out = SerialThrottler(hook=hook).throttle(units=units, accumulate_results=True)
    assert calls == [
        {"kind": "date", "from_date": "2024-01-05", "to_date": "2024-01-08"},
        {"kind": "date", "from_date": "2024-01-09", "to_date": "2024-01-12"},
    ]
    assert out == calls


def test_serial_negative_sleep_raises_value_error():
    def hook(**_):  # pragma: no cover
        pass

    with pytest.raises(ValueError):
        SerialThrottler(hook=hook, sleep_s=-0.01)


# ============================================================
# MultiThreadThrottler (multithreaded)
# ============================================================

def test_multithread_executes_all_units_no_aggregation():
    seen = []

    def hook(*, throttle: dict, **_):
        seen.append(throttle["kind"])  # side-effect only

    units = [ThrottleUnit(kind="k", meta={"i": i}) for i in range(20)]

    MultiThreadThrottler(hook=hook, max_workers=4).throttle(units=units)

    assert len(seen) == 20
    assert set(seen) == {"k"}  # order not guaranteed


def test_multithread_exception_propagates_and_cancels_remaining():
    seen = []

    def hook(*, throttle: dict, **_):
        seen.append(throttle["kind"])
        if throttle["kind"] == "boom":
            raise RuntimeError("boom")

    units = [
        ThrottleUnit(kind="ok", meta={}),
        ThrottleUnit(kind="boom", meta={}),
        ThrottleUnit(kind="ok", meta={}),
    ]

    with pytest.raises(RuntimeError, match="boom"):
        MultiThreadThrottler(hook=hook, max_workers=2).throttle(units=units)

    assert "boom" in seen  # at least the failing unit ran


def test_multithread_negative_sleep_raises_value_error():
    def hook(**_):  # pragma: no cover
        pass

    with pytest.raises(ValueError):
        MultiThreadThrottler(hook=hook, sleep_s=-0.1)


def test_multithread_invalid_max_workers_raises_value_error():
    def hook(**_):  # pragma: no cover
        pass

    with pytest.raises(ValueError):
        MultiThreadThrottler(hook=hook, max_workers=0)


def test_multithread_default_workers_from_cpu_count(monkeypatch):
    monkeypatch.setattr(th_mod.os, "cpu_count", lambda: 8)
    t = MultiThreadThrottler(hook=lambda **_: None)
    assert t.max_workers == 7  # 8 - 1


def test_multithread_limit_event_stops_new_submissions():
    calls = []
    limit = Event()

    def hook(*, throttle: dict, limit_event: Event | None = None, **_):
        calls.append(throttle)
        if limit_event is not None:
            limit_event.set()  # first task flips it

    units = [ThrottleUnit(kind="k", meta={"i": i}) for i in range(2)]
    MultiThreadThrottler(hook=hook, max_workers=1, limit_event=limit).throttle(units=units)

    assert len(calls) == 1  # only the first unit should have run


def test_multithread_limit_event_pre_set_submits_nothing():
    calls = []
    limit = Event()
    limit.set()  # already out of tokens/limit hit

    def hook(*, throttle: dict, **_):
        calls.append(throttle)

    units = [ThrottleUnit(kind="k", meta={"i": i}) for i in range(3)]
    MultiThreadThrottler(hook=hook, max_workers=2, limit_event=limit).throttle(units=units)

    assert calls == []


def test_multithread_limit_event_true_injected_into_hook():
    flags = []

    def hook(*, throttle: dict, limit_event=None, **_):
        flags.append(limit_event is not None)

    units = [ThrottleUnit(kind="k", meta={"i": 0})]
    MultiThreadThrottler(hook=hook, max_workers=1, limit_event=True).throttle(units=units)

    assert flags == [True]


# ============================================================
# MultiProcessThrottler (multiprocessing)
#   NOTE: hooks must be top-level (picklable).
# ============================================================

# ---- picklable hooks ----
def _mp_hook_append(*, throttle: dict, sink=None, **_) -> None:
    if sink is not None:
        sink.append(throttle)

def _mp_hook_kind(*, throttle: dict, sink=None, **_) -> None:
    if sink is not None:
        sink.append(throttle["kind"])

def _mp_hook_raise_on_boom(*, throttle: dict, sink=None, **_) -> None:
    if sink is not None:
        sink.append(throttle)
    if throttle["kind"] == "boom":
        raise RuntimeError("boom")

def _mp_hook_stop_on_flag(*, throttle: dict, limit_event=None, sink=None, **_) -> None:
    if sink is not None:
        sink.append(throttle)
    if throttle.get("stop") and limit_event is not None:
        limit_event.set()

def _mp_hook_capture_kwargs(*, throttle: dict, seen=None, **kwargs) -> None:
    if seen is not None:
        seen.append(kwargs.get("token"))

def _mp_hook_records_event_presence(*, throttle: dict, limit_event=None, sink=None, **_) -> None:
    if sink is not None:
        sink.append(limit_event is not None)



# ---- tests ----
def test_multiprocess_executes_all_units_no_event():
    with Manager() as m:
        sink = m.list()
        units = [ThrottleUnit(kind="k", meta={"i": i}) for i in range(12)]

        mp = MultiProcessThrottler(hook=_mp_hook_append, base_kwargs={"sink": sink}, max_workers=3)
        mp.throttle(units=units)

        assert len(sink) == 12
        assert sorted(u["i"] for u in sink) == list(range(12))


def test_multiprocess_accepts_plain_dict_units_equivalently():
    with Manager() as m:
        sink = m.list()
        dict_units = [
            {"kind": "list", "items": ["AAPL", "MSFT"]},
            {"kind": "list", "items": ["GOOG"]},
        ]
        mp = MultiProcessThrottler(hook=_mp_hook_append, base_kwargs={"sink": sink}, max_workers=2)
        mp.throttle(units=dict_units)

        actual = list(sink)
        # order-independent comparison by a stable key (kind + items tuple)
        key = lambda d: (d["kind"], tuple(d.get("items", [])))
        assert sorted(actual, key=key) == sorted(dict_units, key=key)



def test_multiprocess_generator_units_streamed_no_prematerialization():
    def gen() -> Iterable[ThrottleUnit]:
        for i in range(5):
            yield ThrottleUnit(kind="k", meta={"i": i})

    with Manager() as m:
        sink = m.list()
        mp = MultiProcessThrottler(hook=_mp_hook_append, base_kwargs={"sink": sink}, max_workers=2)
        mp.throttle(units=gen())

        assert len(sink) == 5
        assert sorted(u["i"] for u in sink) == list(range(5))


def test_multiprocess_sleep_invoked_per_submission(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(th_mod.time, "sleep", lambda s: calls.__setitem__("n", calls["n"] + 1))

    with Manager() as m:
        sink = m.list()
        units = [ThrottleUnit(kind="k", meta={"i": i}) for i in range(3)]

        mp = MultiProcessThrottler(hook=_mp_hook_append, base_kwargs={"sink": sink}, sleep_s=0.1, max_workers=2)
        mp.throttle(units=units)

        assert calls["n"] == 3
        assert len(sink) == 3


def test_multiprocess_exception_propagates_and_stops_further_when_capacity_one():
    with Manager() as m:
        sink = m.list()
        units = [
            ThrottleUnit(kind="ok", meta={"i": 0}),
            ThrottleUnit(kind="boom", meta={"i": 1}),
            ThrottleUnit(kind="ok", meta={"i": 2}),  # should not run with max_workers=1
        ]

        mp = MultiProcessThrottler(hook=_mp_hook_raise_on_boom, base_kwargs={"sink": sink}, max_workers=1)
        with pytest.raises(RuntimeError, match="boom"):
            mp.throttle(units=units)

        kinds = [u["kind"] for u in sink]
        assert kinds == ["ok", "boom"]  # deterministic for pool size 1


def test_multiprocess_missing_kind_in_dict_unit_raises_key_error():
    bad_units = [{"from": "2024-01-01", "to": "2024-01-02"}]  # no "kind"

    with Manager() as m:
        sink = m.list()
        mp = MultiProcessThrottler(hook=_mp_hook_append, base_kwargs={"sink": sink}, max_workers=1)

        with pytest.raises((KeyError, ValueError)):
            mp.throttle(units=bad_units)


def test_multiprocess_limit_event_true_internal_creation_and_stop():
    """True → create Manager().Event internally; first unit flips it; with capacity 1 only first runs."""
    with Manager() as m:
        sink = m.list()
        units = [
            ThrottleUnit(kind="k", meta={"i": 0, "stop": True}),
            ThrottleUnit(kind="k", meta={"i": 1}),
        ]
        mp = MultiProcessThrottler(
            hook=_mp_hook_stop_on_flag,
            base_kwargs={"sink": sink},
            max_workers=1,
            limit_event=True,
        )
        mp.throttle(units=units)

        assert len(sink) == 1
        assert dict(sink[0])["i"] == 0


def test_multiprocess_limit_event_existing_proxy_pre_set_prevents_submission():
    with Manager() as m:
        sink = m.list()
        evt = m.Event()
        evt.set()

        units = [ThrottleUnit(kind="k", meta={"i": i}) for i in range(3)]
        mp = MultiProcessThrottler(hook=_mp_hook_append, base_kwargs={"sink": sink}, max_workers=2, limit_event=evt)
        mp.throttle(units=units)

        assert list(sink) == []


def test_multiprocess_limit_event_existing_proxy_set_by_first_task_stops_new_submissions():
    with Manager() as m:
        sink = m.list()
        evt = m.Event()

        units = [
            ThrottleUnit(kind="k", meta={"i": 0, "stop": True}),
            ThrottleUnit(kind="k", meta={"i": 1}),
        ]
        mp = MultiProcessThrottler(
            hook=_mp_hook_stop_on_flag,
            base_kwargs={"sink": sink},
            max_workers=1,
            limit_event=evt,
        )
        mp.throttle(units=units)

        assert len(sink) == 1
        assert dict(sink[0])["i"] == 0
        assert evt.is_set()


def test_multiprocess_limit_event_true_injected_into_hook():
    with Manager() as m:
        flags = m.list()
        units = [ThrottleUnit(kind="k", meta={"i": 0})]

        mp = MultiProcessThrottler(
            hook=_mp_hook_records_event_presence,
            base_kwargs={"sink": flags},
            limit_event=True,
            max_workers=1,
        )
        mp.throttle(units=units)

        assert list(flags) == [True]


def test_multiprocess_ctor_guards_and_defaults(monkeypatch):
    # negative sleep
    with pytest.raises(ValueError):
        MultiProcessThrottler(hook=_mp_hook_append, sleep_s=-0.01)

    # invalid workers
    with pytest.raises(ValueError):
        MultiProcessThrottler(hook=_mp_hook_append, max_workers=0)

    # default workers = max(1, os.cpu_count()-1)
    monkeypatch.setattr(th_mod.os, "cpu_count", lambda: 6)
    mp = MultiProcessThrottler(hook=_mp_hook_append)
    assert mp.max_workers == 5


def test_multiprocess_base_kwargs_are_copied_not_aliases():
    with Manager() as m:
        seen = m.list()
        mutable = {"token": "A"}
        mp = MultiProcessThrottler(hook=_mp_hook_capture_kwargs, base_kwargs={"seen": seen, **mutable}, max_workers=1)

        # mutate original after constructing the throttler
        mutable["token"] = "B"

        units = [ThrottleUnit(kind="k", meta={})]
        mp.throttle(units=units)

        assert list(seen) == ["A"]
