# src/io/throttler.py
import os
import time
from concurrent.futures import as_completed, Future, ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from threading import Event
from typing import Any, Callable, Iterable, Mapping, Optional, Union, Tuple, Literal

from threading import Event as ThreadEvent
from multiprocessing import Manager


@dataclass(frozen=True, slots=True)
class ThrottleUnit:
    kind: str  # "date" | "date_range" | "list"
    meta: Mapping[str, Any]  # e.g. {"from": "...", "to": "..."} or {"items": [...]}

    def as_payload(self) -> dict[str, Any]:
        return {"kind": self.kind, **self.meta}


UnitLike = Union[ThrottleUnit, Mapping[str, Any]]  # allow dicts too if you want


class SerialThrottler:
    """Serial execution helper: calls `hook(**base_kwargs, throttle=unit.as_payload())` per unit."""

    def __init__(
            self,
            hook: Callable[..., Any],
            base_kwargs: Optional[dict] = None,
            sleep_s: float = 0.0,
    ) -> None:
        """Initialize the throttler.

        Args:
            hook: Function executed for each unit; exceptions propagate.
            base_kwargs: Keyword args passed to every hook call.
            sleep_s: Seconds to sleep after each call (must be >= 0).

        Raises:
            ValueError: If `sleep_s` is negative.
        """
        if sleep_s < 0:
            raise ValueError("sleep_s must be >= 0")

        self.hook = hook
        self.base_kwargs = base_kwargs or {}
        self.sleep_s = float(sleep_s)

    def throttle(self, units: Iterable[UnitLike], accumulate_results: bool = False) -> None | list:
        """Run the bound hook once per unit, serially.

        Args:
            units: Iterable of `ThrottleUnit` or dicts with at least a `'kind'` key.
            accumulate_results: If `True`, results are appended after each run.

        Returns:
            None

        Raises:
            KeyError: If a dict-like unit is missing `'kind'`.
            Exception: Any exception raised by `hook` is propagated.
        """
        results = []
        for u in units:
            unit = u if isinstance(u, ThrottleUnit) else ThrottleUnit(kind=u["kind"], meta=u)  # gentle dict support
            if accumulate_results:
                results.append(self.hook(**self.base_kwargs, throttle=unit.as_payload()))
            else:
                self.hook(**self.base_kwargs, throttle=unit.as_payload())

            if self.sleep_s > 0:
                time.sleep(self.sleep_s)

        if accumulate_results:
            return results

        return None


class MultiThreadThrottler:
    """Run `hook(**base_kwargs, throttle=unit.as_payload())` over units concurrently; no aggregation."""

    def __init__(
            self,
            hook: Callable[..., Any],
            base_kwargs: Optional[dict] = None,
            sleep_s: float = 0.0,
            max_workers: Optional[int] = None,
            limit_event: Event | bool | None = None,
    ) -> None:
        """
        Run `hook(**base_kwargs, throttle=unit.as_payload())` over units concurrently; no aggregation.

        :param hook:            A serializable function (eg. static method, class method, or standalone function
        :param base_kwargs:     kwargs tobe passed to the hook
        :param sleep_s:         Time to sleep between executions
        :param max_workers:     Maximum number of concurrent units to run in parallel.
        :param limit_event:     Event() object that will link all workers
        """
        if sleep_s < 0:
            raise ValueError("sleep_s must be >= 0")
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers must be >= 1")

        n = min(64, max(4, (os.cpu_count() or 2) + 12))
        self.max_workers = max(1, (max_workers if max_workers is not None else n - 1))
        self.hook = hook
        self.base_kwargs = dict(base_kwargs or {})
        self.sleep_s = sleep_s
        self.limit_event: Event | None = (  # a threading event
            limit_event if isinstance(limit_event, Event)
            else (ThreadEvent() if limit_event else None)
        )
        self.stopped = False

    def throttle(self, units: Iterable[UnitLike], accumulate_results: bool = False) -> (
            Tuple[list[Any], BaseException | None] | None):
        """
        Executes the hook over 'units' using max_workers number of concurrent threads.

        :param units:               Iterable of `ThrottleUnit` or dicts with at least a `'kind'` key.
        :param accumulate_results:  If `True`, results are appended after each run.

        :return:                    None or List (f accumulate_results = True)
        """
        results = []
        self.stopped = False
        kwargs = self.base_kwargs.copy()

        if self.limit_event is not None and "limit_event" not in kwargs:
            kwargs["limit_event"] = self.limit_event

        it = iter(units)
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="throttle") as ex:
            inflight: set[Future] = set()

            def submit_next() -> bool:
                if self.limit_event and self.limit_event.is_set():
                    return False
                elif self.stopped:
                    return False
                try:
                    raw = next(it)
                except StopIteration:
                    return False
                unit = raw if isinstance(raw, ThrottleUnit) else ThrottleUnit(kind=raw["kind"], meta=raw)
                fut = ex.submit(self.hook, **kwargs, throttle=unit.as_payload())
                inflight.add(fut)
                if self.sleep_s > 0:
                    time.sleep(self.sleep_s)
                return True

            # Prime up to capacity
            for _ in range(self.max_workers):
                if not submit_next():
                    break

            # Drain + keep pipeline full
            # On exception: kill all current and future tasks and trigger limit events to early end throttler
            while inflight:
                done = next(as_completed(inflight))
                if accumulate_results:
                    if done.exception() is not None:
                        self.stopped = True
                        if self.limit_event is not None:
                            self.limit_event.set()
                        ex.shutdown(wait=False, cancel_futures=True)
                        inflight.remove(done)
                        while inflight:
                            fut = next(as_completed(inflight))
                            inflight.remove(fut)
                        return results, done.exception()
                    results.append(done.result())
                else:
                    done.result()  # propagate exception from that task
                inflight.remove(done)
                submit_next()  # backfill one; keeps in-flight <= max_workers

            if accumulate_results:
                return results, None
            else:
                return None


class MultiProcessThrottler:
    def __init__(
            self,
            hook: Callable[..., Any],
            base_kwargs: Optional[dict] = None,
            sleep_s: float = 0.0,
            max_workers: Optional[int] = None,
            limit_event: object | bool | None = None,  # prefer Manager().Event() for cross-process signaling
    ) -> None:
        """
        Run `hook(**base_kwargs, throttle=unit.as_payload())` over units concurrently; no aggregation.

        :param hook:            A serializable function (eg. static method, class method, or standalone function
        :param base_kwargs:     kwargs tobe passed to the hook
        :param sleep_s:         Time to sleep between executions
        :param max_workers:     Maximum number of concurrent units to run in parallel.
        :param limit_event:     Event() object that will link all workers
        """

        if sleep_s < 0:
            raise ValueError("sleep_s must be >= 0")
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers must be >= 1")

        cpu_count = os.cpu_count() or 2
        self.max_workers = max(1, (max_workers if max_workers is not None else cpu_count - 1))
        self.hook = hook  # must be picklable (top-level function or picklable callable)
        self.base_kwargs = dict(base_kwargs or {})  # must be picklable
        self.sleep_s = float(sleep_s)
        self.stopped = False

        # Set the event manager for cross-process notifications
        self._manager = None
        if limit_event is True:
            self._manager = Manager()
            self.limit_event = self._manager.Event()  # cross-process event
        elif limit_event:
            # duck-typed: accept any Event-like object with set()/is_set()/clear()
            if not all(hasattr(limit_event, m) for m in ("set", "is_set", "clear")):
                raise TypeError("limit_event must be a multiprocessing Event or Manager().Event() proxy")
            self.limit_event = limit_event
        else:
            self.limit_event = None

    def throttle(self, units: Iterable[UnitLike], accumulate_results: bool = False) -> (
            Tuple[list[Any], BaseException | None] | None):
        """
        Executes the hook over 'units' using max_workers number of concurrent processes.

        :param units:               Iterable of `ThrottleUnit` or dicts with at least a `'kind'` key.
        :param accumulate_results:  If `True`, results are appended after each run.

        :return:                    None or List (f accumulate_results = True)
        """
        results = []
        self.stopped = False
        kwargs = self.base_kwargs.copy()

        if self.limit_event is not None and "limit_event" not in kwargs:
            kwargs["limit_event"] = self.limit_event

        it = iter(units)
        with ProcessPoolExecutor(max_workers=self.max_workers) as ex:
            inflight: set[Future] = set()

            def submit_next() -> bool:
                # stop submitting when limit_event flips
                if self.limit_event and self.limit_event.is_set():
                    return False
                elif self.stopped:
                    return False
                try:
                    raw = next(it)
                except StopIteration:
                    return False
                unit = raw if isinstance(raw, ThrottleUnit) else ThrottleUnit(kind=raw["kind"], meta=raw)
                fut = ex.submit(self.hook, **kwargs, throttle=unit.as_payload())
                inflight.add(fut)
                if self.sleep_s > 0:
                    time.sleep(self.sleep_s)
                return True

            # Prime up to capacity
            for _ in range(self.max_workers):
                if not submit_next():
                    break

            # Drain + keep pipeline full
            try:
                while inflight:
                    done = next(as_completed(inflight))
                    if accumulate_results:
                        results.append(done.result())
                    else:
                        done.result()  # propagate error from a worker
                    inflight.remove(done)
                    submit_next()
            except Exception:
                for f in inflight:
                    f.cancel()

                if self._manager is not None:
                    self._manager.shutdown()
                    self._manager = None
                return results, Exception()  # Does this work?

        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None

        return results, None
