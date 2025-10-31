# src/io/throttler.py
import os
import time
from concurrent.futures import as_completed, Future, ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from threading import Event
from typing import Any, Callable, Iterable, Mapping, Optional, Union, Tuple, Literal

from multiprocessing import Manager


@dataclass(frozen=True, slots=True)
class ThrottleUnit:
    kind: str                       # "date" | "date_range" | "list"
    meta: Mapping[str, Any]         # e.g. {"from": "...", "to": "..."} or {"items": [...]}

    def __post_init__(self):
        if not isinstance(self.meta, Mapping):
            raise TypeError(
                f"ThrottleUnit.meta must be a Mapping, got {type(self.meta).__name__}"
            )

    def as_payload(self) -> dict[str, Any]:
        return {"kind": self.kind, **self.meta}

UnitLike = Union[ThrottleUnit, Mapping[str, Any]]  # allow dicts too if you want

class Throttler:
    def __init__(
        self,
        hook: Callable[..., Any],
        execution_type: Literal["thread","process","serial"],
        base_kwargs: Optional[dict] = None,
        sleep_s: float = 0.0,
        max_workers: Optional[int] = None,
        stop_event: Event | bool | None = None,
    ) -> None:
        """
        Run `hook(**base_kwargs, throttle=unit.as_payload())` over units concurrently; no aggregation.

        :param hook:                A serializable function (eg. static method, class method, or standalone function
        :param execution_type:      The execution type to run the ETL on, thread, process, or serial
        :param base_kwargs:         (opt) kwargs tobe passed to the hook
        :param sleep_s:             (opt) Time to sleep between executions
        :param max_workers:         (opt) Maximum number of concurrent units to run in parallel.
        :param stop_event:          Bool or Event() object that will link all workers; if True will create event,
                                        if an Event() object is passed instead it will be used
        """
        self.hook = hook
        self.execution_type = execution_type
        self.sleep_s = sleep_s
        self.base_kwargs = base_kwargs or dict()

        if stop_event is None or isinstance(stop_event, bool):
            self._manager = Manager()
            self.stop_event = self._manager.Event()
        elif stop_event is not None:
            self._manager = None
            self.stop_event = stop_event
        else:
            raise TypeError("stop_event must be None, Bool, or an Event")

        match self.execution_type:
            case "serial":
                self.max_workers = 1
            case "thread":
                n = min(64, max(4, (os.cpu_count() or 2) + 12))
                self.max_workers = max(1, (max_workers if max_workers is not None else n))
            case "process":
                self.max_workers = max(1, (max_workers if max_workers is not None else os.cpu_count() - 1))
        self.validate_init_inputs()

    def validate_init_inputs(self):
        if self.hook is None:
            raise TypeError("hook cannot be None")
        if self.execution_type is None or self.execution_type not in ("thread","process","serial"):
            raise TypeError("execution_type must be 'thread', 'process' or 'serial'")
        if self.sleep_s < 0:
            raise ValueError("sleep_s must be >= 0")
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if not isinstance(self.base_kwargs, dict):
            raise TypeError("base_kwargs must be a dict") # noqa
        if not all(hasattr(self.stop_event, m) for m in ("set", "is_set", "clear")):
            raise TypeError("stop_event must be a Manager().Event() proxy")
        elif isinstance(self.stop_event, Event):
            raise TypeError("stop_event must be a Manager().Event() proxy, not an Event()")

    def _kill_manager(self):
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None

    def serial_executor(self, units: Iterable[UnitLike], accumulate_results: bool = False) -> (
            Tuple[list[Any]|None, BaseException|None]|None):
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
            if self.stop_event.is_set():
                break
            unit = u
            try:
                if accumulate_results:
                    results.append(self.hook(**self.base_kwargs, throttle=unit.as_payload()))
                else:
                    self.hook(**self.base_kwargs, throttle=unit.as_payload())
            except Exception as e:
                self._kill_manager()
                raise e

            if self.sleep_s > 0:
                time.sleep(self.sleep_s)

        self._kill_manager()
        if accumulate_results:
            return results, None
        return None, None

    def throttle(self, units: Iterable[UnitLike], accumulate_results: bool = False) -> (
            Tuple[list[Any], BaseException | None] | None):
        """
        Executes the hook over 'units' using max_workers number of concurrent threads.

        :param units:               Iterable of `ThrottleUnit` or dicts with at least a `'kind'` key.
        :param accumulate_results:  If `True`, results are appended after each run.

        :return:                    None or List (f accumulate_results = True)
        """
        results = []
        if self.base_kwargs is not None:
            kwargs = self.base_kwargs.copy()

        match self.execution_type:
            case "serial":
                return self.serial_executor(units, accumulate_results=accumulate_results)
            case "thread":
                pool_executor = ThreadPoolExecutor
            case "process":
                pool_executor = ProcessPoolExecutor

        it = iter(units)
        inflight: set[Future] = set()

        with pool_executor(max_workers=self.max_workers) as executor:
            def submit_next() -> bool:
                if self.stop_event.is_set():
                    return False
                try:
                    raw = next(it)
                except StopIteration:
                    return False
                unit = raw if isinstance(raw, ThrottleUnit) else ThrottleUnit(kind=raw["kind"], meta=raw)
                fut = executor.submit(self.hook, **kwargs, throttle=unit.as_payload())
                inflight.add(fut)
                if self.sleep_s > 0:
                    time.sleep(self.sleep_s)
                return True


            for _ in range(self.max_workers):
                if self.stop_event.is_set():
                    for f in inflight:
                        f.cancel()
                    break
                if not submit_next():
                    break

            # Drain + keep pipeline full
                try:
                    while inflight:
                        if self.stop_event.is_set():
                            break
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

                    self._kill_manager()
                    return results, Exception()  # Does this work?

            self._kill_manager()

            return results, None



def NoOp(throttle, **kwargs) -> Any:
    df = throttle.get('df_chunk')
    for row in df.itertuples():
        print(row.destination_filepath)
    return 0

if __name__ == "__main__":
    import pandas as pd
    from src.io.throttle_unit_builders import dataframe_single_unit

    test_df = pd.DataFrame({
        "row_uid": [f"id_{i}" for i in range(1, 6)],
        "api_endpoint": [
            "endpoint/a",
            "endpoint/b",
            "endpoint/c",
            "endpoint/d",
            "endpoint/e"
        ],
        "destination_filepath": [
            f"/tmp/file_{i}.csv" for i in range(1, 6)
        ]
    })

    test_throttle_unit = dataframe_single_unit(test_df)

    a = Throttler(NoOp, execution_type="serial", stop_event=False)
    b = a.throttle(test_throttle_unit, accumulate_results=True)
    pass

