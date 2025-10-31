# src/utilities/concurrency.py
from typing import Protocol, Union
from threading import Lock as ThreadLock
from multiprocessing import Manager


class CounterProtocol(Protocol):
    def increment(self, delta: int = 1) -> int: ...
    def get(self) -> int: ...
    def set(self, value: int) -> None: ...


class ThreadCounter:
    """Thread-safe counter for single-process multithreading."""
    def __init__(self, initial_value: int = 0) -> None:
        self._count = int(initial_value)
        self._lock = ThreadLock()

    def increment(self, delta: int = 1) -> int:
        if not isinstance(delta, int):
            raise TypeError("delta must be an int")
        with self._lock:
            self._count += delta
            return self._count

    def get(self) -> int:
        return self._count

    def set(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError("value must be an int")
        with self._lock:
            self._count = value


class ProcessCounter:
    """
    Cross-process counter using a Manager-backed Namespace + Lock.
    Safe to pass as an argument into ProcessPoolExecutor tasks.
    """
    def __init__(self, manager: Manager, initial_value: int = 0) -> None:
        if not isinstance(initial_value, int):
            raise TypeError("initial_value must be an int")
        self._ns = manager.Namespace()
        self._ns.count = int(initial_value)
        self._lock = manager.Lock()

    def increment(self, delta: int = 1) -> int:
        if not isinstance(delta, int):
            raise TypeError("delta must be an int")
        with self._lock:
            self._ns.count += delta
            return self._ns.count

    def get(self) -> int:
        return int(self._ns.count)

    def set(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError("value must be an int")
        with self._lock:
            self._ns.count = int(value)


def create_shared_counter(execution_type: str, *, manager: Union[Manager, None] = None, initial_value: int = 0) -> CounterProtocol:
    """
    Factory that returns a thread- or process-safe counter.
    - For 'process', provide a Manager (caller owns lifecycle).
    - For 'thread' or 'serial', returns a ThreadCounter.
    """
    normalized = {
        "thread": "thread", "threads": "thread", "multithread": "thread", "multithreaded": "thread",
        "process": "process", "processes": "process", "multiprocess": "process", "multiprocessed": "process",
        "serial": "serial", "none": "serial", "single": "serial",
    }.get((execution_type or "serial").lower(), None)

    if normalized is None:
        raise ValueError("Invalid execution_type")

    if normalized == "process":
        if manager is None:
            raise ValueError("Manager is required for 'process' counters")
        return ProcessCounter(manager=manager, initial_value=initial_value)

    # thread or serial → simple in-process counter
    return ThreadCounter(initial_value=initial_value)
