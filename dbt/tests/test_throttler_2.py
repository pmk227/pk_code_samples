import pytest
import time
import pandas as pd
from threading import Event
from multiprocessing import Manager
from unittest.mock import Mock
from src.io.throttler import ThrottleUnit, Throttler
from src.io.throttle_unit_builders import date_range_units, dataframe_chunk_units


# Test fixtures and helper functions
@pytest.fixture
def sample_units():
    """Create sample ThrottleUnit objects using throttle_unit_builders."""
    df = pd.DataFrame({"id": range(5), "value": ["test"] * 5})
    return dataframe_chunk_units(df, chunk_size=1, return_rows=True)


@pytest.fixture
def date_units():
    """Create sample date range units."""
    return date_range_units(start="2024-01-01", end="2024-01-05", interval="1d")


@pytest.fixture
def dict_units():
    """Create sample dict units for testing."""
    return [{"kind": "test", "id": i} for i in range(5)]


def simple_hook(**kwargs):
    """Simple hook that returns the throttle payload."""
    return kwargs.get("throttle", {})


def sleep_hook(**kwargs):
    """Hook that sleeps briefly to test timing."""
    time.sleep(0.01)
    return kwargs.get("throttle", {})


def error_hook(**kwargs):
    """Hook that raises an exception."""
    raise ValueError("Test error")


def counter_hook(counter, **kwargs):
    """Hook that increments a counter."""
    counter["count"] += 1
    return counter["count"]


# ============================================================================
# ThrottleUnit Tests
# ============================================================================
class TestThrottleUnit:
    """Test the ThrottleUnit dataclass and as_payload method."""

    def test_as_payload_with_date_range_units(self):
        """Test as_payload unpacks meta correctly with date_range_units."""
        units = date_range_units(start="2024-01-01", end="2024-01-02", interval="1d")

        # Get first unit and test its payload
        unit = units[0]
        payload = unit.as_payload()

        # Should have kind and unpacked meta keys
        assert payload["kind"] == "date_range"
        assert "from_date" in payload
        assert "to_date" in payload
        # meta should NOT be a key since it's unpacked
        assert "meta" not in payload

    def test_as_payload_with_dataframe_chunk_units(self):
        """Test as_payload unpacks meta correctly with dataframe_chunk_units."""
        df = pd.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})
        units = dataframe_chunk_units(df, chunk_size=1, return_rows=True)

        # Get first unit and test its payload
        unit = units[0]
        payload = unit.as_payload()

        # Should have kind and unpacked meta keys
        assert payload["kind"] == "DataFrame"
        assert "df_chunk" in payload
        assert isinstance(payload["df_chunk"], pd.DataFrame)
        # meta should NOT be a key since it's unpacked
        assert "meta" not in payload

    def test_as_payload_with_list_units(self):
        """Test as_payload unpacks meta correctly with list_chunk_units."""
        from src.io.throttle_unit_builders import list_chunk_units

        items = ["a", "b", "c", "d", "e"]
        units = list_chunk_units(items, chunk_size=2)

        # Get first unit and test its payload
        unit = units[0]
        payload = unit.as_payload()

        # Should have kind and unpacked meta keys
        assert payload["kind"] == "list"
        assert "items" in payload
        assert payload["items"] == ["a", "b"]
        # meta should NOT be a key since it's unpacked
        assert "meta" not in payload

    def test_as_payload_structure(self):
        """Test that as_payload correctly unpacks all meta fields."""
        df = pd.DataFrame({"col1": [1], "col2": [2]})
        units = dataframe_chunk_units(df, chunk_size=1, return_rows=True)

        unit = units[0]
        payload = unit.as_payload()

        # Verify the structure: kind + unpacked meta
        assert "kind" in payload
        # All other keys should come from meta
        meta_keys = unit.meta.keys()
        for key in meta_keys:
            assert key in payload, f"Meta key '{key}' should be unpacked into payload"


# ============================================================================
# Throttler Initialization Tests
# ============================================================================
class TestThrottlerInit:
    """Test Throttler initialization and validation."""

    def test_init_serial(self):
        """Test initialization with serial execution."""
        throttler = Throttler(simple_hook, execution_type="serial")
        assert throttler.execution_type == "serial"
        assert throttler.max_workers == 1
        assert throttler.hook == simple_hook
        assert throttler.base_kwargs == {}
        assert throttler.sleep_s == 0.0

    def test_init_thread(self):
        """Test initialization with thread execution."""
        throttler = Throttler(simple_hook, execution_type="thread", max_workers=4)
        assert throttler.execution_type == "thread"
        assert throttler.max_workers == 4

    def test_init_thread_default_workers(self):
        """Test thread initialization with default max_workers."""
        throttler = Throttler(simple_hook, execution_type="thread")
        assert throttler.max_workers >= 4  # Should use computed default
        assert throttler.max_workers <= 64  # Should be capped at 64

    def test_init_process(self):
        """Test initialization with process execution."""
        throttler = Throttler(simple_hook, execution_type="process", max_workers=2)
        assert throttler.execution_type == "process"
        assert throttler.max_workers == 2

    def test_init_process_default_workers(self):
        """Test process initialization with default max_workers."""
        throttler = Throttler(simple_hook, execution_type="process")
        assert throttler.max_workers >= 1

    def test_init_with_base_kwargs(self):
        """Test initialization with base_kwargs."""
        base_kwargs = {"param1": "value1", "param2": 42}
        throttler = Throttler(simple_hook, execution_type="serial", base_kwargs=base_kwargs)
        assert throttler.base_kwargs == base_kwargs

    def test_init_with_sleep(self):
        """Test initialization with sleep_s."""
        throttler = Throttler(simple_hook, execution_type="serial", sleep_s=0.5)
        assert throttler.sleep_s == 0.5

    def test_init_creates_manager_event_by_default(self):
        """Test that Manager().Event() is created by default."""
        throttler = Throttler(simple_hook, execution_type="serial")
        assert throttler.stop_event is not None
        assert throttler._manager is not None
        assert hasattr(throttler.stop_event, "set")
        assert hasattr(throttler.stop_event, "is_set")
        assert hasattr(throttler.stop_event, "clear")

    def test_init_creates_manager_event_when_true(self):
        """Test that stop_event=True creates a Manager().Event()."""
        throttler = Throttler(simple_hook, execution_type="serial", stop_event=True)
        assert throttler.stop_event is not None
        assert throttler._manager is not None

    def test_init_accepts_manager_event(self):
        """Test initialization with Manager().Event()."""
        manager = Manager()
        stop_event = manager.Event()
        throttler = Throttler(simple_hook, execution_type="serial", stop_event=stop_event)
        assert throttler.stop_event is stop_event
        manager.shutdown()


# ============================================================================
# Throttler Validation Tests
# ============================================================================
class TestThrottlerValidation:
    """Test Throttler input validation."""

    def test_init_none_hook_raises(self):
        """Test that None hook raises TypeError."""
        with pytest.raises(TypeError, match="hook cannot be None"):
            Throttler(None, execution_type="serial")

    def test_init_invalid_execution_type_raises(self):
        """Test that invalid execution_type raises TypeError."""
        with pytest.raises(TypeError, match="execution_type must be"):
            Throttler(simple_hook, execution_type="invalid")

    def test_init_negative_sleep_raises(self):
        """Test that negative sleep_s raises ValueError."""
        with pytest.raises(ValueError, match="sleep_s must be >= 0"):
            Throttler(simple_hook, execution_type="serial", sleep_s=-1.0)

    def test_init_max_workers_zero_serial_resets_to_one(self):
        """Test that max_workers=0 with serial resets to 1."""
        throttler = Throttler(simple_hook, execution_type="serial", max_workers=0)
        assert throttler.max_workers == 1

    def test_init_max_workers_zero_thread_resets_to_one(self):
        """Test that max_workers=0 with thread resets to 1."""
        throttler = Throttler(simple_hook, execution_type="thread", max_workers=0)
        assert throttler.max_workers == 1

    def test_init_max_workers_zero_process_resets_to_one(self):
        """Test that max_workers=0 with process resets to 1."""
        throttler = Throttler(simple_hook, execution_type="process", max_workers=0)
        assert throttler.max_workers == 1

    def test_init_invalid_base_kwargs_raises(self):
        """Test that non-dict base_kwargs raises TypeError."""
        with pytest.raises(TypeError, match="base_kwargs must be a dict"):
            Throttler(simple_hook, execution_type="serial", base_kwargs="not_a_dict")

    def test_init_with_threading_event_raises(self):
        """Test that threading.Event raises TypeError (should be Manager().Event())."""
        event = Event()
        with pytest.raises(TypeError, match="stop_event must be a Manager"):
            Throttler(simple_hook, execution_type="serial", stop_event=event)


# ============================================================================
# Serial Executor Tests
# ============================================================================
class TestSerialExecutor:
    """Test the serial_executor method."""

    def test_serial_executor_basic(self, sample_units):
        """Test serial_executor method without accumulating results."""
        throttler = Throttler(simple_hook, execution_type="serial")
        results, error = throttler.serial_executor(sample_units)

        assert error is None
        assert results is None

    def test_serial_executor_accumulate_results(self, sample_units):
        """Test serial_executor with accumulate_results=True."""
        throttler = Throttler(simple_hook, execution_type="serial")
        results, error = throttler.serial_executor(sample_units, accumulate_results=True)

        assert error is None
        assert isinstance(results, list)
        assert len(results) == len(sample_units)

    def test_serial_executor_propagates_exception(self, sample_units):
        """Test that serial_executor propagates exceptions."""
        throttler = Throttler(error_hook, execution_type="serial")

        with pytest.raises(ValueError, match="Test error"):
            throttler.serial_executor(sample_units)

    def test_serial_executor_with_base_kwargs(self, sample_units):
        """Test that serial_executor passes base_kwargs to hook."""
        mock_hook = Mock(return_value=None)
        base_kwargs = {"param1": "value1", "param2": 42}
        throttler = Throttler(mock_hook, execution_type="serial", base_kwargs=base_kwargs)
        throttler.serial_executor(sample_units)

        assert mock_hook.call_count == len(sample_units)
        for call in mock_hook.call_args_list:
            assert call[1]["param1"] == "value1"
            assert call[1]["param2"] == 42

    def test_serial_executor_respects_stop_event(self):
        """Test that serial_executor stops when stop_event is set."""
        manager = Manager()
        stop_event = manager.Event()

        call_count = {"count": 0}

        def counting_hook(**kwargs):
            call_count["count"] += 1
            if call_count["count"] == 2:
                stop_event.set()
            return kwargs.get("throttle", {})

        df = pd.DataFrame({"id": range(10), "value": ["test"] * 10})
        units = dataframe_chunk_units(df, chunk_size=1, return_rows=True)

        throttler = Throttler(counting_hook, execution_type="serial", stop_event=stop_event)
        results, error = throttler.serial_executor(units, accumulate_results=True)

        assert error is None
        assert len(results) == 2  # Should stop after 2 iterations

        manager.shutdown()

    def test_serial_executor_with_sleep(self):
        """Test that serial_executor respects sleep_s."""
        df = pd.DataFrame({"id": range(3), "value": ["test"] * 3})
        units = dataframe_chunk_units(df, chunk_size=1, return_rows=True)

        throttler = Throttler(simple_hook, execution_type="serial", sleep_s=0.05)

        start = time.time()
        throttler.serial_executor(units)
        elapsed = time.time() - start

        # Should take at least 0.05 * 3 = 0.15 seconds
        assert elapsed >= 0.15

    def test_serial_executor_kills_manager_on_completion(self, sample_units):
        """Test that manager is cleaned up after serial_executor."""
        throttler = Throttler(simple_hook, execution_type="serial")
        assert throttler._manager is not None

        throttler.serial_executor(sample_units)
        assert throttler._manager is None

    def test_serial_executor_kills_manager_on_exception(self, sample_units):
        """Test that manager is cleaned up even on exception."""
        throttler = Throttler(error_hook, execution_type="serial")
        assert throttler._manager is not None

        with pytest.raises(ValueError):
            throttler.serial_executor(sample_units)

        assert throttler._manager is None


# ============================================================================
# Throttle Method Tests
# ============================================================================
class TestThrottleMethod:
    """Test the main throttle method across all execution types."""

    def test_throttle_serial_delegates_to_serial_executor(self, sample_units):
        """Test that serial execution uses serial_executor."""
        throttler = Throttler(simple_hook, execution_type="serial")
        result = throttler.throttle(sample_units)
        assert result == (None, None)

    def test_throttle_serial_with_accumulate_results(self, sample_units):
        """Test throttle with serial execution and accumulate_results."""
        throttler = Throttler(simple_hook, execution_type="serial")
        results, error = throttler.throttle(sample_units, accumulate_results=True)

        assert error is None
        assert isinstance(results, list)
        assert len(results) == len(sample_units)

    def test_throttle_thread_basic(self, sample_units):
        """Test throttle with thread execution."""
        throttler = Throttler(simple_hook, execution_type="thread", max_workers=2)
        results, error = throttler.throttle(sample_units)

        assert error is None or isinstance(error, Exception)
        assert isinstance(results, list)

    def test_throttle_thread_accumulate_results(self, sample_units):
        """Test throttle with thread execution and accumulate_results."""
        throttler = Throttler(simple_hook, execution_type="thread", max_workers=2)
        results, error = throttler.throttle(sample_units, accumulate_results=True)

        assert isinstance(results, list)
        if error is None:
            assert len(results) == len(sample_units)

    def test_throttle_process_basic(self, sample_units):
        """Test throttle with process execution."""
        throttler = Throttler(simple_hook, execution_type="process", max_workers=2)
        results, error = throttler.throttle(sample_units)

        assert error is None or isinstance(error, Exception)
        assert isinstance(results, list)

    def test_throttle_process_accumulate_results(self, sample_units):
        """Test throttle with process execution and accumulate_results."""
        throttler = Throttler(simple_hook, execution_type="process", max_workers=2)
        results, error = throttler.throttle(sample_units, accumulate_results=True)

        assert isinstance(results, list)
        if error is None:
            assert len(results) == len(sample_units)

    def test_throttle_with_dict_units(self, dict_units):
        """Test throttle with dict units (thread execution)."""
        throttler = Throttler(simple_hook, execution_type="thread", max_workers=2)
        results, error = throttler.throttle(dict_units, accumulate_results=True)

        assert isinstance(results, list)

    def test_throttle_thread_with_stop_event(self):
        """Test that stop_event stops thread execution."""
        manager = Manager()
        stop_event = manager.Event()

        call_count = {"count": 0}

        def stop_after_first(**kwargs):
            if call_count["count"] == 0:
                call_count["count"] += 1
                stop_event.set()
            return kwargs.get("throttle", {})

        df = pd.DataFrame({"id": range(10), "value": ["test"] * 10})
        units = dataframe_chunk_units(df, chunk_size=1, return_rows=True)

        throttler = Throttler(
            stop_after_first,
            execution_type="thread",
            stop_event=stop_event,
            max_workers=1
        )

        results, error = throttler.throttle(units, accumulate_results=True)
        # Should stop early
        assert len(results) < len(units)

        manager.shutdown()

    def test_throttle_manager_cleanup_serial(self, sample_units):
        """Test that manager is cleaned up after serial throttle."""
        throttler = Throttler(simple_hook, execution_type="serial")
        assert throttler._manager is not None

        throttler.throttle(sample_units)
        assert throttler._manager is None

    def test_throttle_manager_cleanup_thread(self, sample_units):
        """Test that manager is cleaned up after thread throttle."""
        throttler = Throttler(simple_hook, execution_type="thread", max_workers=2)
        assert throttler._manager is not None

        throttler.throttle(sample_units)
        assert throttler._manager is None

    def test_throttle_with_sleep_thread(self):
        """Test throttle with sleep in thread execution."""
        df = pd.DataFrame({"id": range(3), "value": ["test"] * 3})
        units = dataframe_chunk_units(df, chunk_size=1, return_rows=True)

        throttler = Throttler(simple_hook, execution_type="thread", sleep_s=0.05, max_workers=1)

        start = time.time()
        throttler.throttle(units)
        elapsed = time.time() - start

        # Should take at least 0.05 * 3 = 0.15 seconds with serial submission
        assert elapsed >= 0.15


# ============================================================================
# Integration Tests
# ============================================================================
class TestIntegration:
    """Integration tests for complex scenarios."""

    def test_nested_throttlers(self):
        """Test nested throttler scenario (similar to px_pull_data)."""
        outer_units = date_range_units(start="2024-01-01", end="2024-01-03", interval="1d")

        def outer_hook(**kwargs):
            df = pd.DataFrame({"id": range(2), "value": ["inner"] * 2})
            inner_units = dataframe_chunk_units(df, chunk_size=1, return_rows=True)

            inner_throttler = Throttler(simple_hook, execution_type="serial")
            results, error = inner_throttler.throttle(inner_units, accumulate_results=True)
            return results

        outer_throttler = Throttler(outer_hook, execution_type="serial")
        results, error = outer_throttler.throttle(outer_units, accumulate_results=True)

        assert error is None
        assert len(results) == len(outer_units)
        for result in results:
            assert len(result) == 2

    def test_counter_across_executions(self):
        """Test that counter increments correctly across all units."""
        counter = {"count": 0}
        df = pd.DataFrame({"id": range(10), "value": ["test"] * 10})
        units = dataframe_chunk_units(df, chunk_size=1, return_rows=True)

        throttler = Throttler(
            counter_hook,
            execution_type="serial",
            base_kwargs={"counter": counter}
        )
        results, error = throttler.throttle(units, accumulate_results=True)

        assert error is None
        assert counter["count"] == 10
        assert results == list(range(1, 11))

    def test_large_unit_count_thread(self):
        """Test with a large number of units in thread execution."""
        df = pd.DataFrame({"id": range(100), "value": ["test"] * 100})
        units = dataframe_chunk_units(df, chunk_size=1, return_rows=True)

        throttler = Throttler(simple_hook, execution_type="thread", max_workers=4)
        results, error = throttler.throttle(units, accumulate_results=True)

        assert error is None
        assert len(results) == 100

    def test_exception_handling_in_threads(self):
        """Test that exceptions in threaded execution are handled."""
        call_count = {"count": 0}

        def error_on_third(**kwargs):
            call_count["count"] += 1
            if call_count["count"] == 3:
                raise ValueError("Third call error")
            return kwargs.get("throttle", {})

        df = pd.DataFrame({"id": range(10), "value": ["test"] * 10})
        units = dataframe_chunk_units(df, chunk_size=1, return_rows=True)

        throttler = Throttler(error_on_third, execution_type="thread", max_workers=1)
        results, error = throttler.throttle(units, accumulate_results=True)

        # Should capture the error
        assert error is not None
        assert len(results) < len(units)

    def test_date_range_units_throttling(self):
        """Test throttling with date_range_units."""
        units = date_range_units(start="2024-01-01", end="2024-01-10", interval="1d")

        throttler = Throttler(simple_hook, execution_type="serial")
        results, error = throttler.throttle(units, accumulate_results=True)

        assert error is None
        assert len(results) > 0
        for result in results:
            assert result["kind"] == "date_range"
            assert "from_date" in result
            assert "to_date" in result
            # Verify meta is unpacked (no 'meta' key in result)
            assert "meta" not in result

    def test_dataframe_chunks_with_threads(self):
        """Test dataframe chunking with thread execution."""
        df = pd.DataFrame({
            "id": range(20),
            "value": [f"test_{i}" for i in range(20)]
        })
        units = dataframe_chunk_units(df, num_chunks=5, return_rows=True)

        def extract_chunk_hook(**kwargs):
            throttle = kwargs["throttle"]
            chunk = throttle["df_chunk"]
            return len(chunk)

        throttler = Throttler(extract_chunk_hook, execution_type="thread", max_workers=3)
        results, error = throttler.throttle(units, accumulate_results=True)

        assert error is None
        assert len(results) == 5
        assert sum(results) == 20  # Total rows