from io import BytesIO, StringIO
import json
import importlib.util
from pathlib import Path
from threading import Event
from typing import Any

import pandas as pd
import pytest
from requests.models import Response
from unittest.mock import MagicMock, patch

from src.io.data_extractor import ApiExtractor, ExtractorFactory, FileExtractor


# ---------- helpers ----------

def has_pkg(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


# ---------- fixtures ----------

@pytest.fixture
def tmp_files(tmp_path: Path) -> dict[str, Path | None]:
    files: dict[str, Path | None] = {}

    # CSV
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    files["csv"] = csv_path

    # JSON (object -> dict[str, Any])
    json_obj: dict[str, Any] = {"host": "localhost", "port": 5432, "features": {"x": True}}
    json_path = tmp_path / "config.json"
    json_path.write_text(json.dumps(json_obj), encoding="utf-8")
    files["json"] = json_path

    # Parquet (requires pyarrow or fastparquet)
    parquet_path = tmp_path / "table.parquet"
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    files["parquet"] = None
    for engine in ("pyarrow", "fastparquet"):
        if has_pkg(engine):
            df.to_parquet(parquet_path, engine=engine)
            files["parquet"] = parquet_path
            break

    # Excel .xlsx (requires openpyxl for read)
    xlsx_path = tmp_path / "book.xlsx"
    files["xlsx"] = None
    if has_pkg("openpyxl"):
        df.to_excel(xlsx_path, index=False, engine="openpyxl")
        files["xlsx"] = xlsx_path

    # HDF5 (requires tables)
    h5_path = tmp_path / "store.h5"
    files["h5"] = None
    if has_pkg("tables"):
        df.to_hdf(h5_path, key="df", mode="w")
        files["h5"] = h5_path

    # Feather (requires pyarrow)
    feather_path = tmp_path / "table.feather"
    files["feather"] = None
    if has_pkg("pyarrow"):
        df.to_feather(feather_path)
        files["feather"] = feather_path

        # Also produce a .arrow file via Feather/IPC
        arrow_path = tmp_path / "table.arrow"
        df.to_feather(arrow_path)
        files["arrow"] = arrow_path
    else:
        files["arrow"] = None

    # Pickle (DataFrame)
    pkl_df_path = tmp_path / "frame.pkl"
    pd.to_pickle(df, pkl_df_path)
    files["pkl_df"] = pkl_df_path

    # Pickle (non-DataFrame)
    pkl_other_path = tmp_path / "not_df.pkl"
    pd.to_pickle({"a": 1}, pkl_other_path)
    files["pkl_other"] = pkl_other_path

    return files


@pytest.fixture
def tmp_json_records(tmp_path: Path) -> Path:
    """JSON array of dicts for DataFrame tests"""
    records = [
        {"host": "localhost", "port": 5432, "features": {"x": True}},
        {"host": "remote", "port": 5433, "features": {"x": False}},
    ]
    json_path = tmp_path / "records.json"
    json_path.write_text(json.dumps(records), encoding="utf-8")
    return json_path


# ---------- tests: core behaviors ----------

def test_csv_to_dataframe(tmp_files: dict[str, Path | None]) -> None:
    fe = FileExtractor(tmp_files["csv"])
    df = fe.extract(nrows=1)  # kwargs pass-through
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 1


def test_json_to_dict(tmp_files: dict[str, Path | None]) -> None:
    fe = FileExtractor(tmp_files["json"])
    cfg = fe.extract()
    assert isinstance(cfg, dict)
    assert cfg["host"] == "localhost"
    assert isinstance(cfg["features"], dict)


@pytest.mark.skipif(not (has_pkg("pyarrow") or has_pkg("fastparquet")), reason="No parquet engine installed")
def test_parquet_to_dataframe(tmp_files: dict[str, Path | None]) -> None:
    pq = tmp_files["parquet"]
    if pq is None:
        pytest.skip("Could not create parquet file without an engine available.")
    fe = FileExtractor(pq)
    df = fe.extract()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["x", "y"]


@pytest.mark.skipif(not has_pkg("openpyxl"), reason="openpyxl not installed for .xlsx")
def test_xlsx_to_dataframe(tmp_files: dict[str, Path | None]) -> None:
    xlsx = tmp_files["xlsx"]
    if xlsx is None:
        pytest.skip("Could not create .xlsx file (missing writer engine).")
    fe = FileExtractor(xlsx)
    df = fe.extract()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["x", "y"]


@pytest.mark.skipif(not has_pkg("tables"), reason="PyTables (tables) not installed for HDF5")
def test_hdf_to_dataframe(tmp_files: dict[str, Path | None]) -> None:
    h5 = tmp_files["h5"]
    if h5 is None:
        pytest.skip("Could not create HDF5 file (PyTables not installed).")
    fe = FileExtractor(h5)
    df = fe.extract(key="df")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["x", "y"]


@pytest.mark.skipif(not has_pkg("pyarrow"), reason="pyarrow not installed for Feather")
def test_feather_to_dataframe(tmp_files: dict[str, Path | None]) -> None:
    fp = tmp_files["feather"]
    if fp is None:
        pytest.skip("Could not create Feather file (pyarrow not installed).")
    fe = FileExtractor(fp)
    df = fe.extract()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["x", "y"]


@pytest.mark.skipif(not has_pkg("pyarrow"), reason="pyarrow not installed for Arrow/Feather")
def test_arrow_to_dataframe(tmp_files: dict[str, Path | None]) -> None:
    ap = tmp_files["arrow"]
    if ap is None:
        pytest.skip("Could not create .arrow file (pyarrow not installed).")
    fe = FileExtractor(ap)
    df = fe.extract()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["x", "y"]


def test_pickle_dataframe_ok(tmp_files: dict[str, Path | None]) -> None:
    pkl = tmp_files["pkl_df"]
    fe = FileExtractor(pkl)
    df = fe.extract()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["x", "y"]


def test_unsupported_extension(tmp_path: Path) -> None:
    p = tmp_path / "note.txt"
    p.write_text("hello", encoding="utf-8")
    with pytest.raises(NotImplementedError):
        _ = FileExtractor(p).extract()


def test_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        _ = FileExtractor(missing)


def test_json_decode_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json}", encoding="utf-8")
    fe = FileExtractor(bad)
    with pytest.raises(json.JSONDecodeError):
        _ = fe.extract()


def test_json_to_dataframe_flat(tmp_files: dict[str, Path | None]) -> None:
    fe = FileExtractor(tmp_files["json"])
    df = fe.extract(as_dataframe=True, flatten=True)
    assert isinstance(df, pd.DataFrame)
    assert "host" in df.columns
    assert "features.x" in df.columns  # flattened nested field


def test_json_to_dataframe_no_flat(tmp_json_records: Path) -> None:
    fe = FileExtractor(tmp_json_records)
    df = fe.extract(as_dataframe=True, flatten=False)
    assert isinstance(df, pd.DataFrame)
    assert "features" in df.columns
    assert isinstance(df.loc[0, "features"], dict)


def test_json_embedded_dict(tmp_path: Path) -> None:
    data = {
        "metadata": {"version": 1},
        "results": [{"id": 1, "value": 10}, {"id": 2, "value": 20}]
    }
    json_path = tmp_path / "nested.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")
    fe = FileExtractor(json_path)
    df = fe.extract(as_dataframe=True, embedded_dict="results")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["id", "value"]
    assert len(df) == 2


def test_json_embedded_dict_flattened(tmp_path: Path) -> None:
    data = {
        "metadata": {"version": 1},
        "results": [
            {"id": 1, "data": {"score": 95, "passed": True}},
            {"id": 2, "data": {"score": 88, "passed": False}}
        ]
    }
    json_path = tmp_path / "nested_flat.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")
    fe = FileExtractor(json_path)
    df = fe.extract(as_dataframe=True, embedded_dict="results", flatten=True)

    assert isinstance(df, pd.DataFrame)
    assert "id" in df.columns
    assert "data.score" in df.columns
    assert "data.passed" in df.columns
    assert df.loc[0, "data.passed"] == True


# ---------- helpers ----------
def make_mock_response(content, content_type, status_code=200):
    mock = MagicMock(spec=Response)
    mock.status_code = status_code
    mock.ok = status_code == 200
    mock.headers = {"Content-Type": content_type}
    mock.encoding = "utf-8"
    mock.text = content
    mock.content = content.encode("utf-8")
    mock.json = lambda: json.loads(content)
    return mock


# ---------- fixtures ----------
@pytest.fixture
def api_reader_obj():
    return ApiExtractor(base_url="https://example.com")


@pytest.fixture
def patched_session_get():
    with patch("requests.Session.get") as patched:
        yield patched


# ---------- tests ----------
def test_api_json_flat(api_reader_obj, patched_session_get):
    data = {"results": [{"a": 1, "b": 2}]}
    patched_session_get.return_value = make_mock_response(json.dumps(data), "application/json")

    df = api_reader_obj.extract("json", embedded_dict="results")
    assert isinstance(df, pd.DataFrame)
    assert df.loc[0, "a"] == 1


def test_api_csv_text(api_reader_obj, patched_session_get):
    csv = "a,b\n1,2\n3,4\n"
    patched_session_get.return_value = make_mock_response(csv, "text/csv")

    text = api_reader_obj.extract("csv", as_dataframe=False, return_format="csv")
    assert isinstance(text, str)
    assert "1,2" in text


def test_api_xml_to_dataframe(api_reader_obj, patched_session_get):
    xml = """<?xml version="1.0"?><root><items><item><x>5</x></item></items></root>"""
    patched_session_get.return_value = make_mock_response(xml, "application/xml")

    df = api_reader_obj.extract("xml", embedded_dict="root", flatten=True)
    assert isinstance(df, pd.DataFrame)
    assert "items.item" in df.columns or isinstance(df.iloc[0], pd.Series)


def test_api_unauthorized_sets_none(api_reader_obj, patched_session_get, capsys):
    patched_session_get.return_value = make_mock_response("nope", "application/json", status_code=401)

    result = api_reader_obj.extract("fail", as_dataframe=False)
    captured = capsys.readouterr()
    assert "Unauthorized" in captured.out
    assert result is None


def test_api_daily_limit_sets_flag(api_reader_obj, patched_session_get):
    patched_session_get.return_value = make_mock_response("limit", "application/json", status_code=403)

    assert not api_reader_obj.stop_event.is_set()
    result = api_reader_obj.extract("limit", as_dataframe=False)
    assert result is None
    assert api_reader_obj.stop_event.is_set()


def test_api_retries_on_429(api_reader_obj, patched_session_get):
    resp_429 = make_mock_response("wait", "application/json", status_code=429)
    success = make_mock_response(json.dumps({"x": 1}), "application/json")
    patched_session_get.side_effect = [resp_429, success]

    result = api_reader_obj.extract("retry", as_dataframe=False, retry_delay=0.01, max_retries=1)
    assert isinstance(result, dict)
    assert result["x"] == 1


def test_api_raises_on_unknown_content_type(api_reader_obj, patched_session_get):
    bad = make_mock_response("weird", "application/octet-stream")
    patched_session_get.return_value = bad

    with pytest.raises(ValueError):
        _ = api_reader_obj.extract("bad")


def test_api_raises_on_invalid_method(api_reader_obj):
    with pytest.raises(NotImplementedError):
        api_reader_obj.extract("bad", method="POST")


def test_api_raises_if_max_retries_set_without_retry_delay(api_reader_obj):
    with pytest.raises(ValueError):
        api_reader_obj.extract("anything", max_retries=2, retry_delay=None)


def test_api_parquet_returns_bytes(api_reader_obj, patched_session_get):
    binary_data = b"parquet-binary-content"
    mock = make_mock_response("", "application/octet-stream")
    mock.content = binary_data
    patched_session_get.return_value = mock

    result = api_reader_obj.extract("parquet", as_dataframe=False, return_format="parquet")
    assert isinstance(result, bytes)
    assert result == binary_data


def test_api_shared_event_flags_all_instances(patched_session_get):
    event = Event()
    reader1 = ApiExtractor("https://api.com", stop_event=event)
    reader2 = ApiExtractor("https://api.com", stop_event=event)

    patched_session_get.return_value = make_mock_response("limit", "application/json", status_code=403)

    result1 = reader1.extract("limit", as_dataframe=False)
    assert result1 is None
    assert event.is_set()  # shared flag was set

    # second reader should immediately skip request
    result2 = reader2.extract("limit", as_dataframe=False)
    assert result2 is None
    assert patched_session_get.call_count == 1  # second call was skipped


def test_api_binary_save_compatible(api_reader_obj, patched_session_get, tmp_path):
    data = b"binary-parquet-or-image"
    mock = make_mock_response("", "application/octet-stream")
    mock.content = data
    patched_session_get.return_value = mock

    content = api_reader_obj.extract("parquet", as_dataframe=False, return_format="parquet")
    file_path = tmp_path / "output.parquet"
    file_path.write_bytes(content)

    assert file_path.exists()
    assert file_path.read_bytes() == data


def test_api_csv_to_dataframe_end_to_end(api_reader_obj, patched_session_get):
    csv = "a,b\n1,2\n3,4\n"
    patched_session_get.return_value = make_mock_response(csv, "text/csv")

    df = api_reader_obj.extract("csv", as_dataframe=True, return_format="csv")

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)
    assert df.iloc[0].to_list() == [1, 2]


def test_api_parquet_to_dataframe_end_to_end(api_reader_obj, patched_session_get, monkeypatch):
    binary_data = b"parquet-binary-content"
    mock = make_mock_response("", "application/octet-stream")
    mock.content = binary_data
    patched_session_get.return_value = mock

    calls = {"called": False, "buf_is_filelike": False}

    def fake_read_parquet(buf, *args, **kwargs):
        # duck-typed check: should be file-like (BytesIO) with a .read()
        calls["called"] = True
        calls["buf_is_filelike"] = hasattr(buf, "read")
        return pd.DataFrame({"x": [1]})

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    df = api_reader_obj.extract("parquet", as_dataframe=True, return_format="parquet")

    assert calls["called"] is True
    assert calls["buf_is_filelike"] is True
    assert isinstance(df, pd.DataFrame)
    assert df.iloc[0, 0] == 1


def test_api_csv_raw_and_dataframe_types(api_reader_obj, patched_session_get):
    csv = "a,b\n1,2\n"
    patched_session_get.return_value = make_mock_response(csv, "text/csv")

    # raw
    text = api_reader_obj.extract("csv", as_dataframe=False, return_format="csv")
    assert isinstance(text, str)

    # dataframe
    df = api_reader_obj.extract("csv", as_dataframe=True, return_format="csv")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b"]


def test_api_csv_ignores_embedded_dict(api_reader_obj, patched_session_get):
    csv = "a,b\n1,2\n"
    patched_session_get.return_value = make_mock_response(csv, "text/csv")

    # Should not attempt response["whatever"] on CSV payloads
    df = api_reader_obj.extract("csv", as_dataframe=True, return_format="csv", embedded_dict="whatever")

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (1, 2)


def test_api_retries_on_429_csv_dataframe(api_reader_obj, patched_session_get):
    csv_success = "a,b\n1,2\n"
    resp_429 = make_mock_response("rate-limit", "text/plain", status_code=429)
    success = make_mock_response(csv_success, "text/csv")
    patched_session_get.side_effect = [resp_429, success]

    df = api_reader_obj.extract(
        "csv",
        as_dataframe=True,
        retry_delay=0.01,
        max_retries=1,
        return_format="csv",
    )

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 2)


# ---------- tests: ExtractorFactory ----------

def test_factory_creates_file_extractor(tmp_files: dict[str, Path | None]) -> None:
    filepath = tmp_files["csv"]
    extractor = ExtractorFactory.create_extractor(kind="file", source=str(filepath))
    assert isinstance(extractor, FileExtractor)
    data = extractor.extract()
    assert isinstance(data, pd.DataFrame)


def test_factory_creates_api_extractor() -> None:
    extractor = ExtractorFactory.create_extractor(kind="api", source="https://example.com", headers={"x-api-key": "abc"})
    assert isinstance(extractor, ApiExtractor)


def test_factory_raises_on_invalid_kind() -> None:
    with pytest.raises(ValueError, match="Unsupported extractor kind:"):
        _ = ExtractorFactory.create_extractor(kind="ftp", source="something") # noqa
