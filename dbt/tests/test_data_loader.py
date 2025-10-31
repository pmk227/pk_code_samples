import json
import pickle
from pathlib import Path

import pandas as pd
import pytest

from src.io.data_loader import FileLoader, LoaderFactory  # adjust the import as needed


@pytest.fixture
def temp_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("data_loader_tests")


@pytest.fixture
def sample_df():
    return pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})


@pytest.fixture
def sample_dict():
    return {"name": "ChatGPT", "version": 4}


@pytest.fixture
def sample_obj():
    return {"x": [1, 2, 3], "meta": {"y": 42}}


# ---------- Successful cases ----------

def test_save_dataframe_as_csv(sample_df, temp_dir):
    path = temp_dir / "test.csv"
    loader = FileLoader(path)
    loader.load(sample_df, index=False)

    loaded = pd.read_csv(path)
    pd.testing.assert_frame_equal(sample_df, loaded)


def test_save_dataframe_as_parquet(sample_df, temp_dir):
    path = temp_dir / "test.parquet"
    loader = FileLoader(path)
    loader.load(sample_df)

    loaded = pd.read_parquet(path)
    pd.testing.assert_frame_equal(sample_df, loaded)


def test_save_dataframe_as_json(sample_df, temp_dir):
    path = temp_dir / "test.json"
    loader = FileLoader(path)
    loader.load(sample_df, orient="records")

    loaded = pd.read_json(path, orient="records")
    pd.testing.assert_frame_equal(sample_df, loaded)


def test_save_dataframe_as_excel(sample_df, temp_dir):
    path = temp_dir / "test.xlsx"
    loader = FileLoader(path)
    loader.load(sample_df, index=False)

    loaded = pd.read_excel(path)
    pd.testing.assert_frame_equal(sample_df, loaded)


def test_save_dataframe_as_feather(sample_df, temp_dir):
    path = temp_dir / "test.feather"
    loader = FileLoader(path)
    loader.load(sample_df)

    loaded = pd.read_feather(path)
    pd.testing.assert_frame_equal(sample_df, loaded)


def test_save_dict_as_json(sample_dict, temp_dir):
    path = temp_dir / "test.json"
    loader = FileLoader(path)
    loader.load(sample_dict)

    with path.open("r", encoding="utf-8") as f:
        result = json.load(f)

    assert result == sample_dict


def test_save_obj_as_pickle(sample_obj, temp_dir):
    path = temp_dir / "test.pkl"
    loader = FileLoader(path)
    loader.load(sample_obj)

    with path.open("rb") as f:
        result = pickle.load(f)

    assert result == sample_obj


# ---------- Error cases ----------

def test_raise_on_unsupported_dict_format(sample_dict, temp_dir):
    path = temp_dir / "test.csv"
    loader = FileLoader(path)

    with pytest.raises(NotImplementedError):
        loader.load(sample_dict)


def test_raise_on_unsupported_type(temp_dir):
    path = temp_dir / "test.csv"
    loader = FileLoader(path)

    class Unsupported:
        pass

    with pytest.raises(TypeError):
        loader.load(Unsupported())


def test_raise_on_unsupported_df_extension(sample_df, temp_dir):
    path = temp_dir / "test.unsupported"
    loader = FileLoader(path)

    with pytest.raises(NotImplementedError):
        loader.load(sample_df)

def test_loader_factory_returns_file_loader(tmp_path: Path):
    target = tmp_path / "data.csv"
    loader = LoaderFactory.create_loader(target, kind="file")
    assert isinstance(loader, FileLoader)
    assert loader.filepath == target.resolve()

def test_file_loader_saves_dataframe_csv(tmp_path: Path):
    target = tmp_path / "frame.csv"
    df = pd.DataFrame({"alpha": [1, 2], "beta": ["x", "y"]})

    loader = FileLoader(target)
    loader.load(df, index=False)

    loaded = pd.read_csv(target)
    pd.testing.assert_frame_equal(loaded, df)

def test_file_loader_saves_dataframe_json(tmp_path: Path):
    target = tmp_path / "frame.json"
    df = pd.DataFrame([{"a": 1, "b": 2}, {"a": 3, "b": 4}])

    loader = FileLoader(target)
    loader.load(df, orient="records")

    with target.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    assert isinstance(payload, list)
    assert payload == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]

def test_file_loader_saves_dict_json(tmp_path: Path):
    target = tmp_path / "payload.json"
    data = {"k1": "v1", "k2": 2}

    loader = FileLoader(target)
    loader.load(data, indent=2, sort_keys=True)

    with target.open("r", encoding="utf-8") as file_handle:
        roundtrip = json.load(file_handle)
    assert roundtrip == data

def test_file_loader_pickle_roundtrip(tmp_path: Path):
    target = tmp_path / "obj.pkl"
    data = {"numbers": [1, 2, 3], "flag": True}

    loader = FileLoader(target)
    loader.load(data)

    with target.open("rb") as file_handle:
        roundtrip = pickle.load(file_handle)
    assert roundtrip == data

def test_file_loader_rejects_dict_for_non_json(tmp_path: Path):
    target = tmp_path / "not_json.parquet"
    loader = FileLoader(target)
    with pytest.raises(NotImplementedError):
        loader.load({"a": 1})

def test_file_loader_dataframe_unsupported_extension(tmp_path: Path):
    target = tmp_path / "frame.unsupported"
    df = pd.DataFrame({"x": [1]})
    loader = FileLoader(target)
    with pytest.raises(NotImplementedError):
        loader.load(df)
