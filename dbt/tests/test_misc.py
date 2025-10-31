# tests/test_secrets_parser.py
import json

from src.utilities.secrets_parser import SecretsParser
from src.utilities.utils import df_upsert


def _write_secrets(tmp_path, payload: dict) -> str:
    """Helper to write a secrets file and return its path."""
    p = tmp_path / "secrets.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(p)


def test_get_credentials_returns_full_mapping(tmp_path, monkeypatch):
    secrets_payload = {
        "credentials": {
            "alpha": {"API_Key": "alpha-key", "extra": "value"},
            "beta": {"API_Key": "beta-key"},
            "db_conn": {"type": "database_connection", "API_Key": "db-key", "host": "localhost"},
        }
    }
    secrets_path = _write_secrets(tmp_path, secrets_payload)
    monkeypatch.setenv("secrets_path", secrets_path)

    sp = SecretsParser()
    creds = sp.get_credentials("alpha")

    assert isinstance(creds, dict)
    assert creds["API_Key"] == "alpha-key"
    assert creds


import pandas as pd
import pytest


def _df(cols, rows):
    return pd.DataFrame(rows, columns=cols)


def test_upsert_overwrite_true_updates_and_inserts():
    base = _df(
        ["Code", "Listed", "guid"],
        [
            ["AAPL", 1, "g1"],
            ["MSFT", 0, "g2"],
        ],
    )
    incoming = _df(
        ["Code", "Listed"],
        [
            ["AAPL", 0],   # should overwrite (overwrite=True)
            ["GOOG", 1],   # new row
        ],
    )

    result = df_upsert(base, incoming, keys="Code", overwrite=True, add_incoming_cols=False)

    assert set(result["Code"]) == {"AAPL", "MSFT", "GOOG"}
    # AAPL updated
    assert result.loc[result["Code"] == "AAPL", "Listed"].item() == 0
    # MSFT unchanged
    assert result.loc[result["Code"] == "MSFT", "Listed"].item() == 0
    # GOOG inserted
    assert result.loc[result["Code"] == "GOOG", "Listed"].item() == 1
    # guid preserved for existing rows
    assert result.loc[result["Code"] == "AAPL", "guid"].item() == "g1"
    assert result.loc[result["Code"] == "MSFT", "guid"].item() == "g2"


def test_upsert_overwrite_false_fills_only_na():
    base = _df(
        ["Code", "Listed"],
        [
            ["AAPL", pd.NA],  # NA -> should be filled
            ["MSFT", 0],      # not NA -> should NOT be overwritten
        ],
    )
    incoming = _df(
        ["Code", "Listed"],
        [
            ["AAPL", 1],
            ["MSFT", 1],
        ],
    )

    result = df_upsert(base, incoming, keys="Code", overwrite=False)

    # AAPL filled from NA -> 1
    assert int(result.loc[result["Code"] == "AAPL", "Listed"].item()) == 1
    # MSFT unchanged (0) despite incoming 1
    assert int(result.loc[result["Code"] == "MSFT", "Listed"].item()) == 0


def test_upsert_add_incoming_cols_true_adds_and_updates_new_columns():
    base = _df(
        ["Code", "Listed"],
        [
            ["AAPL", 1],
        ],
    )
    incoming = _df(
        ["Code", "Listed", "Sector"],
        [
            ["AAPL", 0, "Tech"],  # update Listed + add Sector
            ["TSLA", 1, "Auto"],  # insert with Sector
        ],
    )

    result = df_upsert(base, incoming, keys="Code", overwrite=True, add_incoming_cols=True)

    # Sector column added
    assert "Sector" in result.columns
    # AAPL updated
    row_aapl = result[result["Code"] == "AAPL"].iloc[0]
    assert int(row_aapl["Listed"]) == 0
    assert row_aapl["Sector"] == "Tech"
    # TSLA inserted
    row_tsla = result[result["Code"] == "TSLA"].iloc[0]
    assert int(row_tsla["Listed"]) == 1
    assert row_tsla["Sector"] == "Auto"


def test_upsert_new_rows_when_add_incoming_cols_false_does_not_add_column_to_existing():
    base = _df(
        ["Code", "Listed"],
        [
            ["AAPL", 1],
        ],
    )
    incoming = _df(
        ["Code", "Listed", "Sector"],
        [
            ["AAPL", 0, "Tech"],  # update Listed but Sector should NOT be added to base row
            ["TSLA", 1, "Auto"],  # inserted row will carry Sector from incoming
        ],
    )

    result = df_upsert(base, incoming, keys="Code", overwrite=True, add_incoming_cols=False)

    # AAPL row: no Sector column added to existing row
    assert "Sector" in result.columns  # present due to inserted row
    # For AAPL, Sector should be NA (since base row didn't have it and we didn't add_incoming_cols)
    aapl_sector = result.loc[result["Code"] == "AAPL", "Sector"].iloc[0]
    assert pd.isna(aapl_sector)
    # TSLA inserted with Sector
    tsla_sector = result.loc[result["Code"] == "TSLA", "Sector"].iloc[0]
    assert tsla_sector == "Auto"


def test_upsert_multi_key_support():
    base = _df(
        ["Code", "Exchange", "Listed"],
        [
            ["AAPL", "US", 1],
            ["AAPL", "DE", 0],
        ],
    )
    incoming = _df(
        ["Code", "Exchange", "Listed"],
        [
            ["AAPL", "US", 0],  # update only US row
            ["MSFT", "US", 1],  # insert new
        ],
    )

    result = df_upsert(base, incoming, keys=["Code", "Exchange"], overwrite=True)

    # AAPL-US updated, AAPL-DE unchanged, MSFT-US inserted
    assert int(result[(result.Code == "AAPL") & (result.Exchange == "US")]["Listed"].item()) == 0
    assert int(result[(result.Code == "AAPL") & (result.Exchange == "DE")]["Listed"].item()) == 0
    assert int(result[(result.Code == "MSFT") & (result.Exchange == "US")]["Listed"].item()) == 1


def test_upsert_duplicates_in_incoming_keep_last_wins():
    base = _df(
        ["Code", "Listed"],
        [
            ["AAPL", 1],
        ],
    )
    # two rows for the same key; keep="last" in function should apply 0
    incoming = _df(
        ["Code", "Listed"],
        [
            ["AAPL", 1],
            ["AAPL", 0],
        ],
    )

    result = df_upsert(base, incoming, keys="Code", overwrite=True)

    assert int(result.loc[result["Code"] == "AAPL", "Listed"].item()) == 0


def test_upsert_does_not_mutate_inputs():
    base = _df(
        ["Code", "Listed"],
        [
            ["AAPL", 1],
        ],
    )
    incoming = _df(
        ["Code", "Listed"],
        [
            ["AAPL", 0],
            ["MSFT", 1],
        ],
    )
    base_copy = base.copy(deep=True)
    incoming_copy = incoming.copy(deep=True)

    _ = df_upsert(base, incoming, keys="Code", overwrite=True)

    # original inputs unchanged
    pd.testing.assert_frame_equal(base, base_copy)
    pd.testing.assert_frame_equal(incoming, incoming_copy)


def test_upsert_with_none_base_returns_incoming_when_add_cols_true():
    base_dataframe = None  # simulate "no existing bronze file"
    incoming_dataframe = pd.DataFrame(
        [
            {"Code": "AAA", "Name": "Alpha", "Extra": 1},
            {"Code": "BBB", "Name": "Beta", "Extra": 2},
        ]
    )

    result = df_upsert(
        base_dataframe,
        incoming_dataframe,
        keys="Code",
        add_incoming_cols=True,  # critical for first-insert case
    )

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        incoming_dataframe.reset_index(drop=True),
    )
