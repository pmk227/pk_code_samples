from abc import ABC, abstractmethod
from typing import Any, overload, Literal
from pathlib import Path
import pickle
import json

import pandas as pd


class BaseLoader(ABC):
    @abstractmethod
    def load(self, data: Any, **kwargs: Any) -> None:
        """Load data to a target destination."""
        pass


class FileLoader(BaseLoader):
    """
    Loader for saving data to local files in various formats.

    Supports:
        - pd.DataFrame: .csv, .parquet, .json, .xlsx, .feather, .arrow, .pkl
        - dict: .json, .pkl
        - any pickle-able object: .pkl
    """

    def __init__(self, filepath: str | Path) -> None:
        self.filepath: Path = Path(filepath).expanduser().resolve()
        self.extension = self.filepath.suffix.lower()

    def load(self, data, **kwargs: Any) -> None:
        """
        Save data to disk using appropriate method.

        :param data: DataFrame, dict, or any pickle-able object
        :param kwargs: Extra args passed to the relevant writer
        """

        if self.extension == ".pkl":
            self._save_obj_as_pkl(data)
        elif isinstance(data, pd.DataFrame):
            self._save_dataframe(data, **kwargs)
        elif isinstance(data, dict):
            if self.extension == ".json":
                self._save_dict_as_json(data, **kwargs)
            else:
                raise NotImplementedError(
                    f"Dict objects can only be saved as .json or .pkl, not {self.extension}"
                )
        else:
            raise TypeError(
                f"Unsupported combination: type={type(data)}, extension={self.extension}"
            )

    def _save_dataframe(self, df: pd.DataFrame, **kwargs: Any) -> None:
        match self.extension:
            case ".csv":
                df.to_csv(self.filepath, **kwargs)
            case ".parquet" | ".pqt":
                df.to_parquet(self.filepath, **kwargs)
            case ".json":
                df.to_json(self.filepath, **kwargs)
            case ".xlsx" | ".xls":
                df.to_excel(self.filepath, **kwargs)
            case ".feather" | ".arrow":
                df.to_feather(self.filepath, **kwargs)
            case _:
                raise NotImplementedError(
                    f"Unsupported file extension for DataFrame: {self.extension}. Use FileLoader.supported_extensions()"
                    f" to list the supported file extensions"
                )


    def _save_obj_as_pkl(self, data: Any) -> None:
        with self.filepath.open("wb") as f:
            pickle.dump(data, f)

    def _save_dict_as_json(self, data: dict, **kwargs: Any) -> None:
        with self.filepath.open("w", encoding=kwargs.pop("encoding", "utf-8")) as f:
            json.dump(data, f, **kwargs)

    @staticmethod
    def supported_extensions() -> None:
        print(".csv/.parquet/.json/.xlsx/.xls/.feather/.arrow/.pkl")


class LoaderFactory:
    @staticmethod
    @overload
    def create_loader(target: str | Path, kind: Literal["file"], **kwargs: Any) -> FileLoader: ...

    @staticmethod
    def create_loader(target: str | Path, kind: str, **kwargs: Any) -> BaseLoader:
        """
        Create an appropriate loader based on kind.

        :param target: file path or other destination identifier
        :param kind: 'file' (future kinds like 'db', 's3' can be added)
        :param kwargs: additional arguments for the loader
        """
        match kind.lower():
            case "file":
                return FileLoader(target)
            case _:
                raise ValueError(f"Unsupported loader kind: {kind}")
