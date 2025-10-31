from pathlib import Path
from typing import Callable
from abc import ABC, abstractmethod
import json
import pandas as pd
import requests
import time
from typing import Any, Dict, Optional, Union, overload, Literal
from threading import Event
from io import StringIO, BytesIO
from etls.eodhd.eodhd_utils.eodhd_constants import StatusCodes
import logging

ReturnType = Union[pd.DataFrame, dict[str, Any]]
logger = logging.getLogger(__name__)

class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, **kwargs: Any) -> ReturnType:
        """Fetch data from a source and return it."""
        pass


class FileExtractor(BaseExtractor):
    """
    Read-only extractor for CSV, Parquet, and JSON.

    API:
        fe = FileExtractor(path)
        data = fe.pull_data(**kwargs)  # kwargs forwarded to pandas/json reader

    Extending:
        - Add a new _read_<ext>() method
        - Add a new `case` in _get_reader()
    """

    def __init__(self, filepath: str | Path, ) -> None:
        self.filepath: Path = Path(filepath).expanduser().resolve()
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        self.reader: Callable[..., ReturnType] = self._get_reader()

    @staticmethod
    def supported_extensions() -> None:
        print(".csv/.parquet/.json/.xls/.xlsx/.h5/.hdf5/.feather/.arrow/.pkl")

    # ---------- factory ----------
    def _get_reader(self) -> Callable[..., ReturnType]:
        ext = self.filepath.suffix.lower()

        match ext:
            case ".csv":
                return pd.read_csv
            case ".parquet":
                return pd.read_parquet
            case ".json":
                return self._read_json
            case ".xls" | ".xlsx":
                return pd.read_excel
            case ".h5" | ".hdf5":
                return pd.read_hdf
            case ".feather" | ".arrow":
                # `.arrow` often used for Feather/Arrow IPC; assumes pyarrow-backed feather.
                return pd.read_feather
            case ".pkl":
                return pd.read_pickle
            case _:
                raise NotImplementedError(
                    f"Unsupported file extension: {ext}. Use FileExtractor.supported_extensions() to list the supported file extensions"
                )

    # ---------- implementations ----------
    def _read_json(self, _, **kwargs: Any) -> dict[str, Any] | pd.DataFrame:
        """
        :param kwargs:
            - encoding: str, default = 'utf-8'
            - as_dataframe: bool, default = False — return pandas DataFrame if True
            - flatten: bool, default = True — flattens nested JSON; only used if as_dataframe is True
            - embedded_dict: str | None, default = None — extract this key from top-level JSON before processing
        """

        # Default to utf-8, allow override


        encoding = kwargs.get("encoding", "utf-8")
        try:
            with self.filepath.open("r", encoding=encoding) as f:
                payload = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"JSON decode error in '{self.filepath}': {e.msg}",
                e.doc,
                e.pos,
            ) from e

        as_dataframe = kwargs.get("as_dataframe", False)
        flatten = kwargs.get("flatten", True)
        embedded_dict: str | None = kwargs.get("embedded_dict", None)

        if embedded_dict is not None:
            payload = payload[embedded_dict]

        if as_dataframe:
            if flatten:
                return pd.json_normalize(payload)
            else:
                return pd.DataFrame(payload)

        return payload

    # ---------- public API ----------
    def extract(self, **kwargs: Any) -> ReturnType:
        """Forward kwargs directly to the selected reader."""
        return self.reader(self.filepath, **kwargs)


class ApiExtractor(BaseExtractor):
    """
    Abstract base class for API readers.
    """

    def __init__(self,
                 base_url: str,
                 headers: Optional[Dict[str, str]] = None,
                 timeout: Optional[float] = 60,
                 stop_event: Optional[Event] = None,
                 retry_delay: Optional[float] = None,
                 max_retries: Optional[int] = None,
                 ) -> None:
        """
        :param base_url: top level url of the api call
        :param headers:  headers to send into the api
        :param timeout:  timeout in seconds
        :param stop_event: Daily limit event; used to share event flag across extractor instances
        """

        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.timeout = timeout
        self.session = requests.Session()
        self.stop_event = stop_event or Event()

        self.retry_delay: Optional[float] = retry_delay
        self.max_retries: Optional[int] = max_retries

    def set_stop_event(self):
        self.stop_event.set()

    def extract(self,
                *,
                endpoint: str,
                as_dataframe: bool = True,
                flatten: bool = False,
                return_format: str | None= None,  # json or csv
                retry_delay: Optional[float] = 0,
                max_retries: Optional[int] = 0,
                embedded_dict: str | None = None,
                method: str = "GET",  # here for potential enhancements
                **kwargs: Any) -> ReturnType | None | int:
        """
        Pull data from a rest API

        :param endpoint:  API endpoint
        :param method:  HTTP method ("GET" or "POST")
        :param as_dataframe:  Whether to return a pandas DataFrame or a dictionary
        :param flatten: If True and parsing JSON, flattens nested dictionaries
        :param return_format: "json" to parse JSON, "csv" to parse CSV from response; can be auto-inferred
        :param embedded_dict: optional, the key of the embedded dict to use as the top level of the json
        :param retry_delay: retry delay in seconds between requests calls; if None will not retry
        :param max_retries: maximum number of retry attempts; if None will not retry
        :param kwargs: Other kwargs (params, any others);
            - params: dict, feeds directly into requests as .get(params=kwargs.get('params'))
        """
        if retry_delay is None and max_retries:
            retry_delay = 10
            logger.warning("Retries detected without retry_delay; setting to 10 seconds")

        max_attempts = 1 if max_retries in (None, 0) else (max_retries + 1)

        if method.upper() != "GET":
            raise NotImplementedError(...)

        if method.upper() == "GET":  # Leave this in for forward compatability
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            params: dict = kwargs.get("params", {})
        else:
            raise NotImplementedError(f"Unsupported HTTP method: {method}.  How did you even do this... ?")

        response = self._get_response_or_fail(url=url, params=params, max_attempts=max_attempts, retry_delay=retry_delay)

        if response is None:
            return None
        elif isinstance(response, int):
            return response

        return_format = return_format or self._get_return_format(response)
        response.encoding = kwargs.get("encoding", "utf-8")
        response = self._decode_response(response, return_format=return_format)
        response = self._prepare_response(response=response, as_dataframe=as_dataframe, flatten=flatten,
                                          embedded_dict=embedded_dict, return_format=return_format)
        return response

    @staticmethod
    def _prepare_response(
            response: Union[dict[str, Any], list, str, bytes],
            *,
            as_dataframe: bool,
            flatten: bool,
            embedded_dict: str | None,
            return_format: str,
    ) -> ReturnType:
        # Apply embedded_dict only for mapping-like payloads (JSON/XML decoded)
        if embedded_dict is not None and isinstance(response, dict):
            response = response[embedded_dict]

        if return_format in {"json", "xml"}:
            if as_dataframe:
                return pd.json_normalize(response) if flatten else pd.DataFrame(response)
            return response  # dict/list

        if return_format == "csv":
            # response is a str from _decode_response
            return pd.read_csv(StringIO(response)) if as_dataframe else response

        if return_format == "parquet":
            # response is bytes from _decode_response
            return pd.read_parquet(BytesIO(response)) if as_dataframe else response

        raise NotImplementedError(f"Unsupported return format in prepare: {return_format}")

    @staticmethod
    def _decode_response(response: requests.Response, return_format: str) -> Union[dict[str, Any], str, bytes]:
        match return_format:
            case "json":
                return response.json()
            case "csv":
                return response.text
            case "xml":
                try:
                    import xmltodict
                except ImportError as e:
                    raise ImportError(
                        "xmltodict is required to parse XML responses. Install it via `pip install xmltodict`.") from e
                return xmltodict.parse(response.text)
            case "parquet":
                return response.content
            case _:
                raise NotImplementedError(f"Unsupported return format: {return_format}")

    @staticmethod
    def _get_return_format(response) -> str:
        content_type = response.headers.get("Content-Type", "").lower()
        if "json" in content_type:
            return "json"
        elif "csv" in content_type:
            return "csv"
        elif "xml" in content_type:
            return "xml"
        else:
            raise ValueError("Could not infer response format from Content-Type header. Please specify return_format.")

    def _get_response_or_fail(self,
                              url: str,
                              params: dict | None,
                              max_attempts: int,
                              retry_delay: float
                              ) -> Union[requests.Response, int]:

        for retry in range(max_attempts):
            if self.stop_event.is_set():
                return 402  # Daily limit hit

            response = self.session.get(url, headers=self.headers, params=params, timeout=self.timeout)

            if response.ok:
                return response

            if response.status_code in StatusCodes.daily_limit_status_codes:
                logger.info(f"Failed to retrieve response from {url}, status code: {response.status_code}. \nShutting down ETL.")
                self.set_stop_event()
                return response.status_code

            elif response.status_code in StatusCodes.malformed_request_status_codes:
                logger.info(f"Malformed request from {url}, status code: {response.status_code}. Skipping symbol.")
                return response.status_code

            elif response.status_code in StatusCodes.retryable_status_codes:
                if retry <= max_attempts:
                    logger.debug(f"Error code: {response.status_code}. Retrying in {retry_delay} seconds.")
                    time.sleep(retry_delay)
                    continue
                else:
                    return response.status_code

            elif response.status_code in StatusCodes.unauthorized_status_codes:
                logger.warning(f"Error code {response.status_code}. Check API key or token. \nShutting down ETL.")
                self.set_stop_event()
                return response.status_code

            else:
                response.raise_for_status()
        return None


class ExtractorFactory:
    @staticmethod
    @overload
    def create_extractor(source: str, kind: Literal["file"], **kwargs: Any) -> FileExtractor:
        ...

    @staticmethod
    @overload
    def create_extractor(source: str, kind: Literal["api"], **kwargs: Any) -> ApiExtractor:
        ...

    @staticmethod
    def create_extractor(kind: str,
                         source: str,
                         **kwargs: Any) -> BaseExtractor:
        """
        Create an appropriate extractor based on kind.

        :param kind: 'file' or 'api'
        :param source: file path or API base URL
        :param kwargs: additional arguments for the extractor
        """
        match kind.lower():
            case "file":
                return FileExtractor(source)
            case "api":
                return ApiExtractor(base_url=source, **kwargs)
            case _:
                raise ValueError(f"Unsupported extractor kind: {kind}")
