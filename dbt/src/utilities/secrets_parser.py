import json
import os
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(f'utils.{__name__.split(".")[-1]}')


@dataclass(frozen=True, slots=True)
class SecretsParser:
    """
    Immutable, pickle-friendly secrets container.

    Reads the secrets file path from the environment variable `secrets`
    (set by src.environment_initialization). Safe to share across threads
    and processes.
    """
    secrets_path: str = field(default_factory=lambda: os.environ["secrets_path"])
    configuration: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        # Load once at construction; object is frozen after
        with open(self.secrets_path, "r", encoding="utf-8") as file_handle:
            configuration_data = json.load(file_handle)
        object.__setattr__(self, "configuration", configuration_data)
        logger.debug(
            "Secrets loaded from %s (top-level keys: %s)",
            self.secrets_path,
            list(self.configuration.keys()),
        )

    def get_credentials(self, app: str) -> dict[str, Any]:
        return self.configuration["credentials"][app]

    def get_api_key(self, app: str) -> str:
        return self.configuration["credentials"][app]["API_Key"]

    def get_database_connections(self) -> list[str]:
        db_conns: list[str] = []
        for key, value in self.configuration.get("credentials", {}).items():
            if isinstance(value, dict) and value.get("type") == "database_connection":
                db_conns.append(key)
        return db_conns
