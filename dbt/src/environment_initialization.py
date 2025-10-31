# This script should be at the top of all experiments to initialize the environment

from pathlib import Path
import os
import logging
import src.utilities.logging_config as lc


def set_env_filepaths() -> None:
    """Sets up environment variables before logging is initialized."""

    # Hephaestus root path
    hephaestus_project_root = Path.home() / "PycharmProjects" / "Hephaestus"
    hephaestus_root_path = Path(os.environ.get('hephaestus_root_path', hephaestus_project_root)).resolve()
    os.environ['hephaestus_root_path'] = str(hephaestus_root_path)
    print(f"[ENV INIT] Hephaestus root filepath: {hephaestus_root_path}")

    # Prometheus_v2 root path
    prometheus_project_root = Path.home() / "PycharmProjects" / "Prometheus_v2"
    prometheus_root_path = Path(os.environ.get('prometheus_root_path', prometheus_project_root)).resolve()
    os.environ['prometheus_root_path'] = str(prometheus_root_path)
    print(f"[ENV INIT] Prometheus root filepath: {prometheus_root_path}")

    # Ensure filestore_root_path is set
    default_filestore_root = Path.home() / "PycharmProjects" / "filestore"
    filestore_root_path = Path(os.environ.get('filestore_root_path', default_filestore_root)).resolve()
    os.environ['filestore_root_path'] = str(filestore_root_path)
    print(f"[ENV INIT] Filestore root filepath: {filestore_root_path}")

    # Set the project root path (reset for each project)
    project_root_path = prometheus_project_root
    os.environ['project_root_path'] = str(project_root_path)
    print(f"[ENV INIT] Project root filepath: {project_root_path}")

    # Define remaining environment paths
    env_paths = {
        "secrets_path": project_root_path / "resources" / "secrets.json",
        # "etl_system_settings": project_root_path / "resources" / "sys_config_files" / "etl_system_settings.json",
        "lz_path": project_root_path / "_lz",
        "bronze_path": filestore_root_path / "bronze",
        "silver_path": filestore_root_path / "silver",
        "gold_path": filestore_root_path / "gold",
        "platinum_path": filestore_root_path / "platinum",
    }

    for key, path in env_paths.items():
        if key not in os.environ:
            os.environ[key] = str(path.resolve())
            print(f"[ENV INIT] Set {key}: {os.environ[key]}")  # Visibility without logging

if "bronze_path" in os.environ:
    lc.setup_logging()
    pass
else:
    set_env_filepaths()
    lc.setup_logging()
    logger = logging.getLogger(__name__)
    lc.log_initialization_test(logger)
