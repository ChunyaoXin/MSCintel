"""Utilities for loading YAML configuration files."""

from pathlib import Path

import yaml


def load_config(config_path=None, repo_root=None):
    """Load a YAML configuration with nested sections."""

    if config_path is None:
        if repo_root is None:
            raise ValueError(
                "Either config_path or repo_root must be provided."
            )

        config_path = Path(repo_root) / "config.yaml"

    config_path = Path(config_path).expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid or empty YAML configuration: {config_path}"
        )

    # Resolve project root relative to the config file.
    root_value = Path(config.get("repo_root", "."))

    if repo_root is not None:
        project_root = Path(repo_root).expanduser().resolve()
    elif root_value.is_absolute():
        project_root = root_value.resolve()
    else:
        project_root = (config_path.parent / root_value).resolve()

    config["repo_root"] = str(project_root)

    return config