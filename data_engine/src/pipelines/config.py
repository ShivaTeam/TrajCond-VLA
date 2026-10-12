"""Resolve new pipeline paths after loading the complete dataset config."""

from copy import deepcopy
from pathlib import Path
import yaml

ENGINE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ENGINE_ROOT.parent


def load_yaml(path):
    with open(path, encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return data


def merge_config(base, updates):
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def prepare_config(config):
    config = deepcopy(config)
    ds = config["dataset"]
    if "splits" not in ds and "data_path" not in ds:
        path = ds.get("config_path", ENGINE_ROOT / "configs" / "datasets" / f"{ds['name']}.yaml")
        if not Path(path).is_absolute() and config.get("path_base") == "project_root":
            path = PROJECT_ROOT / path
        if Path(path).is_file():
            config["dataset"] = merge_config(load_yaml(path), ds)
    base = config.get("path_base")
    if base is None:
        return config
    if base != "project_root":
        raise ValueError("path_base must be project_root when supplied")
    path_keys = {"data_path", "sensor_blobs_path", "log_names_file", "config_path", "source_path",
                 "codebook_path", "variances_path", "stats_file", "template_path", "format_contract_path",
                 "cache_dir", "output_dir"}

    def resolve(mapping):
        for key, value in mapping.items():
            if isinstance(value, dict):
                resolve(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        resolve(item)
            elif key in path_keys and isinstance(value, str):
                mapping[key] = str((PROJECT_ROOT / value).resolve())
    resolve(config)
    config["output"]["path"] = str((PROJECT_ROOT / config["output"]["path"]).resolve())
    return config
