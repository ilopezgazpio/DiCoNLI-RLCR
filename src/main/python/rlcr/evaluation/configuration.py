"""Load named evaluation YAML sections without importing ML libraries."""
from dataclasses import asdict
from pathlib import Path

import yaml

from rlcr.arguments.evaluation_global_args import GlobalArgs
from rlcr.arguments.evaluation_model_config import LocalConfig


def load_evaluation_config(path, **overrides):
    with Path(path).open() as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("Evaluation YAML must have dataset, models, and output_dir sections.")
    _check_keys(data, {"dataset", "models", "output_dir", "fresh"}, "evaluation")
    dataset = data.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be a mapping with a name (Hub ID or local directory).")
    _check_keys(dataset, {"name", "config", "split", "id_column", "sample_size"}, "dataset")
    settings = {
        "dataset_name": dataset.get("name"),
        "dataset_config": dataset.get("config"),
        **{key: value for key, value in dataset.items() if key not in {"name", "config"}},
        "output_dir": data.get("output_dir"),
        "fresh": data.get("fresh", False),
    }
    _check_keys(
        overrides,
        {
            "dataset_name",
            "dataset_config",
            "split",
            "id_column",
            "sample_size",
            "output_dir",
            "fresh",
            "model",
        },
        "overrides",
    )
    model_override = overrides.pop("model", None)
    settings.update({key: value for key, value in overrides.items() if value is not None})
    for key in ("dataset_name", "output_dir"):
        if not isinstance(settings[key], str) or not settings[key].strip():
            raise ValueError(f"{key} must be a nonempty string.")
    sample_size = settings.get("sample_size")
    if sample_size is not None and (type(sample_size) is not int or sample_size < 1):
        raise ValueError("dataset.sample_size must be a positive integer.")
    if type(settings["fresh"]) is not bool:
        raise ValueError("fresh must be a boolean.")
    models = data.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a nonempty list.")
    configs = []
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("Each models entry must be a mapping.")
        if model_override is not None:
            if len(models) != 1:
                raise ValueError("--model is only supported for single-model evaluation recipes.")
            model = {**model, "model": model_override}
        for key in ("name", "model"):
            if not isinstance(model.get(key), str) or not model[key].strip():
                raise ValueError(f"Each models entry needs a nonempty {key}.")
        try:
            configs.append(LocalConfig(**model))
        except TypeError as error:
            raise ValueError(f"Invalid evaluation model configuration: {error}") from error
    if len({config.name for config in configs}) != len(configs):
        raise ValueError("Evaluation model names must be unique (they identify result columns).")
    return GlobalArgs(**settings), configs


def evaluation_config_dict(args, models):
    """Return the same YAML schema with defaults and overrides resolved."""
    settings = asdict(args)
    output_dir = settings.pop("output_dir")
    fresh = settings.pop("fresh")
    settings["name"] = settings.pop("dataset_name")
    settings["config"] = settings.pop("dataset_config")
    return {
        "dataset": settings,
        "models": [asdict(model) for model in models],
        "output_dir": output_dir,
        "fresh": fresh,
    }


def _check_keys(mapping, allowed, section):
    unknown = mapping.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown {section} settings: {', '.join(sorted(unknown))}")
