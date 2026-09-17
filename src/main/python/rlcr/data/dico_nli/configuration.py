"""Strict data-only recipe; paths follow the application's working directory."""
from pathlib import Path
import re

import yaml

SOURCE_REPOSITORY = "https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI"


def _mapping(value, allowed, required, context):
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a mapping with string keys.")
    missing, extra = set(required) - value.keys(), value.keys() - set(allowed)
    if missing or extra:
        raise ValueError(
            f"{context}: missing settings {sorted(missing)}; unknown settings {sorted(extra)}."
        )


def _file_spec(spec, context):
    _mapping(spec, {"path", "sha256"}, {"path", "sha256"}, context)
    if not isinstance(spec["path"], str) or not spec["path"].strip():
        raise ValueError(f"{context}.path must be nonempty.")
    if not isinstance(spec["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", spec["sha256"]):
        raise ValueError(f"{context}.sha256 must be a lowercase SHA-256 digest.")


def load_preparation_config(path, *, output_dir=None):
    try:
        text = Path(path).read_text(encoding="utf-8")
        # Reject duplicate keys as well as unknown keys; never accept last-value-wins.
        node = yaml.compose(text, Loader=yaml.SafeLoader)
        _check_duplicate_keys(node)
        config = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid data YAML: {error}") from error
    _mapping(
        config,
        {"source", "output_dir", "seed", "splits"},
        {"source", "output_dir", "splits"},
        "data",
    )
    source = config["source"]
    _mapping(source, {"repository", "revision"}, {"repository", "revision"}, "source")
    if source["repository"] != SOURCE_REPOSITORY:
        raise ValueError("source.repository must identify the official DiCo-NLI repository.")
    if not isinstance(source["revision"], str) or not re.fullmatch(
        "[0-9a-f]{40}", source["revision"]
    ):
        raise ValueError("source.revision must be a full 40-character commit SHA, not a branch.")
    if output_dir is not None:
        config["output_dir"] = str(output_dir)
    if not isinstance(config["output_dir"], str) or not config["output_dir"].strip():
        raise ValueError("output_dir must be nonempty.")
    config.setdefault("seed", 42)
    if type(config["seed"]) is not int:
        raise ValueError("seed must be an integer.")
    splits = config["splits"]
    if not isinstance(splits, dict) or not splits:
        raise ValueError("splits must be a nonempty mapping.")
    for name, spec in splits.items():
        if not isinstance(name, str) or not re.fullmatch("[a-z][a-z0-9_]*", name):
            raise ValueError(f"Invalid split name: {name!r}.")
        _mapping(
            spec,
            {"participant", "reference", "labeled", "max_source_pairs"},
            {"participant", "labeled"},
            f"split {name}",
        )
        if type(spec["labeled"]) is not bool:
            raise ValueError(f"{name}.labeled must be a boolean.")
        _file_spec(spec["participant"], f"{name}.participant")
        if "reference" in spec:
            if not spec["labeled"]:
                raise ValueError("Unlabeled inference inputs cannot have a reference.")
            _file_spec(spec["reference"], f"{name}.reference")
        maximum = spec.get("max_source_pairs")
        if maximum is not None and (type(maximum) is not int or maximum < 1):
            raise ValueError(f"{name}.max_source_pairs must be a positive integer.")
    return config


def _check_duplicate_keys(node, seen=None):
    seen = set() if seen is None else seen
    if id(node) in seen:
        raise ValueError("YAML aliases are unsupported in data recipes.")
    seen.add(id(node))
    if isinstance(node, yaml.MappingNode):
        keys = []
        for key, value in node.value:
            if not isinstance(key, yaml.ScalarNode) or key.tag != "tag:yaml.org,2002:str":
                raise ValueError(
                    "Data YAML mappings require string keys; aliases/merges are unsupported."
                )
            if key.value in keys:
                raise ValueError(f"Duplicate YAML key: {key.value}")
            keys.append(key.value)
            _check_duplicate_keys(value, seen)
    elif isinstance(node, yaml.SequenceNode):
        for value in node.value:
            _check_duplicate_keys(value, seen)
