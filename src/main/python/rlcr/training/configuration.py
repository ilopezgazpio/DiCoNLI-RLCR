"""Strict training YAML loading with the existing Transformers/TRL CLI semantics."""
from dataclasses import fields
import os
from pathlib import Path

from trl import TrlParser
import yaml

from rlcr.arguments.grpo_config import GRPOConfig
from rlcr.arguments.model_config import ModelConfig
from rlcr.arguments.script_arguments import GRPOScriptArguments


def load_training_config(path, overrides=()):
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or any(not isinstance(key, str) for key in config):
        raise ValueError("Training YAML must be a mapping with string keys.")

    argument_types = (GRPOScriptArguments, GRPOConfig, ModelConfig)
    supported = {field.name for cls in argument_types for field in fields(cls) if field.init}
    unknown = config.keys() - supported - {"env"}
    if unknown:
        raise ValueError(
            f"Unknown training settings: {', '.join(sorted(unknown))}. "
            "Remove unsupported keys; process counts belong in the Accelerate launcher."
        )

    # Preserve TRL's optional env section, but validate the YAML before applying it.
    environment = config.pop("env", {})
    if not isinstance(environment, dict) or any(not isinstance(key, str) for key in environment):
        raise ValueError("env must be a mapping with string keys.")
    for key, value in environment.items():
        os.environ[key] = str(value)

    parser = TrlParser(argument_types)
    remaining = parser.set_defaults_with_config(**config)
    if remaining:
        raise ValueError(f"Unsupported training settings: {' '.join(remaining)}")
    # Explicit args avoid loading unrelated *.args files next to an entry script.
    return parser.parse_args_into_dataclasses(args=list(overrides), look_for_args_file=False)
