"""Explicit, local-only output for the repository's dataset creation recipes."""
from importlib import import_module
from pathlib import Path
from .recipes import RECIPES


def prepare_dataset(recipe, output, seed=42):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset output: {output}")
    builder = import_module(f"rlcr.data.recipes.{RECIPES[recipe]}")
    dataset = builder.build_dataset(seed=seed)
    dataset.save_to_disk(str(output))
    return dataset
