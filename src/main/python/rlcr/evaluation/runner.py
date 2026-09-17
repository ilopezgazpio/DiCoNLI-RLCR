"""Persist raw batch predictions; export and task scoring are separate commands."""
from copy import deepcopy
from rlcr.configuration.snapshots import save_resolved_config
from rlcr.data.validation import validate_prompt_dataset
from .configuration import evaluation_config_dict
from .generation import generate_columns
from .storage import (
    add_or_replace_columns,
    load_evaluation_dataset,
    load_existing_results,
    load_existing_metrics,
    save_results,
)


def run_evaluation(global_args, local_configs):
    dataset = load_evaluation_dataset(global_args)
    for config in local_configs:
        validate_prompt_dataset(dataset, prompt_column=config.tokenize_key)
    existing = load_existing_results(global_args)
    if existing is not None and (
        len(existing) != len(dataset)
        or any(
            column not in existing.column_names or list(existing[column]) != list(dataset[column])
            for column in dataset.column_names
        )
    ):
        raise ValueError(
            "Existing predictions belong to different examples or ordering. "
            "Choose a new output_dir for a different dataset/split/sample size."
        )
    save_resolved_config(global_args.output_dir, evaluation_config_dict(global_args, local_configs))
    final_dataset = deepcopy(existing if existing is not None else dataset)
    metrics = load_existing_metrics(global_args) if existing is not None else {}
    updated = False
    for config in local_configs:
        if (
            existing is not None
            and all(f"{config.name}-output_{i}" in existing.column_names for i in range(config.n))
            and not (config.fresh or global_args.fresh)
        ):
            print(f"Skipping {config.name} because it already exists")
            continue
        columns, metrics[config.name] = generate_columns(dataset, config)
        final_dataset = add_or_replace_columns(final_dataset, columns)
        updated = True
    save_results(final_dataset, metrics, global_args, updated)
    return metrics
