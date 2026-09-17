"""Orchestrate evaluation stages without owning their model or scoring logic."""
from copy import deepcopy
from rlcr.configuration.snapshots import save_resolved_config
from rlcr.data.processing import process_dataset
from .configuration import evaluation_config_dict
from .generation import generate_columns
from .storage import (
    add_or_replace_columns,
    load_evaluation_dataset,
    load_existing_results,
    load_existing_metrics,
    save_results,
)
from .verifiers.judge import llm_confidence_verifier
from .verifiers.symbolic import confidence_verifier


VERIFIERS = {
    "confidence_verifier": confidence_verifier,
    "llm_confidence_verifier": llm_confidence_verifier,
}


def run_evaluation(global_args, local_configs):
    dataset = load_evaluation_dataset(global_args)
    existing = load_existing_results(global_args)
    if existing is not None and (
        len(existing) != len(dataset) or list(existing["id"]) != list(dataset["id"])
    ):
        raise ValueError(
            "Existing predictions belong to different examples or ordering. "
            "Choose a new output_dir for a different dataset/split/sample size."
        )
    save_resolved_config(global_args.output_dir, evaluation_config_dict(global_args, local_configs))
    final_dataset = deepcopy(existing if existing is not None else dataset)
    metrics = load_existing_metrics(global_args) if existing is not None else {}
    run_metrics = {}
    updated = False
    for original in local_configs:
        config = deepcopy(original)
        config.split = global_args.split
        config.dataset_name = global_args.dataset_name
        config.fresh = config.fresh or global_args.fresh
        if (
            existing is not None
            and f"{config.name}-output_0" in existing.column_names
            and not config.fresh
        ):
            print(f"Skipping {config.name} because it already exists")
            continue
        local_dataset = process_dataset(deepcopy(dataset), config)
        columns, run_metrics[config.name] = generate_columns(local_dataset, config)
        final_dataset = add_or_replace_columns(final_dataset, columns)
        local_dataset = add_or_replace_columns(local_dataset, columns)
        updated = True
        metrics.pop(config.name, None)
        if config.check_fn is not None:
            if config.check_fn not in VERIFIERS:
                raise ValueError(f"Unknown evaluation check function: {config.check_fn}")
            labels, metrics[config.name] = VERIFIERS[config.check_fn](
                local_dataset, config, **config.check_fn_args
            )
            final_dataset = add_or_replace_columns(final_dataset, labels)

    for name, values in metrics.items():
        print(f"Metrics for {name}: {values}")
    for name, values in run_metrics.items():
        print(f"Run metrics for {name}: {values}")
    save_results(final_dataset, metrics, global_args, updated)
    return metrics
