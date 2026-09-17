"""Evaluation dataset loading, incremental columns, and result persistence."""
import json
from pathlib import Path
from datasets import DatasetDict, load_dataset, load_from_disk


def load_evaluation_dataset(args):
    if Path(args.dataset_name).is_dir():
        dataset = load_from_disk(args.dataset_name)
    else:
        dataset = load_dataset(args.dataset_name, name=args.dataset_config)
    if isinstance(dataset, DatasetDict):
        dataset = dataset[args.split]
    if args.id_column not in dataset.column_names:
        raise ValueError(f"Prepared evaluation data needs an {args.id_column} column.")
    identifiers = list(dataset[args.id_column])
    if any(not isinstance(value, str) or not value.strip() for value in identifiers):
        raise ValueError("Evaluation identifiers must be nonempty strings.")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Evaluation identifiers must be unique.")
    if args.sample_size is not None:
        dataset = dataset.select(range(args.sample_size))
    return dataset


def load_existing_results(args):
    output = Path(args.output_dir) / "predictions"
    if output.is_dir():
        # Existing Arrow files must not remain memory-mapped when saved back here.
        dataset = load_from_disk(str(output), keep_in_memory=True)
        return dataset[args.split] if isinstance(dataset, DatasetDict) else dataset
    return None


def load_existing_metrics(args):
    path = Path(args.output_dir) / "metrics.json"
    if path.is_file():
        with path.open() as stream:
            return json.load(stream)
    return {}


def add_or_replace_columns(dataset, columns):
    for name, values in columns.items():
        if name in dataset.column_names:
            dataset = dataset.remove_columns(name)
        dataset = dataset.add_column(name, values)
    return dataset


def save_results(dataset, metrics, args, updated):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if updated:
        dataset.save_to_disk(str(output / "predictions"))
    with (output / "metrics.json").open("w") as stream:
        json.dump(metrics, stream, indent=4)
