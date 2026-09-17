"""Load and prepare the dataset splits consumed by a training experiment."""
from pathlib import Path
from datasets import load_dataset, load_from_disk
from rlcr.data.processing import process_dataset


def load_training_datasets(script_args, training_args):
    if Path(script_args.dataset_name).is_dir():
        dataset = load_from_disk(script_args.dataset_name)
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    dataset = process_dataset(dataset, script_args)
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
    train = dataset[script_args.dataset_train_split]
    evaluation = (
        dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None
    )
    if script_args.train_subset_size is not None:
        train = train.select(range(script_args.train_subset_size))
    if evaluation is not None and script_args.eval_subset_size is not None:
        evaluation = evaluation.select(range(script_args.eval_subset_size))
    return train, evaluation
