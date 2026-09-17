"""Load prepared dataset splits without injecting task-specific prompts."""
from pathlib import Path
from datasets import DatasetDict, load_dataset, load_from_disk
from rlcr.data.validation import validate_prompt_dataset
from rlcr.data.dico_nli.labels import LABELS
from rlcr.rewards.registry import DICO_REWARDS, LABEL_REWARDS


def load_training_datasets(script_args, training_args):
    dico = bool(DICO_REWARDS & set(script_args.reward_funcs))
    if dico and (
        script_args.train_subset_size is not None or script_args.eval_subset_size is not None
    ):
        raise ValueError(
            "DiCo-NLI subsets must use prepare-data source-group sampling, not first-N rows."
        )
    if Path(script_args.dataset_name).is_dir():
        dataset = load_from_disk(script_args.dataset_name)
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    if not isinstance(dataset, DatasetDict):
        raise ValueError("Training requires a DatasetDict with named splits.")
    train = dataset[script_args.dataset_train_split]
    evaluation = (
        dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None
    )
    if script_args.train_subset_size is not None:
        train = train.select(range(script_args.train_subset_size))
    if evaluation is not None and script_args.eval_subset_size is not None:
        evaluation = evaluation.select(range(script_args.eval_subset_size))
    require_labels = bool(LABEL_REWARDS & set(script_args.reward_funcs))
    allowed = LABELS if dico else None
    if "allowed_labels" in train.column_names or (
        evaluation is not None and "allowed_labels" in evaluation.column_names
    ):
        raise ValueError("allowed_labels is a reward setting, not a dataset metadata column.")
    validate_prompt_dataset(train, require_labels=require_labels, allowed_labels=allowed)
    if evaluation is not None:
        validate_prompt_dataset(evaluation, require_labels=require_labels, allowed_labels=allowed)
    return train, evaluation
