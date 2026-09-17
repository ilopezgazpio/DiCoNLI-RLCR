"""Local dataset boundaries; export reads IDs/predictions, scoring explicitly reads gold."""
from dataclasses import fields
from pathlib import Path

from rlcr.data.dico_nli.pairing import validate_pairs
from rlcr.data.dico_nli.record import NLIRecord
from rlcr.data.dico_nli.labels import LABELS


def load_local_dataset(path, split):
    from datasets import Dataset, DatasetDict, load_from_disk

    if not Path(path).is_dir():
        raise ValueError(
            f"Expected a local saved dataset directory: {path}. No Hub loading occurs."
        )
    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise ValueError(f"Dataset {path} has no split {split!r}.")
        dataset = dataset[split]
    if not isinstance(dataset, Dataset) or not len(dataset):
        raise ValueError(f"Expected a nonempty Dataset: {path}.")
    return dataset


def load_export_inputs(predictions_path, instances_path, *, split, prediction_column):
    if Path(predictions_path).resolve() == Path(instances_path).resolve():
        raise ValueError(
            "Expected instances must come from an independent dataset, not predictions."
        )
    if prediction_column in {field.name for field in fields(NLIRecord)} | {"prompt", "messages"}:
        raise ValueError(
            "prediction_column must be a generated completion column, not task metadata."
        )
    expected = load_local_dataset(instances_path, split)
    predictions = load_local_dataset(predictions_path, split)
    if "instance_id" not in expected.column_names:
        raise ValueError("Expected instances need an instance_id column.")
    required = {"instance_id", prediction_column}
    if not required <= set(predictions.column_names):
        raise ValueError(f"Prediction dataset needs columns {sorted(required)}.")
    # Never expose gold labels/text/pairing metadata to export logic.
    identifiers = list(expected.select_columns(["instance_id"])["instance_id"])
    rows = [
        {"instance_id": row["instance_id"], "completion": row[prediction_column]}
        for row in predictions.select_columns(["instance_id", prediction_column])
    ]
    return rows, identifiers


def reference_from_dataset(path, split):
    """Explicit opt-in reference generation for a complete-pair prepared subset."""
    from .submission import csv_bytes, validate_identifiers

    dataset = load_local_dataset(path, split)
    required = {field.name for field in fields(NLIRecord)}
    if not required <= set(dataset.column_names):
        raise ValueError("Reference dataset needs all canonical DiCo-NLI record fields.")
    records = [NLIRecord(**row) for row in dataset.select_columns(sorted(required))]
    validate_identifiers([row.instance_id for row in records], "reference")
    for row in records:
        validate_identifiers([row.pair_id], "source pair")
        if row.label not in LABELS or row.pairing_available is not True:
            raise ValueError("Reference export requires gold labels and known reference pairing.")
    validate_pairs(records)
    return csv_bytes(
        [
            {
                "instance_id": row.instance_id,
                "pair_id": row.pair_id,
                "reverse_pair_id": row.reverse_pair_id or "",
                "label": row.label,
            }
            for row in sorted(records, key=lambda item: item.instance_id)
        ],
        ("instance_id", "pair_id", "reverse_pair_id", "label"),
    )
