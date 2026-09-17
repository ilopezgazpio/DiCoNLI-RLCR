"""Offline canonical-record to prompt transformation, with immutable provenance."""
from dataclasses import fields
import hashlib
import json
from pathlib import Path

from .audit import audit_splits
from .labels import LABELS
from .pairing import validate_pairs
from .prompts import build_nli_prompt, INPUT_FIELDS, INSTRUCTIONS, PROMPT_VERSION
from .record import NLIRecord


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _records(dataset, split):
    required = {field.name for field in fields(NLIRecord)}
    if not required <= set(dataset.column_names) or not len(dataset):
        raise ValueError(f"{split}: expected nonempty canonical DiCo-NLI records.")
    if "prompt" in dataset.column_names:
        raise ValueError(f"{split}: prompt already exists; use the original canonical records.")
    records = [NLIRecord(**row) for row in dataset.select_columns(sorted(required))]
    for row in records:
        for key in INPUT_FIELDS:
            if not isinstance(getattr(row, key), str):
                raise ValueError(f"{split}: {key} must be a string.")
        for key in ("instance_id", "pair_id", "reverse_pair_id"):
            value = getattr(row, key)
            if key == "reverse_pair_id" and value is None:
                continue
            if (
                not isinstance(value, str)
                or not value
                or "\x00" in value
                or any(c.isspace() for c in value)
            ):
                raise ValueError(f"{split}: invalid {key}.")
        if row.label is not None and row.label not in LABELS:
            raise ValueError(f"{split}: unknown gold label {row.label!r}.")
        if type(row.pairing_available) is not bool:
            raise ValueError(f"{split}: pairing_available must be boolean.")
        if row.pairing_available and row.label is None:
            raise ValueError(f"{split}: unlabeled records cannot have known reference pairing.")
    validate_pairs(records)
    return records


def prepare_nli_prompts(dataset_path, *, output_dir):
    from datasets import DatasetDict, load_from_disk

    source, output = Path(dataset_path), Path(output_dir)
    if output.exists() or output.is_symlink():
        raise ValueError(f"Output already exists: {output}. Choose a new directory.")
    if not source.is_dir():
        raise ValueError("Prompt preparation requires a local canonical DatasetDict.")
    dataset = load_from_disk(str(source))
    if not isinstance(dataset, DatasetDict) or not dataset:
        raise ValueError("Prompt preparation requires a nonempty DatasetDict with named splits.")
    records = {name: _records(split, name) for name, split in dataset.items()}
    audit = audit_splits(records)
    prompted, hashes = {}, {}
    for name, split in dataset.items():
        # Whitelist before calling the builder. Gold and bookkeeping cannot affect the prompt.
        inputs = split.select_columns(list(INPUT_FIELDS))
        prompts = [build_nli_prompt(**row) for row in inputs]
        prompted[name] = split.add_column("prompt", prompts)
        hashes[name] = {
            "instances": len(split),
            "input_records_sha256": _digest(
                sorted(split.to_list(), key=lambda row: row["instance_id"])
            ),
            "prompts_sha256": _digest(sorted(zip(split["instance_id"], prompts))),
        }
    source_manifest = source / "manifest.json"
    source_bytes = source_manifest.read_bytes() if source_manifest.is_file() else None
    manifest = {
        "schema_version": 1,
        "stage": "model_ready_nli_prompts",
        "prompt_ready": True,
        "prompt_version": PROMPT_VERSION,
        "instructions_sha256": hashlib.sha256(INSTRUCTIONS.encode()).hexdigest(),
        "model_input_fields": list(INPUT_FIELDS),
        "evidence_enabled": False,
        "source_dataset": str(source.resolve()),
        "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest()
        if source_bytes
        else None,
        "splits": hashes,
    }
    # All validation/building completes before the exclusive destination creation.
    output.mkdir(parents=True, exist_ok=False)
    DatasetDict(prompted).save_to_disk(str(output))
    (output / "audit.json").write_bytes(_json_bytes(audit))
    if source_bytes is not None:
        (output / "source-manifest.json").write_bytes(source_bytes)
    (output / ".manifest.json.tmp").write_bytes(_json_bytes(manifest))
    (output / ".manifest.json.tmp").replace(output / "manifest.json")
    return manifest
