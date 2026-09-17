"""Validate all inputs, audit isolation, sample whole source groups, then persist."""
from pathlib import Path

from .audit import audit_splits
from .configuration import load_preparation_config
from .reader import load_split
from .sampling import select_source_pairs
from .storage import save_prepared_dataset

SCHEMA_VERSION = 1
SAMPLING_VERSION = "source-label-signature-sha256-v1"


def prepare_dico_data(config_path, *, output_dir=None):
    config = load_preparation_config(config_path, output_dir=output_dir)
    destination = Path(config["output_dir"])
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Output already exists: {destination}. Choose a new directory.")
    full, sources = {}, {}
    for name, spec in config["splits"].items():
        full[name], sources[name] = load_split(
            spec["participant"], labeled=spec["labeled"], reference=spec.get("reference")
        )
    full_audit = audit_splits(full)
    selected = {
        name: select_source_pairs(
            records, config["splits"][name].get("max_source_pairs"), seed=config["seed"]
        )
        for name, records in full.items()
    }
    audit = {"full_inputs": full_audit, "selected": audit_splits(selected)}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "validated_task_records",
        "prompt_ready": False,
        "source": config["source"],
        "provenance_note": (
            "Revision is declared by the recipe; file bytes are verified against its SHA-256 pins. "
            "Preparation is offline and does not verify repository membership."
        ),
        "inputs": sources,
        "sampling": {
            "algorithm": SAMPLING_VERSION,
            "seed": config["seed"],
            "max_source_pairs": {
                name: spec.get("max_source_pairs") for name, spec in config["splits"].items()
            },
        },
        "selected_instance_ids": {
            name: [row.instance_id for row in records] for name, records in selected.items()
        },
    }
    save_prepared_dataset(selected, config=config, audit=audit, manifest=manifest)
    return audit
