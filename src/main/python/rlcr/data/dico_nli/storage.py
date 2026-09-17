"""Save validated task records and provenance; never replace an existing dataset."""
from dataclasses import asdict, fields
import json
from pathlib import Path

from rlcr.configuration.snapshots import save_resolved_config
from .record import NLIRecord


def save_prepared_dataset(splits, *, config, audit, manifest):
    # Import only when executing preparation, not when importing the data contract.
    from datasets import Dataset, DatasetDict, Features, Value

    output = Path(config["output_dir"])
    # Exclusive creation is the overwrite guard, including concurrent invocations.
    output.mkdir(parents=True, exist_ok=False)
    features = Features(
        {
            field.name: Value("bool" if field.name == "pairing_available" else "string")
            for field in fields(NLIRecord)
        }
    )
    dataset = DatasetDict(
        {
            name: Dataset.from_list([asdict(row) for row in rows], features=features)
            for name, rows in splits.items()
        }
    )
    dataset.save_to_disk(str(output))
    save_resolved_config(output, config)
    with (output / "audit.json").open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    # Written last: absent manifest means the write did not finish successfully.
    manifest_path = output / ".manifest.json.tmp"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    manifest_path.replace(output / "manifest.json")
