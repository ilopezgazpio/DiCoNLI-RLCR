"""Score immutable input snapshots with the pinned upstream implementation."""
from pathlib import Path

from .artifacts import digest, finish_manifest, read_bytes, require_new_output
from .datasets import reference_from_dataset
from .official_scorer import run_official_scorer
from .scorer_pin import DEFAULT_SCORER_DIR, FILE_HASHES, REPOSITORY, REVISION


def score_submission(
    predictions,
    *,
    output_dir,
    gold=None,
    reference_dataset=None,
    split="dev",
    scorer_dir=DEFAULT_SCORER_DIR,
):
    output = require_new_output(output_dir)
    if (gold is None) == (reference_dataset is None):
        raise ValueError("Supply exactly one gold CSV or reference dataset.")
    submission = read_bytes(predictions)
    reference = (
        read_bytes(gold) if gold is not None else reference_from_dataset(reference_dataset, split)
    )
    report, files = run_official_scorer(reference, submission, scorer_dir=scorer_dir)
    # Invalid submissions/scorer failures must not leave official-looking reports.
    output.mkdir(parents=True, exist_ok=False)
    files.update({"reference.csv": reference, "submission.csv": submission})
    for name, content in files.items():
        (output / name).write_bytes(content)
    finish_manifest(
        output,
        {
            "schema_version": 1,
            "stage": "official_scoring",
            "status": "complete",
            "scorer": {"repository": REPOSITORY, "revision": REVISION, "file_sha256": FILE_HASHES},
            "inputs": {
                "predictions": str(Path(predictions).resolve()),
                "gold": str(Path(gold).resolve()) if gold is not None else None,
                "reference_dataset": str(Path(reference_dataset).resolve())
                if reference_dataset
                else None,
                "split": split if reference_dataset else None,
            },
            "artifact_sha256": {name: digest(content) for name, content in files.items()},
        },
    )
    return report
