"""Persist submission candidates and diagnostics, without reading gold labels."""
import json
from pathlib import Path

from .artifacts import digest, finish_manifest, require_new_output, write_json
from .datasets import load_export_inputs
from .submission import prepare_submission


def export_submission(predictions, instances, *, split, prediction_column, output_dir):
    output = require_new_output(output_dir)
    rows, expected_ids = load_export_inputs(
        predictions, instances, split=split, prediction_column=prediction_column
    )
    content, diagnostics, report = prepare_submission(rows, expected_ids)
    diagnostic_bytes = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in diagnostics
    ).encode("utf-8")
    ids_bytes = json.dumps(sorted(expected_ids), ensure_ascii=False).encode("utf-8")
    output.mkdir(parents=True, exist_ok=False)
    (output / "diagnostics.jsonl").write_bytes(diagnostic_bytes)
    write_json(output / "export-report.json", report)
    if content is not None:
        (output / "submission.csv").write_bytes(content)
    finish_manifest(
        output,
        {
            "schema_version": 1,
            "stage": "submission_export",
            "status": report["status"],
            "predictions_dataset": str(Path(predictions).resolve()),
            "instances_dataset": str(Path(instances).resolve()),
            "split": split,
            "prediction_column": prediction_column,
            "expected_ids_sha256": digest(ids_bytes),
            "diagnostics_sha256": digest(diagnostic_bytes),
            "submission_sha256": digest(content) if content is not None else None,
        },
    )
    if content is None:
        raise ValueError(
            f"Submission blocked: {report['invalid_predictions']}/{report['instances']} invalid "
            f"predictions. Diagnostics saved in {output}; no submission.csv was created."
        )
    return report
