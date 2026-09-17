"""Gold-free validation of generated responses and exact-ID submission serialization."""
import csv
import io

from rlcr.data.dico_nli.labels import LABELS
from rlcr.text.prediction import parse_prediction


def validate_identifiers(values, context):
    if not values:
        raise ValueError(f"{context}: identifiers must not be empty.")
    seen = set()
    for identifier in values:
        if (
            not isinstance(identifier, str)
            or not identifier
            or "\x00" in identifier
            or any(character.isspace() for character in identifier)
        ):
            raise ValueError(f"{context}: invalid instance identifier {identifier!r}.")
        if identifier in seen:
            raise ValueError(f"{context}: duplicate identifier {identifier!r}.")
        seen.add(identifier)
    return seen


def prepare_submission(rows, expected_ids):
    expected = validate_identifiers(expected_ids, "expected instances")
    predicted = validate_identifiers([row["instance_id"] for row in rows], "predictions")
    if expected != predicted:
        raise ValueError(
            f"Prediction coverage mismatch: missing IDs {sorted(expected - predicted)[:10]}; "
            f"unknown IDs {sorted(predicted - expected)[:10]}."
        )
    diagnostics, valid = [], []
    for row in sorted(rows, key=lambda item: item["instance_id"]):
        parsed = parse_prediction(row["completion"])
        reason = (
            "invalid_structure_or_confidence"
            if parsed is None
            else "unknown_label"
            if parsed[0] not in LABELS
            else None
        )
        diagnostic = {
            "instance_id": row["instance_id"],
            "raw_completion": row["completion"],
            "predicted_label": parsed[0] if parsed else None,
            "confidence": parsed[1] if parsed else None,
            "error": reason,
        }
        diagnostics.append(diagnostic)
        if reason is None:
            valid.append({"instance_id": row["instance_id"], "label": parsed[0]})
    report = {
        "instances": len(rows),
        "valid_predictions": len(valid),
        "invalid_predictions": len(rows) - len(valid),
        "invalid_rate": (len(rows) - len(valid)) / len(rows),
        "status": "ready" if len(valid) == len(rows) else "blocked",
    }
    # No partial submission, synthetic negative labels, repairs, or confidence clamping.
    content = csv_bytes(valid, ("instance_id", "label")) if report["status"] == "ready" else None
    return content, diagnostics, report


def csv_bytes(rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")
