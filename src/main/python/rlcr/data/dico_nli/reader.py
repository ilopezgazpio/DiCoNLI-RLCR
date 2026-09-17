"""Read participant rows and explicitly join optional public gold references."""
from dataclasses import replace

from .csv_io import read_csv
from .labels import LABELS
from .pairing import validate_pairs
from .record import NLIRecord

INPUT_FIELDS = {"instance_id", "pair_id", "text1_lang", "text2_lang", "text1", "text2"}
REFERENCE_FIELDS = {"instance_id", "pair_id", "reverse_pair_id", "label"}
LANGUAGES = {"en", "es", "eu"}


def _identifier(value, field, context):
    if not value or any(character.isspace() for character in value):
        raise ValueError(f"{context}: {field} must be nonempty and contain no whitespace.")


def _index(rows, context):
    indexed = {}
    for row in rows:
        for field in ("instance_id", "pair_id"):
            _identifier(row[field], field, context)
        identifier = row["instance_id"]
        if identifier in indexed:
            raise ValueError(f"{context}: duplicate instance_id {identifier!r}.")
        if "label" in row and row["label"] not in LABELS:
            raise ValueError(f"{context}: invalid label {row['label']!r} for {identifier}.")
        indexed[identifier] = row
    return indexed


def load_split(participant, *, labeled, reference=None):
    """File specs contain path/optional SHA-256. Unlabeled inputs forbid gold joins."""
    if type(labeled) is not bool:
        raise ValueError("labeled must be a boolean.")
    if reference is not None and not labeled:
        raise ValueError("Unlabeled inference inputs cannot be joined to gold references.")
    fields = INPUT_FIELDS | ({"label"} if labeled else set())
    rows, provenance = read_csv(
        participant["path"],
        required=fields,
        allowed=fields,
        expected_sha256=participant.get("sha256"),
    )
    by_id = _index(rows, "participant")
    records = []
    for row in rows:
        for field in ("text1", "text2"):
            if not row[field].strip():
                raise ValueError(f"{row['instance_id']}: {field} must not be empty.")
        for field in ("text1_lang", "text2_lang"):
            if row[field] not in LANGUAGES:
                raise ValueError(f"{row['instance_id']}: unsupported {field} {row[field]!r}.")
        records.append(NLIRecord(**row))
    sources = {"participant": provenance}
    if reference is not None:
        gold, sources["reference"] = read_csv(
            reference["path"],
            required=REFERENCE_FIELDS,
            allowed=REFERENCE_FIELDS | INPUT_FIELDS,
            expected_sha256=reference.get("sha256"),
        )
        gold_by_id = _index(gold, "reference")
        if set(gold_by_id) != set(by_id):
            raise ValueError("Participant/reference instance_id sets differ.")
        joined = []
        for record in records:
            row, target = by_id[record.instance_id], gold_by_id[record.instance_id]
            # Compare every shared field, including optional texts/languages.
            for field in row.keys() & target.keys():
                if row[field] != target[field]:
                    raise ValueError(
                        f"{record.instance_id}: participant/reference {field} mismatch."
                    )
            reverse_id = target["reverse_pair_id"] or None
            if reverse_id is not None:
                _identifier(reverse_id, "reverse_pair_id", record.instance_id)
            joined.append(replace(record, reverse_pair_id=reverse_id, pairing_available=True))
        records = joined
    validate_pairs(records)
    return records, sources
