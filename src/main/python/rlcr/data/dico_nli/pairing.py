"""Validate public reverse links and disjoint source identities across splits."""
from itertools import combinations

from .labels import reverse_label


def validate_pairs(records):
    by_id = {record.instance_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate instance_id values.")
    for record in records:
        if not record.pairing_available:
            if record.reverse_pair_id is not None:
                raise ValueError(f"{record.instance_id}: reverse link without reference metadata.")
            continue
        target = record.reverse_pair_id
        if record.label == "NEGATIVE_OTHER":
            if target is not None:
                raise ValueError(
                    f"{record.instance_id}: NEGATIVE_OTHER cannot have a reverse link."
                )
            continue
        if target is None:
            raise ValueError(f"{record.instance_id}: reversible label is missing its reverse link.")
        if target == record.instance_id or target not in by_id:
            raise ValueError(f"{record.instance_id}: self or unknown reverse link {target!r}.")
        other = by_id[target]
        if not other.pairing_available or other.reverse_pair_id != record.instance_id:
            raise ValueError(f"{record.instance_id}: non-reciprocal reverse link.")
        if other.pair_id != record.pair_id:
            raise ValueError(f"{record.instance_id}: reverse link crosses source pair_id.")
        if other.label != reverse_label(record.label):
            raise ValueError(f"{record.instance_id}: incompatible reverse labels.")
        if (record.text1, record.text2, record.text1_lang, record.text2_lang) != (
            other.text2,
            other.text1,
            other.text2_lang,
            other.text1_lang,
        ):
            raise ValueError(f"{record.instance_id}: reverse texts/languages are not swapped.")


def validate_split_isolation(splits):
    """Run before subsetting so selection cannot hide leakage."""
    for first, second in combinations(splits, 2):
        for field in ("instance_id", "pair_id"):
            common = {getattr(row, field) for row in splits[first]} & {
                getattr(row, field) for row in splits[second]
            }
            if common:
                raise ValueError(
                    f"Split leakage: {first}/{second} share {field}: {sorted(common)[:5]}."
                )
