"""Summarize source units and label/language counts without task scoring."""
from collections import Counter
from itertools import combinations

from .pairing import validate_split_isolation


def summarize_split(records):
    links = {
        tuple(sorted((row.instance_id, row.reverse_pair_id)))
        for row in records
        if row.reverse_pair_id is not None
    }
    return {
        "instances": len(records),
        "source_pairs": len({row.pair_id for row in records}),
        "reciprocal_pairs": len(links),
        "pairing_unknown_instances": sum(not row.pairing_available for row in records),
        "negative_instances": sum(row.label == "NEGATIVE_OTHER" for row in records),
        "label_counts": dict(sorted(Counter(row.label or "UNLABELED" for row in records).items())),
        "language_counts": dict(
            sorted(Counter(f"{row.text1_lang}-{row.text2_lang}" for row in records).items())
        ),
    }


def audit_splits(splits):
    """Exact surface overlaps are warnings, not evidence of shared source identity."""
    validate_split_isolation(splits)
    surfaces = {
        split: {
            tuple(sorted(((row.text1_lang, row.text1), (row.text2_lang, row.text2))))
            for row in records
        }
        for split, records in splits.items()
    }
    overlaps = {
        f"{first}/{second}": len(surfaces[first] & surfaces[second])
        for first, second in combinations(splits, 2)
    }
    examples = {}
    for first, second in combinations(splits, 2):
        common = sorted(surfaces[first] & surfaces[second])[:20]
        examples[f"{first}/{second}"] = [
            {
                "texts": [{"language": language, "text": text} for language, text in key],
                "instance_ids": {
                    name: sorted(
                        row.instance_id
                        for row in splits[name]
                        if tuple(sorted(((row.text1_lang, row.text1), (row.text2_lang, row.text2))))
                        == key
                    )
                    for name in (first, second)
                },
            }
            for key in common
        ]
    return {
        "splits": {name: summarize_split(rows) for name, rows in splits.items()},
        "source_pair_overlap": 0,
        "instance_id_overlap": 0,
        "exact_surface_pair_overlap": overlaps,
        "exact_surface_pair_examples": examples,
        "warnings": [
            f"{names}: {count} identical unordered text/language pairs under different source IDs; "
            "review before experiments. Official splits were not modified."
            for names, count in overlaps.items()
            if count
        ],
    }
