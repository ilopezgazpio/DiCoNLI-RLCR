"""Source grouping, leakage checks, and deterministic small-data sampling."""
from dataclasses import replace

import pytest

from rlcr.data.dico_nli.audit import audit_splits
from rlcr.data.dico_nli.pairing import validate_split_isolation
from rlcr.data.dico_nli.reader import load_split
from rlcr.data.dico_nli.sampling import select_source_pairs


def replicate(records, count):
    return [
        replace(
            row,
            instance_id=f"{index}-{row.instance_id}",
            pair_id=f"{index}-{row.pair_id}",
            reverse_pair_id=f"{index}-{row.reverse_pair_id}" if row.reverse_pair_id else None,
        )
        for index in range(count)
        for row in records
    ]


def test_sampling_preserves_strata_and_complete_pairs(split_spec):
    records = replicate(load_split(**split_spec)[0], 12)
    selected = select_source_pairs(records, 9, seed=42)
    assert len({row.pair_id for row in selected}) == 9
    assert {row.label for row in selected} == {row.label for row in records}
    assert selected == select_source_pairs(list(reversed(records)), 9, seed=42)
    assert selected != select_source_pairs(records, 9, seed=43)
    ids = {row.instance_id for row in selected}
    assert all(row.reverse_pair_id in ids for row in selected if row.reverse_pair_id)
    assert len(selected) == 15


@pytest.mark.parametrize("maximum", [0, -1, True, 1.5, 2, 1000])
def test_impossible_budgets_are_rejected(split_spec, maximum):
    with pytest.raises(ValueError, match="max_source_pairs"):
        select_source_pairs(load_split(**split_spec)[0], maximum)


def test_multiple_language_variants_stay_in_one_source_group(split_spec):
    english = load_split(**split_spec)[0]
    spanish = [
        replace(
            row,
            instance_id=f"es-{row.instance_id}",
            text1_lang="es",
            text2_lang="es",
            reverse_pair_id=f"es-{row.reverse_pair_id}" if row.reverse_pair_id else None,
        )
        for row in english
    ]
    records = replicate(english + spanish, 4)
    selected = select_source_pairs(records, 3)
    for pair_id in {row.pair_id for row in selected}:
        assert {row.instance_id for row in selected if row.pair_id == pair_id} == {
            row.instance_id for row in records if row.pair_id == pair_id
        }


def test_cross_language_reversal_preserves_language_direction(split_spec):
    records = load_split(**split_spec)[0][:2]
    mixed = [
        replace(records[0], text1_lang="en", text2_lang="eu"),
        replace(records[1], text1_lang="eu", text2_lang="en"),
    ]
    assert len(select_source_pairs(mixed, 1)) == 2


@pytest.mark.parametrize("shared", ["instance_id", "pair_id"])
def test_cross_split_leakage_detected_even_with_different_language(split_spec, shared):
    records = load_split(**split_spec)[0]
    other = replicate(records, 1)
    other = [replace(row, text1_lang="es", text2_lang="es") for row in other]
    other[0] = replace(other[0], **{shared: getattr(records[0], shared)})
    with pytest.raises(ValueError, match="Split leakage"):
        validate_split_isolation({"train": records, "dev": other})


def test_same_text_different_sources_is_a_warning_not_deduplication(split_spec):
    records = load_split(**split_spec)[0]
    other = replicate(records, 1)
    report = audit_splits({"train": records, "dev": other})
    assert report["source_pair_overlap"] == 0
    assert report["exact_surface_pair_overlap"] == {"train/dev": 3}
    assert report["warnings"]
    assert report["splits"]["train"]["instances"] == 5
    assert report["splits"]["train"]["source_pairs"] == 3
    assert report["splits"]["train"]["reciprocal_pairs"] == 2


def test_sampling_does_not_hide_malformed_source_records(split_spec):
    records = load_split(**split_spec)[0]
    records[0] = replace(records[0], reverse_pair_id="bad")
    with pytest.raises(ValueError, match="unknown reverse"):
        select_source_pairs(records, 3)


def test_unlabeled_sampling_has_no_invented_pairs(split_spec):
    records = [
        replace(row, label=None, pairing_available=False, reverse_pair_id=None)
        for row in load_split(**split_spec)[0]
    ]
    result = select_source_pairs(records, 2)
    assert len({row.pair_id for row in result}) == 2
    assert all(row.label is None and not row.pairing_available for row in result)
