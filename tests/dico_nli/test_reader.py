"""Validate the boundary between official CSVs and internal task records."""
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from rlcr.data.dico_nli.labels import LABELS, reverse_label
from rlcr.data.dico_nli.reader import load_split


def test_label_contract():
    assert len(LABELS) == 4
    for label in LABELS[:3]:
        assert reverse_label(reverse_label(label)) == label
    with pytest.raises(ValueError, match="no deterministic reverse"):
        reverse_label("NEGATIVE_OTHER")


def test_join_uses_ids_and_preserves_text(split_spec, task_rows, references, write_csv):
    task_rows[0]["text1"] = references[0]["text1"] = "  a poodle,\n犬  "
    task_rows[1]["text2"] = references[1]["text2"] = "  a poodle,\n犬  "
    split_spec["participant"] = write_csv("participant.csv", task_rows)
    split_spec["reference"] = write_csv("reference.csv", list(reversed(references)))
    records, sources = load_split(**split_spec)
    assert records[0].text1 == "  a poodle,\n犬  "
    assert records[0].reverse_pair_id == "b"
    assert all(row.pairing_available for row in records)
    assert records[-1].reverse_pair_id is None
    assert sources["participant"]["sha256"] == split_spec["participant"]["sha256"]
    with pytest.raises(FrozenInstanceError):
        records[0].label = "NEGATIVE_OTHER"


def test_minimal_reference_is_supported(split_spec, references, write_csv):
    fields = ("instance_id", "pair_id", "reverse_pair_id", "label")
    split_spec["reference"] = write_csv(
        "minimal.csv", [{key: row[key] for key in fields} for row in references]
    )
    assert load_split(**split_spec)[0][0].pairing_available


def test_labeled_data_without_reference_has_unknown_pairing(split_spec):
    split_spec.pop("reference")
    records, _ = load_split(**split_spec)
    assert all(not row.pairing_available and row.reverse_pair_id is None for row in records)


def test_unlabeled_inputs_need_no_reference(task_rows, write_csv):
    rows = [{key: value for key, value in row.items() if key != "label"} for row in task_rows]
    records, _ = load_split(write_csv("inference.csv", rows), labeled=False)
    assert all(row.label is None and not row.pairing_available for row in records)


def test_unlabeled_path_rejects_gold_fields_and_reference(split_spec):
    with pytest.raises(ValueError, match="cannot be joined"):
        load_split(**{**split_spec, "labeled": False})
    with pytest.raises(ValueError, match="unexpected columns"):
        load_split(split_spec["participant"], labeled=False)


@pytest.mark.parametrize(
    "column,value",
    [
        ("instance_id", ""),
        ("instance_id", "bad id"),
        ("pair_id", " p"),
        ("text1", " "),
        ("text2", ""),
        ("label", "entailment"),
        ("label", " EQUIVALENCE"),
        ("text1_lang", "fr"),
        ("text2_lang", "EN"),
    ],
)
def test_bad_participant_fields(task_rows, write_csv, column, value):
    task_rows[0][column] = value
    with pytest.raises(ValueError):
        load_split(write_csv("bad.csv", task_rows), labeled=True)


@pytest.mark.parametrize(
    "change,message",
    [
        ("duplicate", "duplicate instance_id"),
        ("missing", "sets differ"),
        ("extra", "sets differ"),
        ("pair", "pair_id mismatch"),
        ("label", "label mismatch"),
        ("text", "text1 mismatch"),
        ("language", "text1_lang mismatch"),
    ],
)
def test_bad_reference_join(split_spec, references, write_csv, change, message):
    if change == "duplicate":
        references.append(references[0])
    elif change == "missing":
        references.pop()
    elif change == "extra":
        references.append({**references[0], "instance_id": "extra"})
    else:
        field, value = {
            "pair": ("pair_id", "other"),
            "label": ("label", "EQUIVALENCE"),
            "text": ("text1", "wrong"),
            "language": ("text1_lang", "es"),
        }[change]
        references[0][field] = value
    split_spec["reference"] = write_csv("reference.csv", references)
    with pytest.raises(ValueError, match=message):
        load_split(**split_spec)


@pytest.mark.parametrize(
    "index,target,message",
    [
        (0, "f", "self or unknown"),
        (0, "missing", "self or unknown"),
        (0, "", "missing its reverse"),
        (0, "e1", "non-reciprocal"),
        (4, "f", "NEGATIVE_OTHER"),
        (0, "b ", "whitespace"),
    ],
)
def test_bad_reverse_links(split_spec, references, write_csv, index, target, message):
    references[index]["reverse_pair_id"] = target
    split_spec["reference"] = write_csv("reference.csv", references)
    with pytest.raises(ValueError, match=message):
        load_split(**split_spec)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("pair_id", "wrong-source", "crosses source"),
        ("label", "FORWARD_ENTAILMENT", "incompatible reverse"),
        ("text1", "unrelated phrase", "not swapped"),
        ("text1_lang", "es", "not swapped"),
    ],
)
def test_pairs_must_match_beyond_reference_join(
    split_spec, task_rows, references, write_csv, field, value, message
):
    task_rows[1][field] = references[1][field] = value
    split_spec["participant"] = write_csv("participant.csv", task_rows)
    split_spec["reference"] = write_csv("reference.csv", references)
    with pytest.raises(ValueError, match=message):
        load_split(**split_spec)


def test_hash_changes_rejected_before_parsing(split_spec):
    path = Path(split_spec["participant"]["path"])
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_split(**split_spec)


@pytest.mark.parametrize(
    "content,message",
    [
        ("", "nonempty CSV"),
        ("instance_id,instance_id\na,b\n", "duplicate header"),
        ("instance_id,\na,b\n", "invalid CSV header"),
        ("instance_id,pair_id\na,p\n", "missing columns"),
        ("instance_id,pair_id,text1_lang,text2_lang,text1,text2,label\n", "no records"),
        (
            "instance_id,pair_id,text1_lang,text2_lang,text1,text2,label\na,p,en,en,x,y,EQUIVALENCE,z\n",
            "field count",
        ),
        (
            "instance_id,pair_id,text1_lang,text2_lang,text1,text2,label\na,p,en,en,x\n",
            "field count",
        ),
        ("instance_id,pair_id,text1_lang,text2_lang,text1,text2,label\n\n", "blank row"),
        (
            'instance_id,pair_id,text1_lang,text2_lang,text1,text2,label\n"unterminated',
            "malformed CSV",
        ),
        ("header\x00\nvalue\n", "NUL bytes"),
    ],
)
def test_malformed_csv_fails(tmp_path, content, message):
    path = tmp_path / "bad.csv"
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        load_split({"path": path}, labeled=True)


def test_duplicate_participant_is_rejected(task_rows, write_csv):
    with pytest.raises(ValueError, match="duplicate instance_id"):
        load_split(write_csv("bad.csv", task_rows + task_rows[:1]), labeled=True)


def test_bom_is_accepted_but_invalid_utf8_is_not(split_spec, tmp_path):
    path = tmp_path / "input.csv"
    path.write_bytes(b"\xef\xbb\xbf" + Path(split_spec["participant"]["path"]).read_bytes())
    assert len(load_split({"path": path}, labeled=True)[0]) == 5
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        load_split({"path": path}, labeled=True)
