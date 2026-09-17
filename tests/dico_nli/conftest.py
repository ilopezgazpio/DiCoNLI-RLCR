"""Synthetic task fixtures; no official examples or downloads are required."""
import csv
import hashlib

import pytest


@pytest.fixture
def write_csv(tmp_path):
    def write(name, rows, fields=None):
        path = tmp_path / name
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    return write


@pytest.fixture
def task_rows():
    return [
        dict(
            instance_id="f",
            pair_id="p",
            text1_lang="en",
            text2_lang="en",
            text1="a poodle",
            text2="a dog",
            label="FORWARD_ENTAILMENT",
        ),
        dict(
            instance_id="b",
            pair_id="p",
            text1_lang="en",
            text2_lang="en",
            text1="a dog",
            text2="a poodle",
            label="BACKWARD_ENTAILMENT",
        ),
        dict(
            instance_id="e1",
            pair_id="e",
            text1_lang="en",
            text2_lang="en",
            text1="a sofa",
            text2="a couch",
            label="EQUIVALENCE",
        ),
        dict(
            instance_id="e2",
            pair_id="e",
            text1_lang="en",
            text2_lang="en",
            text1="a couch",
            text2="a sofa",
            label="EQUIVALENCE",
        ),
        dict(
            instance_id="n",
            pair_id="n",
            text1_lang="en",
            text2_lang="en",
            text1="a lamp",
            text2="a river",
            label="NEGATIVE_OTHER",
        ),
    ]


@pytest.fixture
def references(task_rows):
    targets = {"f": "b", "b": "f", "e1": "e2", "e2": "e1", "n": ""}
    return [{**row, "reverse_pair_id": targets[row["instance_id"]]} for row in task_rows]


@pytest.fixture
def split_spec(write_csv, task_rows, references):
    return {
        "participant": write_csv("participant.csv", task_rows),
        "reference": write_csv("reference.csv", references),
        "labeled": True,
    }
