"""Gold-free export contract: exact coverage, no implicit repairs or selected winners."""
import csv
import io
import json

from datasets import Dataset, DatasetDict, load_from_disk
import pytest

from rlcr.evaluation.dico_nli.export import export_submission
from rlcr.evaluation.dico_nli.submission import prepare_submission


def response(label="EQUIVALENCE", confidence="0.75"):
    return f"<answer>{label}</answer><confidence>{confidence}</confidence>"


def test_two_column_csv_and_separate_confidence():
    rows = [
        {"instance_id": "b", "completion": response("BACKWARD_ENTAILMENT", "0")},
        {"instance_id": "a", "completion": response("FORWARD_ENTAILMENT", "1")},
    ]
    data, diagnostics, report = prepare_submission(rows, ["a", "b"])
    reader = csv.DictReader(io.StringIO(data.decode()))
    assert reader.fieldnames == ["instance_id", "label"]
    assert list(reader) == [
        {"instance_id": "a", "label": "FORWARD_ENTAILMENT"},
        {"instance_id": "b", "label": "BACKWARD_ENTAILMENT"},
    ]
    assert report["status"] == "ready"
    assert [row["confidence"] for row in diagnostics] == [1, 0]
    assert data == prepare_submission(list(reversed(rows)), ["b", "a"])[0]


@pytest.mark.parametrize(
    "completion",
    [
        None,
        "",
        "EQUIVALENCE",
        response("entailment"),
        response("UNKNOWN"),
        response(confidence="NaN"),
        response(confidence="inf"),
        response(confidence="-1"),
        response(confidence="1.1"),
        response(confidence="75%"),
        response() + response(),
        "reasoning " + response(),
    ],
)
def test_invalid_candidates_never_become_partial_submissions(completion):
    rows = [
        {"instance_id": "a", "completion": response()},
        {"instance_id": "b", "completion": completion},
    ]
    data, diagnostics, report = prepare_submission(rows, ["a", "b"])
    assert data is None
    assert report == {
        "instances": 2,
        "valid_predictions": 1,
        "invalid_predictions": 1,
        "invalid_rate": 0.5,
        "status": "blocked",
    }
    assert diagnostics[1]["raw_completion"] == completion
    assert diagnostics[1]["error"]


def test_conversational_response_is_supported():
    data, _, report = prepare_submission(
        [{"instance_id": "a", "completion": [{"role": "assistant", "content": response()}]}], ["a"]
    )
    assert report["status"] == "ready"
    assert b"EQUIVALENCE" in data


@pytest.mark.parametrize(
    "expected,ids",
    [
        ([], []),
        (["a", "a"], ["a"]),
        (["a"], ["a", "a"]),
        (["a", "b"], ["a"]),
        (["a"], ["a", "b"]),
        (["a"], ["unknown"]),
        ([""], [""]),
        (["bad id"], ["bad id"]),
        (["a\x00"], ["a\x00"]),
        ([None], [None]),
    ],
)
def test_invalid_ids_or_coverage_fail(expected, ids):
    with pytest.raises(ValueError):
        prepare_submission(
            [{"instance_id": identifier, "completion": response()} for identifier in ids], expected
        )


@pytest.fixture
def export_data(tmp_path):
    expected = tmp_path / "instances"
    predictions = tmp_path / "predictions"
    DatasetDict(
        {
            "dev": Dataset.from_dict(
                {"instance_id": ["a", "b"], "label": ["DO_NOT_READ", "DO_NOT_READ"]}
            )
        }
    ).save_to_disk(expected)
    Dataset.from_dict(
        {
            "instance_id": ["b", "a"],
            "candidate-output_0": [response(), response("NEGATIVE_OTHER")],
            "label": ["SECRET", "SECRET"],
            "candidate-output_1": [response("UNKNOWN")] * 2,
        }
    ).save_to_disk(predictions)
    return expected, predictions


def test_export_reads_only_explicit_prediction_column(export_data, tmp_path):
    expected, predictions = export_data
    output = tmp_path / "export"
    report = export_submission(
        predictions,
        expected,
        split="dev",
        prediction_column="candidate-output_0",
        output_dir=output,
    )
    assert report["status"] == "ready"
    assert (output / "submission.csv").read_text() == (
        "instance_id,label\na,NEGATIVE_OTHER\nb,EQUIVALENCE\n"
    )
    assert "SECRET" not in (output / "diagnostics.jsonl").read_text()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["submission_sha256"]
    assert manifest["prediction_column"] == "candidate-output_0"
    assert load_from_disk(predictions)["label"] == ["SECRET", "SECRET"]


def test_failed_export_retains_diagnostics_but_no_submission(export_data, tmp_path):
    expected, predictions = export_data
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="Submission blocked"):
        export_submission(
            predictions,
            expected,
            split="dev",
            prediction_column="candidate-output_1",
            output_dir=output,
        )
    assert not (output / "submission.csv").exists()
    assert json.loads((output / "export-report.json").read_text())["invalid_rate"] == 1
    assert len((output / "diagnostics.jsonl").read_text().splitlines()) == 2
    assert json.loads((output / "manifest.json").read_text())["status"] == "blocked"


@pytest.mark.parametrize("column", ["label", "pair_id", "prompt", "messages", "missing-output"])
def test_metadata_or_missing_prediction_column_is_rejected(export_data, tmp_path, column):
    expected, predictions = export_data
    with pytest.raises(ValueError):
        export_submission(
            predictions,
            expected,
            split="dev",
            prediction_column=column,
            output_dir=tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


def test_self_validating_prediction_id_set_is_rejected(export_data, tmp_path):
    _, predictions = export_data
    with pytest.raises(ValueError, match="independent"):
        export_submission(
            predictions,
            predictions,
            split="dev",
            prediction_column="candidate-output_0",
            output_dir=tmp_path / "out",
        )


def test_existing_export_never_overwritten(export_data, tmp_path):
    expected, predictions = export_data
    output = tmp_path / "out"
    output.mkdir()
    marker = output / "marker"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="already exists"):
        export_submission(
            predictions,
            expected,
            split="dev",
            prediction_column="candidate-output_0",
            output_dir=output,
        )
    assert marker.read_text() == "keep"
