"""Integration tests execute the real pinned scorer; never a reimplemented metric."""
import json
import subprocess
import sys

from datasets import Dataset, DatasetDict
import pytest

from rlcr.evaluation.dico_nli.official_scorer import run_official_scorer
from rlcr.evaluation.dico_nli.scoring import score_submission
from rlcr.evaluation.dico_nli.submission import csv_bytes


@pytest.fixture
def gold(references):
    return csv_bytes(references, tuple(references[0]))


def predictions(task_rows, mode="perfect"):
    rows = [{"instance_id": row["instance_id"], "label": row["label"]} for row in task_rows]
    if mode == "all-equivalence":
        for row in rows:
            row["label"] = "EQUIVALENCE"
    elif mode == "consistent-wrong":
        rows[0]["label"], rows[1]["label"] = rows[1]["label"], rows[0]["label"]
    elif mode == "inconsistent":
        rows[1]["label"] = rows[0]["label"]
    elif mode == "negative-wrong":
        rows[-1]["label"] = "EQUIVALENCE"
    return csv_bytes(rows, ("instance_id", "label"))


@pytest.mark.parametrize(
    "mode,f1,soft,hard",
    [
        ("perfect", 1, 1, 1),
        ("all-equivalence", 8 / 35, 1, 0.5),
        ("consistent-wrong", 0.6, 1, 0.5),
        ("inconsistent", 11 / 15, 0.5, 0.5),
        ("negative-wrong", 18 / 25, 1, 1),
    ],
)
def test_hand_computed_metrics(official_scorer_dir, gold, task_rows, mode, f1, soft, hard):
    report, _ = run_official_scorer(
        gold, predictions(task_rows, mode), scorer_dir=official_scorer_dir
    )
    assert report["weighted_f1"] == pytest.approx(f1)
    assert report["soft_cons"] == soft
    assert report["hard_cons"] == hard
    assert report["consistency"]["reversible_pairs"] == 2
    assert report["classification"]["total_instances"] == 5


def test_submission_order_does_not_change_results(official_scorer_dir, gold, task_rows):
    first, _ = run_official_scorer(gold, predictions(task_rows), scorer_dir=official_scorer_dir)
    second, _ = run_official_scorer(
        gold, predictions(list(reversed(task_rows))), scorer_dir=official_scorer_dir
    )
    assert first == second


def test_reference_order_does_not_change_metrics(official_scorer_dir, gold, task_rows, references):
    first, _ = run_official_scorer(gold, predictions(task_rows), scorer_dir=official_scorer_dir)
    reordered = csv_bytes(list(reversed(references)), tuple(references[0]))
    second, _ = run_official_scorer(
        reordered, predictions(task_rows), scorer_dir=official_scorer_dir
    )
    for key in ("weighted_f1", "soft_cons", "hard_cons", "classification"):
        assert first[key] == second[key]


@pytest.mark.parametrize(
    "bad",
    [
        b"",
        b"instance_id,label\n",
        b"instance_id,label\nf,EQUIVALENCE\n",
        b"instance_id,label\nunknown,EQUIVALENCE\n",
        b"instance_id,label\nf,EQUIVALENCE\nf,EQUIVALENCE\n",
        b"instance_id,label,confidence\nf,EQUIVALENCE,0.9\n",
        b"instance_id,label\nf,UNSUPPORTED\n",
        b"instance_id,label\nf,\n",
        b"instance_id,label\nf,NaN\n",
        b"instance_id,label\nf,\x00\n",
    ],
)
def test_official_submission_rejection(official_scorer_dir, gold, bad):
    with pytest.raises(ValueError, match="Official scorer rejected"):
        run_official_scorer(gold, bad, scorer_dir=official_scorer_dir)


@pytest.mark.parametrize(
    "mutation", ["self", "unknown", "nonreciprocal", "source", "label", "duplicate"]
)
def test_invalid_references_are_rejected(official_scorer_dir, references, task_rows, mutation):
    if mutation == "self":
        references[0]["reverse_pair_id"] = "f"
    elif mutation == "unknown":
        references[0]["reverse_pair_id"] = "missing"
    elif mutation == "nonreciprocal":
        references[1]["reverse_pair_id"] = ""
    elif mutation == "source":
        references[1]["pair_id"] = "different"
    elif mutation == "label":
        references[1]["label"] = "FORWARD_ENTAILMENT"
    else:
        references.append(references[0])
    with pytest.raises(ValueError, match="Official scorer rejected"):
        run_official_scorer(
            csv_bytes(references, tuple(references[0])),
            predictions(task_rows),
            scorer_dir=official_scorer_dir,
        )


def test_no_reversible_pairs_does_not_fabricate_zero_scores(official_scorer_dir):
    with pytest.raises(ValueError, match="No reversible pairs"):
        run_official_scorer(
            b"instance_id,pair_id,reverse_pair_id,label\nn,n,,NEGATIVE_OTHER\n",
            b"instance_id,label\nn,NEGATIVE_OTHER\n",
            scorer_dir=official_scorer_dir,
        )


def test_reports_snapshots_and_provenance_match_upstream(
    official_scorer_dir, gold, task_rows, tmp_path
):
    reference_path, prediction_path = tmp_path / "gold.csv", tmp_path / "pred.csv"
    reference_path.write_bytes(gold)
    prediction_path.write_bytes(predictions(task_rows))
    output = tmp_path / "scores"
    report = score_submission(
        prediction_path, gold=reference_path, scorer_dir=official_scorer_dir, output_dir=output
    )
    direct = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "evaluation_functions",
            "--gold",
            str(reference_path),
            "--predictions",
            str(prediction_path),
            "--output-dir",
            str(tmp_path / "direct"),
        ],
        cwd=official_scorer_dir,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert direct.returncode == 0, direct.stderr
    assert report == json.loads(direct.stdout)
    for name in ("scores.json", "scores.txt"):
        assert (output / name).read_bytes() == (tmp_path / "direct" / name).read_bytes()
    assert (output / "reference.csv").read_bytes() == gold
    assert (output / "submission.csv").read_bytes() == predictions(task_rows)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["scorer"]["revision"] == "588968e610197ddc4c440314701cbc587afc4c1b"
    assert len(manifest["scorer"]["file_sha256"]) == 14
    with pytest.raises(ValueError, match="already exists"):
        score_submission(
            prediction_path, gold=reference_path, scorer_dir=official_scorer_dir, output_dir=output
        )


def test_failed_score_does_not_publish_results(official_scorer_dir, gold, tmp_path):
    ref, pred = tmp_path / "ref.csv", tmp_path / "pred.csv"
    ref.write_bytes(gold)
    pred.write_text("instance_id,label\nmissing,EQUIVALENCE\n")
    with pytest.raises(ValueError, match="Official scorer rejected"):
        score_submission(
            pred, gold=ref, scorer_dir=official_scorer_dir, output_dir=tmp_path / "scores"
        )
    assert not (tmp_path / "scores").exists()


def test_score_prepared_subset_without_expanding_reference(
    official_scorer_dir, task_rows, references, tmp_path
):
    canonical = [
        {**row, "reverse_pair_id": ref["reverse_pair_id"] or None, "pairing_available": True}
        for row, ref in zip(task_rows, references)
    ][:2]
    dataset = tmp_path / "data"
    DatasetDict({"dev": Dataset.from_list(canonical)}).save_to_disk(dataset)
    pred = tmp_path / "pred.csv"
    pred.write_bytes(predictions(task_rows[:2]))
    report = score_submission(
        pred,
        reference_dataset=dataset,
        scorer_dir=official_scorer_dir,
        output_dir=tmp_path / "scores",
    )
    assert report["classification"]["total_instances"] == 2
    assert report["consistency"]["reversible_pairs"] == 1
    assert report["weighted_f1"] == 1


@pytest.mark.parametrize("mode", ["unknown-pairing", "unlabeled", "broken-pair"])
def test_prepared_reference_requires_known_complete_gold(task_rows, references, tmp_path, mode):
    from rlcr.evaluation.dico_nli.datasets import reference_from_dataset

    canonical = [
        {**row, "reverse_pair_id": ref["reverse_pair_id"] or None, "pairing_available": True}
        for row, ref in zip(task_rows, references)
    ]
    if mode == "unknown-pairing":
        canonical[0]["pairing_available"] = False
    elif mode == "unlabeled":
        canonical[0]["label"] = None
    else:
        canonical.pop(1)
    Dataset.from_list(canonical).save_to_disk(tmp_path / "data")
    with pytest.raises(ValueError):
        reference_from_dataset(tmp_path / "data", "dev")
