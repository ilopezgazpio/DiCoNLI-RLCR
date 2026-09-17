"""Scorer pinning, explicit setup, and isolation from ambient Python modules."""
import io
import json
import shutil
import subprocess
import sys
from urllib.error import URLError

import pytest

from rlcr.evaluation.dico_nli import scorer_source
from rlcr.evaluation.dico_nli.artifacts import digest
from rlcr.evaluation.dico_nli.official_scorer import run_official_scorer

GOLD = b"instance_id,pair_id,reverse_pair_id,label\na,p,b,EQUIVALENCE\nb,p,a,EQUIVALENCE\n"
PREDICTIONS = b"instance_id,label\na,EQUIVALENCE\nb,EQUIVALENCE\n"


@pytest.fixture
def fake_source(monkeypatch):
    content = b"synthetic source bytes"
    monkeypatch.setattr(
        scorer_source, "FILE_HASHES", {"evaluation_functions/example.py": digest(content)}
    )
    return content


def test_fetch_verifies_and_reuses_without_network(fake_source, monkeypatch, tmp_path):
    calls = []

    def download(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(fake_source)

    monkeypatch.setattr(scorer_source, "urlopen", download)
    destination = tmp_path / "scorer"
    assert scorer_source.fetch_scorer(destination)["downloaded"] is True
    assert scorer_source.verified_scorer_files(destination) == {
        "evaluation_functions/example.py": fake_source
    }
    assert scorer_source.fetch_scorer(destination)["downloaded"] is False
    assert len(calls) == 1
    assert scorer_source.REVISION in calls[0][0]


def test_invalid_download_does_not_publish_cache(fake_source, monkeypatch, tmp_path):
    monkeypatch.setattr(scorer_source, "urlopen", lambda *a, **k: io.BytesIO(b"changed"))
    with pytest.raises(ValueError, match="Downloaded scorer checksum mismatch"):
        scorer_source.fetch_scorer(tmp_path / "scorer")
    assert not (tmp_path / "scorer").exists()


def test_network_error_does_not_publish_cache(fake_source, monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise URLError("offline")

    monkeypatch.setattr(scorer_source, "urlopen", fail)
    with pytest.raises(ValueError, match="Could not fetch"):
        scorer_source.fetch_scorer(tmp_path / "scorer")
    assert not (tmp_path / "scorer").exists()


def test_partial_install_is_not_replaced(fake_source, tmp_path):
    with pytest.raises(ValueError, match="Missing scorer file"):
        scorer_source.fetch_scorer(tmp_path)
    assert not list(tmp_path.iterdir())


def test_missing_scorer_fails_before_execution(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Unverified code must not run")

    monkeypatch.setattr(subprocess, "run", unexpected)
    with pytest.raises(ValueError, match="fetch-scorer"):
        run_official_scorer(GOLD, PREDICTIONS, scorer_dir=tmp_path)


def test_modified_scorer_fails_before_execution(official_scorer_dir, tmp_path, monkeypatch):
    directory = tmp_path / "scorer"
    shutil.copytree(official_scorer_dir, directory)
    (directory / "evaluation_functions/metrics.py").write_text("raise RuntimeError('changed')")

    def unexpected(*args, **kwargs):
        pytest.fail("Unverified code must not run")

    monkeypatch.setattr(subprocess, "run", unexpected)
    with pytest.raises(ValueError, match="checksum mismatch"):
        run_official_scorer(GOLD, PREDICTIONS, scorer_dir=directory)


def test_ambient_modules_and_unpinned_cache_files_are_not_used(
    official_scorer_dir, tmp_path, monkeypatch
):
    directory = tmp_path / "scorer"
    shutil.copytree(official_scorer_dir, directory)
    (directory / "evaluation_functions.py").write_text("raise RuntimeError('unpinned')")
    ambient = tmp_path / "ambient"
    ambient.mkdir()
    (ambient / "evaluation_functions.py").write_text("raise RuntimeError('ambient')")
    (ambient / "sitecustomize.py").write_text("raise RuntimeError('sitecustomize')")
    monkeypatch.setenv("PYTHONPATH", str(ambient))
    monkeypatch.chdir(ambient)
    report, _ = run_official_scorer(GOLD, PREDICTIONS, scorer_dir=directory)
    assert report["weighted_f1"] == 1


def test_raw_scoring_does_not_import_ml_libraries(official_scorer_dir, tmp_path):
    (tmp_path / "gold.tsv").write_bytes(GOLD.replace(b",", b"\t"))
    (tmp_path / "pred.tsv").write_bytes(PREDICTIONS.replace(b",", b"\t"))
    program = (
        "import sys; from rlcr.evaluation.dico_nli.scoring import score_submission; "
        "result = score_submission('pred.tsv', gold='gold.tsv', output_dir='scores', "
        "scorer_dir=sys.argv[1]); assert result['hard_cons'] == 1; "
        "assert not {'torch', 'datasets', 'transformers', 'trl'} & sys.modules.keys()"
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(official_scorer_dir)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_cli_export_then_score_outside_checkout(official_scorer_dir, tmp_path):
    from datasets import Dataset

    Dataset.from_dict({"instance_id": ["b", "a"]}).save_to_disk(tmp_path / "instances")
    Dataset.from_dict(
        {
            "instance_id": ["a", "b"],
            "candidate-output_0": ["<answer>EQUIVALENCE</answer><confidence>0.8</confidence>"] * 2,
        }
    ).save_to_disk(tmp_path / "predictions")
    (tmp_path / "gold.csv").write_bytes(GOLD)
    for args in [
        [
            "export-submission",
            "--instances",
            "instances",
            "--predictions",
            "predictions",
            "--prediction-column",
            "candidate-output_0",
            "--output-dir",
            "export",
        ],
        [
            "score",
            "--gold",
            "gold.csv",
            "--predictions",
            "export/submission.csv",
            "--scorer-dir",
            str(official_scorer_dir),
            "--output-dir",
            "scores",
        ],
    ]:
        result = subprocess.run(
            [sys.executable, "-m", "rlcr", *args],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "scores/scores.json").read_text())["soft_cons"] == 1


def test_cli_requires_one_explicit_gold_source(tmp_path):
    for extra in [[], ["--gold", "x", "--reference-dataset", "y"]]:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "rlcr",
                "score",
                "--predictions",
                "x",
                "--output-dir",
                "scores",
                *extra,
            ],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            timeout=10,
        )
        assert result.returncode == 2
        assert not (tmp_path / "scores").exists()
