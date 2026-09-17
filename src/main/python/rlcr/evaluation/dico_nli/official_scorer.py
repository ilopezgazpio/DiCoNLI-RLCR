"""Execute verified, unchanged upstream code in an isolated stdlib-only process."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from .scorer_source import verified_scorer_files

# No ambient PYTHONPATH/site-packages, cached bytecode, or installed scorer is used.
_LAUNCH = (
    "import runpy,sys; "
    "sys.path.insert(0,sys.argv.pop(1)); "
    "runpy.run_module('evaluation_functions',run_name='__main__')"
)


def run_official_scorer(reference, predictions, *, scorer_dir):
    source_files = verified_scorer_files(scorer_dir)
    with TemporaryDirectory(prefix="rlcr-dico-score-") as temporary:
        root = Path(temporary)
        for name, content in source_files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (root / "reference.csv").write_bytes(reference)
        (root / "submission.csv").write_bytes(predictions)
        command = [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            _LAUNCH,
            str(root),
            "--gold",
            str(root / "reference.csv"),
            "--predictions",
            str(root / "submission.csv"),
            "--output-dir",
            str(root / "scores"),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired as error:
            raise ValueError("Official scorer exceeded the 60-second timeout.") from error
        if result.returncode:
            raise ValueError(f"Official scorer rejected the run: {result.stderr.strip()}")
        score_json = (root / "scores/scores.json").read_bytes()
        score_text = (root / "scores/scores.txt").read_bytes()
        return json.loads(score_json), {"scores.json": score_json, "scores.txt": score_text}
