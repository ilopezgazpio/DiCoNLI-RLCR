"""Bounded snapshots and non-overwriting artifact persistence."""
import hashlib
import json
from pathlib import Path

MAX_INPUT_BYTES = 64 * 1024 * 1024


def digest(content):
    return hashlib.sha256(content).hexdigest()


def read_bytes(path):
    path = Path(path)
    with path.open("rb") as stream:
        content = stream.read(MAX_INPUT_BYTES + 1)
    if not 0 < len(content) <= MAX_INPUT_BYTES:
        raise ValueError(f"{path}: expected a nonempty file of at most {MAX_INPUT_BYTES} bytes.")
    return content


def require_new_output(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError(f"Output already exists: {path}. Choose a new directory.")
    return path


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def finish_manifest(output, manifest):
    output = Path(output)
    temporary = output / ".manifest.json.tmp"
    write_json(temporary, manifest)
    temporary.replace(output / "manifest.json")
