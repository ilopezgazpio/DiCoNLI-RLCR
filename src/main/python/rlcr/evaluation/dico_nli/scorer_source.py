"""Explicit scorer acquisition and verification; importing/scoring never downloads."""
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from .artifacts import digest
from .scorer_pin import DEFAULT_SCORER_DIR, DOWNLOAD_ROOT, FILE_HASHES, REVISION

MAX_SCORER_FILE_BYTES = 1024 * 1024


def verified_scorer_files(directory=DEFAULT_SCORER_DIR):
    """Read verified bytes once; execution uses these bytes, never live cache imports."""
    files = {}
    for name, expected in FILE_HASHES.items():
        path = Path(directory) / name
        if not path.is_file():
            raise ValueError(f"Missing scorer file {path}. Run rlcr fetch-scorer first.")
        with path.open("rb") as stream:
            content = stream.read(MAX_SCORER_FILE_BYTES + 1)
        if digest(content) != expected:
            raise ValueError(f"Official scorer checksum mismatch: {path}.")
        files[name] = content
    return files


def fetch_scorer(directory=DEFAULT_SCORER_DIR):
    destination = Path(directory)
    if destination.exists() or destination.is_symlink():
        verified_scorer_files(destination)
        return {"revision": REVISION, "directory": str(destination.resolve()), "downloaded": False}
    files = {}
    for name, expected in FILE_HASHES.items():
        try:
            with urlopen(f"{DOWNLOAD_ROOT}/{name}", timeout=30) as response:
                content = response.read(MAX_SCORER_FILE_BYTES + 1)
        except (URLError, TimeoutError) as error:
            raise ValueError(f"Could not fetch scorer file {name}: {error}") from error
        if digest(content) != expected:
            raise ValueError(f"Downloaded scorer checksum mismatch: {name}.")
        files[name] = content
    # No incomplete installation is published on download/hash failure.
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return {"revision": REVISION, "directory": str(destination.resolve()), "downloaded": True}
