"""Strict UTF-8 CSV loading and byte-level provenance, without normalization."""
import csv
import hashlib
import io
from pathlib import Path

MAX_BYTES = 64 * 1024 * 1024


def read_csv(path, *, required, allowed, expected_sha256=None):
    path = Path(path)
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError(f"{path}: expected a nonempty CSV file of at most {MAX_BYTES} bytes.")
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"{path}: SHA-256 mismatch; input differs from the pinned file.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: expected UTF-8.") from error
    if "\x00" in text:
        raise ValueError(f"{path}: NUL bytes are not allowed.")
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader, [])
        if not header or any(not item or item != item.strip() for item in header):
            raise ValueError(f"{path}: invalid CSV header.")
        if len(header) != len(set(header)):
            raise ValueError(f"{path}: duplicate header columns.")
        missing, extra = set(required) - set(header), set(header) - set(allowed)
        if missing or extra:
            raise ValueError(
                f"{path}: missing columns {sorted(missing)}; unexpected columns {sorted(extra)}."
            )
        rows = []
        for values in reader:
            if len(values) != len(header):
                raise ValueError(
                    f"{path}: wrong field count or blank row at line {reader.line_num}."
                )
            rows.append(dict(zip(header, values)))
    except csv.Error as error:
        raise ValueError(f"{path}: malformed CSV at line {reader.line_num}: {error}") from error
    if not rows:
        raise ValueError(f"{path}: CSV has no records.")
    return rows, {
        "path": str(path.resolve()),
        "sha256": digest,
        "bytes": len(content),
        "rows": len(rows),
    }
