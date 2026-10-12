"""Stream QA records and atomically replace each complete output file."""

from contextlib import contextmanager
from itertools import islice
import json
import os
from pathlib import Path
import tempfile


@contextmanager
def atomic_text(path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_records(records, path, fmt, first_record_fields=None):
    count = 0
    with atomic_text(path) as stream:
        if fmt == "json":
            for record in records:
                if count == 0 and first_record_fields:
                    record = {**first_record_fields, **record}
                stream.write("[\n" if count == 0 else ",\n")
                encoded = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n".join("  " + line for line in encoded.split("\n")))
                count += 1
            stream.write("\n]" if count else "[]")
        else:
            for record in records:
                if count == 0 and first_record_fields:
                    record = {**first_record_fields, **record}
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
    return count


def write_output(qa_pairs, output_cfg, first_record_fields=None):
    """Write records, prepending optional fields to the first record of each file."""
    fmt, path = output_cfg.get("format", "json"), output_cfg["path"]
    if fmt not in {"json", "jsonl"}:
        raise ValueError(f"Unsupported output format: {fmt}")
    chunk_size = output_cfg.get("chunk_size")
    if chunk_size is not None and (type(chunk_size) is not int or chunk_size <= 0):
        raise ValueError("chunk_size must be a positive integer")
    if not chunk_size:
        return _write_records(qa_pairs, path, fmt, first_record_fields)
    records, total, chunk_index = iter(qa_pairs), 0, 0
    base, ext = os.path.splitext(path)
    while True:
        chunk = list(islice(records, chunk_size))
        if not chunk:
            break
        chunk_index += 1
        count = _write_records(chunk, f"{base}_{chunk_index}{ext}", fmt, first_record_fields)
        total += count
        print(f"Wrote chunk {chunk_index}: {count} samples")
    return total
