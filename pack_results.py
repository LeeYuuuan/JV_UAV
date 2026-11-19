#!/usr/bin/env python3
"""Snapshot JV diagnostics during training; no checkpoints, no third-party dependencies."""
import argparse
from datetime import datetime, timezone
import gzip
import io
import json
import os
from pathlib import Path
import zlib
from zipfile import ZipFile, ZIP_DEFLATED


class LimitedReader:
    def __init__(self, stream):
        self.stream = stream
        self.remaining = os.fstat(stream.fileno()).st_size

    def read(self, size=-1):
        size = self.remaining if size < 0 else min(size, self.remaining)
        chunk = self.stream.read(size)
        self.remaining -= len(chunk)
        return chunk


def snapshot_jsonl(path, compressed=False):
    """Copy only valid complete lines within the file's initial byte boundary."""
    result = io.BytesIO()
    target = gzip.GzipFile(fileobj=result, mode="wb", mtime=0) if compressed else result
    count = 0
    pending = b""
    truncated = False
    with path.open("rb") as raw:
        bounded = LimitedReader(raw)
        source = gzip.GzipFile(fileobj=bounded, mode="rb") if compressed else bounded
        try:
            while True:
                chunk = source.read1(65536) if compressed else source.read(65536)
                if not chunk:
                    break
                lines = (pending + chunk).split(b"\n")
                pending = lines.pop()
                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        truncated = True
                        continue
                    target.write(line + b"\n")
                    count += 1
        except (EOFError, OSError, zlib.error):
            # An active gzip member need not yet have its footer on disk.
            truncated = True
        finally:
            if compressed:
                source.close()
    truncated = truncated or bool(pending)
    if compressed:
        target.close()  # Finalize a new, readable gzip stream in the ZIP.
    return result.getvalue(), count, truncated


def pack(run_dir, output=None, recent_evals=3):
    run_dir = Path(run_dir).resolve()
    if not (run_dir / "resolved_config.json").is_file():
        raise ValueError(f"No resolved_config.json in {run_dir}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output).resolve() if output else run_dir.with_name(f"{run_dir.name}_{stamp}.zip")
    if output.is_relative_to(run_dir):
        raise ValueError("ZIP output must be outside the run directory")
    manifest = dict(run_dir=str(run_dir), snapshot_started_utc=stamp,
                    checkpoints_included=False, recent_evals=recent_evals, files={}, skipped=[])
    names = ["resolved_config.json", "live_status.json", "latest_eval.json", "train.jsonl", "eval.jsonl",
             "charging_frames.jsonl.gz", "episodes.jsonl.gz", "training_curves.json",
             "training_curves.png", "lower_training_curves.png", "upper_request_probability.png",
             "evaluation_curves.png", "loss_curves.png"]
    paths = [run_dir / name for name in names]
    evals = sorted(p for p in run_dir.glob("eval_*") if p.is_dir())
    if recent_evals:
        for folder in evals[-recent_evals:]:
            paths.extend(p for p in folder.rglob("*") if p.suffix in {".json", ".png"})
    paths.extend((run_dir / "episodes").glob("*.json.gz"))
    provenance = sorted(p for p in (run_dir / "provenance").glob("*") if p.is_dir())
    for folder in provenance:
        paths.extend(folder.glob("*.json"))
    if provenance and (provenance[-1] / "source.zip").is_file():
        paths.append(provenance[-1] / "source.zip")
    # This is an independent read snapshot; training files and checkpoint stay untouched.
    with ZipFile(output, "x", compression=ZIP_DEFLATED, compresslevel=5) as archive:
        for path in paths:
            if not path.is_file():
                continue
            name = path.relative_to(run_dir).as_posix()
            try:
                if path.name.endswith((".jsonl", ".jsonl.gz")):
                    data, lines, partial = snapshot_jsonl(path, path.name.endswith(".gz"))
                    manifest["files"][name] = dict(complete_records=lines, partial_tail_ignored=partial)
                else:
                    # Read exactly the bytes present at open; ignore later growth.
                    with path.open("rb") as source:
                        data = source.read(os.fstat(source.fileno()).st_size)
                    if path.suffix == ".json":
                        json.loads(data)
                    elif path.suffix == ".png" and not data.endswith(b"\x00\x00\x00\x00IEND\xaeB`\x82"):
                        raise ValueError("PNG is still being written")
                    elif path.name.endswith(".json.gz"):
                        json.loads(gzip.decompress(data))
                    manifest["files"][name] = dict(bytes=len(data))
                archive.writestr(name, data)
            except (OSError, ValueError, EOFError, zlib.error) as error:
                manifest["skipped"].append(dict(file=name, reason=str(error)))
        archive.writestr("package_manifest.json", json.dumps(manifest, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--recent-evals", type=int, default=3, help="Recent full evaluation directories; 0 omits them")
    args = parser.parse_args()
    if args.recent_evals < 0:
        parser.error("recent-evals must be nonnegative")
    try:
        output = pack(args.run_dir, args.output, args.recent_evals)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(f"{output} ({output.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
