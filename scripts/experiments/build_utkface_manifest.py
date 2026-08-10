"""Freeze the canonical UTKFace conference split manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split


THRESHOLDS = (20.0, 40.0, 60.0, 80.0)
SPLITS = ("train", "validation", "calibration", "test")
CONTRACT_VERSION = "0.2.0"
DATASET_VERSION = "utkface-filename-corpus-v1"


def age_bin(age: float, thresholds: tuple[float, ...] = THRESHOLDS) -> int:
    """Return the right-boundary ordinal bin for a finite UTKFace age."""
    if age < thresholds[0]:
        return 0
    for index, threshold in enumerate(thresholds[1:], start=1):
        if age < threshold:
            return index
    return len(thresholds)


def canonical_bytes(records: list[dict[str, Any]]) -> bytes:
    """Serialize records independently of JSON key or platform path ordering."""
    return b"".join(
        (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        for record in records
    )


def parse_source(data_dir: Path) -> list[tuple[str, float, int]]:
    """Return deterministically ordered, parseable UTKFace filename metadata."""
    parsed: list[tuple[str, float, int]] = []
    for path in sorted(data_dir.glob("*.jpg"), key=lambda item: item.name):
        try:
            age = float(path.name.split("_", 1)[0])
        except ValueError:
            continue
        if age >= 0.0 and age < float("inf"):
            parsed.append((path.name, age, age_bin(age)))
    if not parsed:
        raise ValueError(f"No parseable .jpg UTKFace records found in {data_dir}")
    return parsed


def build_records(data_dir: Path, seed: int) -> tuple[list[dict[str, Any]], str]:
    """Build the documented 60/10/20/10 stratified image-level split."""
    source = parse_source(data_dir)
    names = [item[0] for item in source]
    labels = [item[2] for item in source]
    train_names, temporary_names, _, temporary_y = train_test_split(
        names, labels, test_size=0.4, stratify=labels, random_state=seed
    )
    validation_names, remaining_names, _, remaining_y = train_test_split(
        temporary_names, temporary_y, test_size=0.75, stratify=temporary_y, random_state=seed
    )
    calibration_names, test_names = train_test_split(
        remaining_names, test_size=0.3333, stratify=remaining_y, random_state=seed
    )
    split_by_name = {
        **{name: "train" for name in train_names},
        **{name: "validation" for name in validation_names},
        **{name: "calibration" for name in calibration_names},
        **{name: "test" for name in test_names},
    }
    if len(split_by_name) != len(source):
        raise RuntimeError("UTKFace split construction did not assign every source sample exactly once")
    source_index = {name: index for index, (name, _, _) in enumerate(source)}
    records = [
        {
            "sample_id": f"utkface:{name}",
            "source_index": source_index[name],
            "source_split": "corpus",
            "canonical_split": split_by_name[name],
            "Z": age,
            "Y_ord": label,
            "dataset_version": DATASET_VERSION,
        }
        for name, age, label in source
    ]
    records.sort(key=lambda record: record["sample_id"])
    source_digest = hashlib.sha256(
        "".join(f"{name}\t{age:.1f}\t{label}\n" for name, age, label in source).encode("utf-8")
    ).hexdigest()
    return records, source_digest


def validate_records(records: list[dict[str, Any]]) -> None:
    """Validate the immutable manifest invariants before writing it."""
    if len({record["sample_id"] for record in records}) != len(records):
        raise ValueError("Manifest has duplicate sample identifiers")
    if {record["canonical_split"] for record in records} - set(SPLITS):
        raise ValueError("Manifest has an unknown canonical split")
    if any(record["Y_ord"] != age_bin(float(record["Z"])) for record in records):
        raise ValueError("Manifest has a target-label-bin inconsistency")
    if any(record["Y_ord"] not in range(len(THRESHOLDS) + 1) for record in records):
        raise ValueError("Manifest has an out-of-range ordinal label")


def build_metadata(records: list[dict[str, Any]], seed: int, source_digest: str) -> dict[str, Any]:
    """Create durable split metadata for downstream run validation."""
    counts = {
        split: dict(sorted(Counter(record["Y_ord"] for record in records if record["canonical_split"] == split).items()))
        for split in SPLITS
    }
    return {
        "dataset": "utkface",
        "dataset_contract_version": CONTRACT_VERSION,
        "dataset_version": DATASET_VERSION,
        "source_listing_sha256": source_digest,
        "split_policy": "sorted_filename_stratified_60_10_20_10_v1",
        "split_seed": seed,
        "source_subdivision": {
            "source_split": "corpus",
            "rule": "stratified 60/40, then 10/30, then 20/10",
            "reason": "independent train, validation, calibration, and test subsets",
        },
        "bins": {"thresholds": list(THRESHOLDS), "internal_boundary": "right_bin"},
        "manifest_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest(),
        "split_counts": {split: sum(counts[split].values()) for split in SPLITS},
        "class_counts": counts,
        "validation": {
            "unique_sample_ids": True,
            "no_overlap": True,
            "target_label_bin_consistent": True,
            "deterministic_regeneration": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/storage/data/utkface/UTKFace"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/conference_v0_3/utkface"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = args.output / "manifest.jsonl"
    if manifest.exists() and not args.overwrite:
        raise SystemExit(f"{manifest} exists; pass --overwrite after verifying its provenance.")

    records, source_digest = build_records(args.data_dir, args.seed)
    validate_records(records)
    regenerated, regenerated_digest = build_records(args.data_dir, args.seed)
    if canonical_bytes(records) != canonical_bytes(regenerated) or source_digest != regenerated_digest:
        raise RuntimeError("UTKFace manifest regeneration was not deterministic")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest.write_bytes(canonical_bytes(records))
    summary = build_metadata(records, args.seed, source_digest)
    for filename in ("manifest_metadata.json", "split_summary.json"):
        (args.output / filename).write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
