"""Unit tests for the deterministic UTKFace manifest builder."""

from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts/experiments/build_utkface_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_utkface_manifest", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MANIFEST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MANIFEST)


def test_age_bin_observes_right_boundary_convention() -> None:
    assert [MANIFEST.age_bin(age) for age in (19.0, 20.0, 40.0, 60.0, 80.0)] == [0, 1, 2, 3, 4]


def test_manifest_records_are_deterministic_and_consistent(tmp_path: Path) -> None:
    # Enough examples per class for each stratified split operation.
    for age in (10, 25, 45, 65, 85):
        for index in range(24):
            (tmp_path / f"{age}_0_0_{index}.jpg").touch()
    first, first_digest = MANIFEST.build_records(tmp_path, seed=0)
    second, second_digest = MANIFEST.build_records(tmp_path, seed=0)
    MANIFEST.validate_records(first)
    assert MANIFEST.canonical_bytes(first) == MANIFEST.canonical_bytes(second)
    assert first_digest == second_digest
    assert len(first) == 120
    assert {record["canonical_split"] for record in first} == set(MANIFEST.SPLITS)
    assert len({record["sample_id"] for record in first}) == len(first)
