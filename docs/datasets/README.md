# OCQR Dataset Contracts

This directory contains dataset-specific cards and metadata contracts for canonical OCQR experiments.

| File | Dataset | Representation |
|---|---|---|
| `retinamnist.md` | RetinaMNIST | Class-only surrogate embedding, \(Z=Y_{\mathrm{ord}}\) |
| `utkface.md` | UTKFace | Observed chronological age |
| `adience.md` | Adience | Canonical numeric representation not yet finalized |
| `solar_flare.md` | Solar flare prediction | Peak X-ray flux or fixed deterministic transformation |

These documents supplement:

- `../methods/ocqr_theory.md`
- `../methods/ocqr_contract.md`
- `../ocqr_project_summary.md`

In case of inconsistency, the OCQR theory and method contract are authoritative. Dataset-specific choices must be frozen in these cards before calibration data are accessed.

## Status convention

- `draft`: the intended representation is known, but required metadata remains incomplete;
- `provisional`: a canonical representation or other core design decision remains unresolved;
- `active`: all mandatory contract fields and validation artifacts are complete;
- `deprecated`: retained only for historical reproducibility.

## Versioning

Update `card_version` whenever a change can alter retained samples, targets, labels, bins, splits, or reported results. Such changes should also update configuration hashes and invalidate incompatible cached experiment outputs.
