import importlib


def test_canonical_package_and_legacy_checkpoint_namespace_resolve() -> None:
    canonical = importlib.import_module("ordinal_cqr.models.module")
    legacy = importlib.import_module("ocqr_solar.models.module")

    assert canonical.ResNetQR.__name__ == "ResNetQR"
    assert legacy.ResNetQR.__name__ == "ResNetQR"
