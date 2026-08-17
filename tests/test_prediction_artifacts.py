from ordinal_cqr.experiments.prediction_artifacts import evaluate


def test_structural_metrics_distinguish_fragmented_and_empty_variants():
    rows = [
        {
            "sample_id": "contiguous-covered",
            "Y_ord": 0,
            "Z": 0.0,
            "point_prediction": 0,
            "prediction_set_raw": [0],
            "prediction_set_final": [0],
            "fallback_activated": False,
            "hull_activated": False,
        },
        {
            "sample_id": "fragmented-covered",
            "Y_ord": 2,
            "Z": 2.0,
            "point_prediction": 1,
            "prediction_set_raw": [0, 2],
            "prediction_set_final": [0, 2],
            "fallback_activated": False,
            "hull_activated": False,
        },
        {
            "sample_id": "empty-no-fallback",
            "Y_ord": 3,
            "Z": 3.0,
            "point_prediction": 3,
            "prediction_set_raw": [],
            "prediction_set_final": [],
            "fallback_activated": False,
            "hull_activated": False,
        },
    ]

    aggregate = evaluate(rows, k=5, alpha=0.1, ocqr=True)["aggregate"]

    assert aggregate["avg_sfs"] == 1.0
    assert aggregate["avg_mdj"] == 1.0 / 3.0
    assert aggregate["ccr"] == 1.0 / 3.0
    assert aggregate["contiguous_set_rate"] == 1.0 / 3.0
    assert aggregate["fallback_activation_rate"] == 0.0
