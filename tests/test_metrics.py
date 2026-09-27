import numpy as np

from training.metrics import evaluate, select_thresholds


def test_thresholds_create_uncertain_zone() -> None:
    labels = np.asarray(
        [0] * 100
        + [1] * 100,
        dtype=np.int64,
    )

    probabilities = np.concatenate(
        [
            np.linspace(
                0.01,
                0.45,
                100,
            ),
            np.linspace(
                0.55,
                0.99,
                100,
            ),
        ]
    )

    human, ai = select_thresholds(
        labels,
        probabilities,
    )

    assert human < ai
    assert 0.03 <= human <= 0.45
    assert 0.55 <= ai <= 0.97


def test_evaluate_reports_selective_metrics() -> None:
    labels = np.asarray(
        [
            0,
            0,
            1,
            1,
        ],
        dtype=np.int64,
    )

    probabilities = np.asarray(
        [
            0.05,
            0.30,
            0.70,
            0.95,
        ],
        dtype=np.float64,
    )

    metrics = evaluate(
        labels,
        probabilities,
        0.10,
        0.90,
    )

    assert (
        metrics[
            "selective_coverage"
        ]
        == 0.5
    )

    assert (
        metrics[
            "selective_accuracy"
        ]
        == 1.0
    )