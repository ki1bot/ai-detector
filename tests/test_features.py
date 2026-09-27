import numpy as np

from detector.features import (
    STYLE_FEATURE_NAMES,
    extract_stylometry,
    meta_score_features,
)


def test_stylometry_dimension() -> None:
    values = extract_stylometry(
        "Ini adalah contoh teks bahasa Indonesia. "
        "Kalimat kedua memiliki panjang yang berbeda."
    )

    assert (
        len(values)
        == len(
            STYLE_FEATURE_NAMES
        )
    )

    assert (
        np.isfinite(
            np.asarray(
                values,
                dtype=np.float64,
            )
        ).all()
    )


def test_meta_score_features_shape() -> None:
    components = np.asarray(
        [
            [
                0.1,
                0.2,
                0.3,
            ],
            [
                0.7,
                0.8,
                0.9,
            ],
        ],
        dtype=np.float64,
    )

    result = meta_score_features(
        components
    )

    assert result.shape == (
        2,
        11,
    )

    assert (
        np.isfinite(
            result
        ).all()
    )