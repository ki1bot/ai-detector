import numpy as np
import pandas as pd


def source_sample_weights(
    frame: pd.DataFrame,
) -> np.ndarray:
    keys = (
        frame["label"]
        .astype(str)
        + "|"
        + frame["source"]
        .astype(str)
    )

    counts = keys.value_counts()

    median_count = float(
        np.median(
            counts.to_numpy(
                dtype=np.float64
            )
        )
    )

    weights = keys.map(
        lambda key: np.sqrt(
            median_count
            / max(
                float(
                    counts[key]
                ),
                1.0,
            )
        )
    )

    values = np.clip(
        weights.to_numpy(
            dtype=np.float64
        ),
        0.5,
        3.0,
    )

    return (
        values
        / max(
            float(
                values.mean()
            ),
            1e-9,
        )
    )