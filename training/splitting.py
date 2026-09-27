import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from training.common import RANDOM_STATE


def _distribution(
    series: pd.Series,
) -> dict[str, float]:
    counts = (
        series
        .astype(str)
        .value_counts(
            normalize=True
        )
    )

    return {
        str(key): float(value)
        for key, value in counts.items()
    }


def _distribution_distance(
    left: dict[str, float],
    right: dict[str, float],
) -> float:
    keys = set(left) | set(right)

    return sum(
        abs(
            left.get(
                key,
                0.0,
            )
            - right.get(
                key,
                0.0,
            )
        )
        for key in keys
    )


def best_group_split(
    frame: pd.DataFrame,
    test_size: float,
    random_state: int,
    attempts: int = 64,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    if (
        frame["group_id"]
        .nunique()
        < 8
    ):
        raise ValueError(
            "Jumlah group_id terlalu sedikit untuk pemisahan data yang aman."
        )

    target_label_rate = float(
        frame["label"].mean()
    )

    target_source_distribution = (
        _distribution(
            frame["source"]
        )
    )

    best: tuple[
        float,
        np.ndarray,
        np.ndarray,
    ] | None = None

    for attempt in range(
        attempts
    ):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=(
                random_state
                + attempt
            ),
        )

        train_index, test_index = next(
            splitter.split(
                frame,
                y=frame["label"],
                groups=frame["group_id"],
            )
        )

        train = frame.iloc[
            train_index
        ]

        test = frame.iloc[
            test_index
        ]

        if (
            train["label"].nunique()
            < 2
            or test["label"].nunique()
            < 2
        ):
            continue

        label_penalty = abs(
            float(
                train["label"].mean()
            )
            - target_label_rate
        )

        label_penalty += abs(
            float(
                test["label"].mean()
            )
            - target_label_rate
        )

        size_penalty = abs(
            len(test)
            / len(frame)
            - test_size
        )

        source_penalty = (
            _distribution_distance(
                _distribution(
                    test["source"]
                ),
                target_source_distribution,
            )
        )

        score = (
            5.0 * label_penalty
            + 2.0 * size_penalty
            + 0.35 * source_penalty
        )

        if (
            best is None
            or score < best[0]
        ):
            best = (
                score,
                train_index,
                test_index,
            )

    if best is None:
        raise ValueError(
            "Tidak dapat membuat group-aware split yang memiliki kedua kelas."
        )

    train = (
        frame
        .iloc[
            best[1]
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    test = (
        frame
        .iloc[
            best[2]
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    overlap = (
        set(
            train["group_id"]
        )
        & set(
            test["group_id"]
        )
    )

    if overlap:
        raise RuntimeError(
            "Group leakage terdeteksi pada pemisahan dataset."
        )

    return train, test


def split_development_threshold_test(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    development, holdout = (
        best_group_split(
            frame,
            test_size=0.25,
            random_state=RANDOM_STATE,
        )
    )

    threshold, test = (
        best_group_split(
            holdout,
            test_size=0.60,
            random_state=(
                RANDOM_STATE
                + 1000
            ),
        )
    )

    development_groups = set(
        development["group_id"]
    )

    threshold_groups = set(
        threshold["group_id"]
    )

    test_groups = set(
        test["group_id"]
    )

    if (
        development_groups
        & threshold_groups
    ):
        raise RuntimeError(
            "Group leakage antara development dan threshold set."
        )

    if (
        development_groups
        & test_groups
    ):
        raise RuntimeError(
            "Group leakage antara development dan test set."
        )

    if (
        threshold_groups
        & test_groups
    ):
        raise RuntimeError(
            "Group leakage antara threshold dan test set."
        )

    return (
        development,
        threshold,
        test,
    )