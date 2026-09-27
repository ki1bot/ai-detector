import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    target_false_positive_rate: float = 0.02,
    target_ai_missed_as_human_rate: float = 0.02,
) -> tuple[float, float]:
    human_scores = probabilities[
        y_true == 0
    ]

    ai_scores = probabilities[
        y_true == 1
    ]

    if (
        len(human_scores) < 20
        or len(ai_scores) < 20
    ):
        return 0.20, 0.80

    ai_grid = np.linspace(
        0.55,
        0.98,
        431,
    )

    ai_valid = [
        threshold
        for threshold in ai_grid
        if float(
            np.mean(
                human_scores
                >= threshold
            )
        )
        <= target_false_positive_rate
    ]

    ai_threshold = (
        float(
            min(ai_valid)
        )
        if ai_valid
        else float(
            np.quantile(
                human_scores,
                0.99,
            )
        )
    )

    human_grid = np.linspace(
        0.02,
        0.45,
        431,
    )

    human_valid = [
        threshold
        for threshold in human_grid
        if float(
            np.mean(
                ai_scores
                <= threshold
            )
        )
        <= target_ai_missed_as_human_rate
    ]

    human_threshold = (
        float(
            max(human_valid)
        )
        if human_valid
        else float(
            np.quantile(
                ai_scores,
                0.01,
            )
        )
    )

    human_threshold = float(
        np.clip(
            human_threshold,
            0.03,
            0.45,
        )
    )

    ai_threshold = float(
        np.clip(
            ai_threshold,
            0.55,
            0.97,
        )
    )

    if (
        human_threshold
        >= ai_threshold
        - 0.18
    ):
        midpoint = (
            human_threshold
            + ai_threshold
        ) / 2

        human_threshold = max(
            0.05,
            midpoint - 0.10,
        )

        ai_threshold = min(
            0.95,
            midpoint + 0.10,
        )

    return (
        round(
            human_threshold,
            4,
        ),
        round(
            ai_threshold,
            4,
        ),
    )


def evaluate(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    human_threshold: float,
    ai_threshold: float,
) -> dict:
    binary = (
        probabilities
        >= 0.5
    ).astype(
        np.int64
    )

    selective = np.full(
        len(probabilities),
        -1,
        dtype=np.int64,
    )

    selective[
        probabilities
        <= human_threshold
    ] = 0

    selective[
        probabilities
        >= ai_threshold
    ] = 1

    selected_mask = (
        selective != -1
    )

    human_total = max(
        int(
            np.sum(
                y_true == 0
            )
        ),
        1,
    )

    ai_total = max(
        int(
            np.sum(
                y_true == 1
            )
        ),
        1,
    )

    selective_accuracy = None
    selective_precision_ai = None
    selective_recall_ai = None

    if np.any(
        selected_mask
    ):
        selective_accuracy = float(
            accuracy_score(
                y_true[
                    selected_mask
                ],
                selective[
                    selected_mask
                ],
            )
        )

        selective_precision_ai = float(
            precision_score(
                y_true[
                    selected_mask
                ],
                selective[
                    selected_mask
                ],
                zero_division=0,
            )
        )

        selective_recall_ai = float(
            recall_score(
                y_true[
                    selected_mask
                ],
                selective[
                    selected_mask
                ],
                zero_division=0,
            )
        )

    has_both_classes = (
        len(
            np.unique(
                y_true
            )
        )
        == 2
    )

    result = {
        "accuracy_at_0_5": float(
            accuracy_score(
                y_true,
                binary,
            )
        ),
        "balanced_accuracy_at_0_5": (
            float(
                balanced_accuracy_score(
                    y_true,
                    binary,
                )
            )
            if has_both_classes
            else None
        ),
        "precision_ai_at_0_5": float(
            precision_score(
                y_true,
                binary,
                zero_division=0,
            )
        ),
        "recall_ai_at_0_5": float(
            recall_score(
                y_true,
                binary,
                zero_division=0,
            )
        ),
        "f1_at_0_5": float(
            f1_score(
                y_true,
                binary,
                zero_division=0,
            )
        ),
        "average_precision": (
            float(
                average_precision_score(
                    y_true,
                    probabilities,
                )
            )
            if has_both_classes
            else None
        ),
        "brier": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                np.column_stack(
                    [
                        1 - probabilities,
                        probabilities,
                    ]
                ),
                labels=[
                    0,
                    1,
                ],
            )
        ),
        "confusion_matrix_at_0_5": (
            confusion_matrix(
                y_true,
                binary,
                labels=[
                    0,
                    1,
                ],
            ).tolist()
        ),
        "selective_coverage": float(
            np.mean(
                selected_mask
            )
        ),
        "selective_accuracy": (
            selective_accuracy
        ),
        "selective_precision_ai": (
            selective_precision_ai
        ),
        "selective_recall_ai": (
            selective_recall_ai
        ),
        "ai_false_positive_rate": float(
            np.sum(
                (
                    selective == 1
                )
                & (
                    y_true == 0
                )
            )
            / human_total
        ),
        "ai_detection_rate": float(
            np.sum(
                (
                    selective == 1
                )
                & (
                    y_true == 1
                )
            )
            / ai_total
        ),
        "ai_missed_as_human_rate": float(
            np.sum(
                (
                    selective == 0
                )
                & (
                    y_true == 1
                )
            )
            / ai_total
        ),
        "human_detection_rate": float(
            np.sum(
                (
                    selective == 0
                )
                & (
                    y_true == 0
                )
            )
            / human_total
        ),
    }

    if has_both_classes:
        result["roc_auc"] = float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        )

    else:
        result["roc_auc"] = None

    return result


def source_metrics(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    human_threshold: float,
    ai_threshold: float,
) -> dict:
    result: dict[
        str,
        dict,
    ] = {}

    source_values = (
        frame["source"]
        .astype(str)
        .to_numpy()
    )

    for source in sorted(
        frame["source"]
        .astype(str)
        .unique()
    ):
        mask = (
            source_values
            == source
        )

        labels = (
            frame.loc[
                mask,
                "label",
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        scores = probabilities[
            mask
        ]

        if len(labels) == 0:
            continue

        metrics = evaluate(
            labels,
            scores,
            human_threshold,
            ai_threshold,
        )

        item = {
            "samples": int(
                len(labels)
            ),
            "accuracy_at_0_5": (
                metrics[
                    "accuracy_at_0_5"
                ]
            ),
            "balanced_accuracy_at_0_5": (
                metrics[
                    "balanced_accuracy_at_0_5"
                ]
            ),
            "mean_ai_score": float(
                np.mean(
                    scores
                )
            ),
            "selective_coverage": (
                metrics[
                    "selective_coverage"
                ]
            ),
            "selective_accuracy": (
                metrics[
                    "selective_accuracy"
                ]
            ),
        }

        if (
            len(
                np.unique(
                    labels
                )
            )
            == 2
        ):
            item["roc_auc"] = (
                metrics[
                    "roc_auc"
                ]
            )

        result[
            source
        ] = item

    return result