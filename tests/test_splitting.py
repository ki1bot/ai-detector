import pandas as pd

from training.splitting import split_development_threshold_test


def build_frame() -> pd.DataFrame:
    rows = []

    for group in range(80):
        for label in (
            0,
            1,
        ):
            rows.append(
                {
                    "text": (
                        f"teks group {group} "
                        f"label {label} "
                        + "kata " * 30
                    ),
                    "label": label,
                    "source": "synthetic",
                    "group_id": (
                        f"group-{group}"
                    ),
                    "word_count": 35,
                }
            )

    return pd.DataFrame(
        rows
    )


def test_group_aware_split_has_no_overlap() -> None:
    (
        development,
        threshold,
        test,
    ) = (
        split_development_threshold_test(
            build_frame()
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

    assert not (
        development_groups
        & threshold_groups
    )

    assert not (
        development_groups
        & test_groups
    )

    assert not (
        threshold_groups
        & test_groups
    )

    assert (
        development["label"]
        .nunique()
        == 2
    )

    assert (
        threshold["label"]
        .nunique()
        == 2
    )

    assert (
        test["label"]
        .nunique()
        == 2
    )