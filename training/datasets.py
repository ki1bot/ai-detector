from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

from training.common import clean_dataframe, empty_training_frame, stable_hash


def load_primary_dataset() -> pd.DataFrame:
    dataset = load_dataset(
        "shouwiku/detectai-indonesian",
        split="train",
    )

    frame = dataset.to_pandas()

    labels = pd.to_numeric(
        frame["label"],
        errors="raise",
    ).astype("int64")

    if "model" in frame.columns:
        model_names = (
            frame["model"]
            .fillna("unknown")
            .astype(str)
        )

    else:
        model_names = pd.Series(
            "unknown",
            index=frame.index,
            dtype="string",
        )

    frame["source"] = np.where(
        labels == 0,
        "detectai:wikipedia",
        "detectai:" + model_names,
    )

    topic_values = (
        frame["topic"]
        .fillna("")
        .astype(str)
        if "topic" in frame.columns
        else pd.Series(
            "",
            index=frame.index,
            dtype="string",
        )
    )

    title_values = (
        frame["page_title"]
        .fillna("")
        .astype(str)
        if "page_title" in frame.columns
        else pd.Series(
            "",
            index=frame.index,
            dtype="string",
        )
    )

    group_ids: list[str] = []

    for index, (
        topic,
        title,
        text,
    ) in enumerate(
        zip(
            topic_values,
            title_values,
            frame["text"].astype(str),
        )
    ):
        seed = (
            topic.strip()
            or title.strip()
            or (
                f"row-{index}:"
                f"{text[:120]}"
            )
        )

        group_ids.append(
            "detectai-topic:"
            f"{stable_hash(seed)}"
        )

    frame["group_id"] = group_ids

    return clean_dataframe(frame)


def load_extra_csv(
    paths: list[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for path_string in paths:
        path = Path(
            path_string
        )

        frame = pd.read_csv(
            path
        )

        if "source" not in frame.columns:
            frame["source"] = (
                f"csv:{path.stem}"
            )

        frames.append(
            clean_dataframe(
                frame
            )
        )

    if not frames:
        return empty_training_frame()

    return clean_dataframe(
        pd.concat(
            frames,
            ignore_index=True,
        )
    )


def load_external_test_csv(
    paths: list[str],
) -> pd.DataFrame:
    frame = load_extra_csv(
        paths
    )

    return (
        frame.reset_index(
            drop=True
        )
        if not frame.empty
        else frame
    )