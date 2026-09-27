import hashlib
import re

import pandas as pd

from detector.text import normalize_text, split_sentences, tokenize_words

RANDOM_STATE = 42


def stable_hash(value: str) -> str:
    normalized = (
        normalize_text(value)
        .lower()
        .encode(
            "utf-8",
            errors="ignore",
        )
    )

    return hashlib.sha1(
        normalized
    ).hexdigest()


def canonical_fingerprint(text: str) -> str:
    value = normalize_text(
        text
    ).lower()

    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return stable_hash(value)


def empty_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": pd.Series(
                dtype="string"
            ),
            "label": pd.Series(
                dtype="int64"
            ),
            "source": pd.Series(
                dtype="string"
            ),
            "group_id": pd.Series(
                dtype="string"
            ),
            "word_count": pd.Series(
                dtype="int64"
            ),
        }
    )


def training_chunks(
    text: str,
    min_words: int = 35,
    target_words: int = 150,
    max_words: int = 220,
) -> list[str]:
    value = normalize_text(text)

    if not value:
        return []

    sentences = split_sentences(
        value
    )

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    def flush() -> None:
        nonlocal current
        nonlocal current_words

        if current_words >= min_words:
            chunks.append(
                " ".join(
                    current
                ).strip()
            )

        current = []
        current_words = 0

    for sentence in sentences:
        words = tokenize_words(
            sentence
        )

        if not words:
            continue

        if len(words) > max_words:
            flush()

            raw_words = (
                sentence.split()
            )

            for start in range(
                0,
                len(raw_words),
                target_words,
            ):
                piece = " ".join(
                    raw_words[
                        start:
                        start + target_words
                    ]
                ).strip()

                if (
                    len(
                        tokenize_words(
                            piece
                        )
                    )
                    >= min_words
                ):
                    chunks.append(
                        piece
                    )

            continue

        if (
            current
            and current_words
            + len(words)
            > max_words
        ):
            flush()

        current.append(
            sentence
        )

        current_words += len(
            words
        )

        if (
            current_words
            >= target_words
        ):
            flush()

    flush()

    if (
        not chunks
        and len(
            tokenize_words(
                value
            )
        )
        >= 25
    ):
        chunks.append(
            value
        )

    return chunks


def clean_dataframe(
    dataframe: pd.DataFrame,
    source_name: str | None = None,
) -> pd.DataFrame:
    if dataframe.empty:
        return empty_training_frame()

    required_columns = {
        "text",
        "label",
    }

    if not required_columns.issubset(
        dataframe.columns
    ):
        raise ValueError(
            "Dataset harus memiliki kolom 'text' dan 'label'."
        )

    frame = dataframe.copy()

    frame["text"] = (
        frame["text"]
        .astype(str)
        .map(
            normalize_text
        )
    )

    frame["label"] = (
        pd.to_numeric(
            frame["label"],
            errors="coerce",
        )
    )

    frame = frame.dropna(
        subset=[
            "text",
            "label",
        ]
    )

    frame = frame[
        frame["label"].isin(
            [
                0,
                1,
            ]
        )
    ].copy()

    frame["label"] = (
        frame["label"]
        .astype(
            "int64"
        )
    )

    frame["word_count"] = (
        frame["text"]
        .map(
            lambda value: len(
                tokenize_words(
                    value
                )
            )
        )
        .astype(
            "int64"
        )
    )

    frame = frame[
        (
            frame["word_count"]
            >= 25
        )
        & (
            frame["word_count"]
            <= 5000
        )
    ].copy()

    if "source" not in frame.columns:
        frame["source"] = (
            source_name
            or "external"
        )

    else:
        frame["source"] = (
            frame["source"]
            .fillna(
                source_name
                or "external"
            )
            .astype(str)
        )

    if "group_id" not in frame.columns:
        frame["group_id"] = (
            frame["text"]
            .map(
                lambda value: (
                    f"text:"
                    f"{stable_hash(value)}"
                )
            )
        )

    else:
        frame["group_id"] = (
            frame["group_id"]
            .fillna("")
            .astype(str)
        )

        missing_group = (
            frame["group_id"]
            .str.strip()
            .eq("")
        )

        frame.loc[
            missing_group,
            "group_id",
        ] = (
            frame.loc[
                missing_group,
                "text",
            ]
            .map(
                lambda value: (
                    f"text:"
                    f"{stable_hash(value)}"
                )
            )
        )

    frame = frame[
        [
            "text",
            "label",
            "source",
            "group_id",
            "word_count",
        ]
    ].reset_index(
        drop=True
    )

    frame["text"] = (
        frame["text"]
        .astype(str)
    )

    frame["label"] = (
        frame["label"]
        .astype(
            "int64"
        )
    )

    frame["source"] = (
        frame["source"]
        .astype(str)
    )

    frame["group_id"] = (
        frame["group_id"]
        .astype(str)
    )

    frame["word_count"] = (
        frame["word_count"]
        .astype(
            "int64"
        )
    )

    return frame


def remove_conflicts_and_duplicates(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    frame = clean_dataframe(
        dataframe
    )

    frame["fingerprint"] = (
        frame["text"]
        .map(
            canonical_fingerprint
        )
    )

    conflicts = (
        frame
        .groupby(
            "fingerprint"
        )["label"]
        .nunique()
    )

    conflicting = set(
        conflicts[
            conflicts > 1
        ].index
    )

    if conflicting:
        frame = frame[
            ~frame[
                "fingerprint"
            ].isin(
                conflicting
            )
        ].copy()

    frame = (
        frame
        .drop_duplicates(
            subset=[
                "fingerprint",
                "label",
            ],
            keep="first",
        )
    )

    frame = (
        frame
        .drop(
            columns=[
                "fingerprint"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return clean_dataframe(
        frame
    )