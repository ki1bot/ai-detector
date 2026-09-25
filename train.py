import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import (
    meta_score_features,
    split_sentences,
    stylometry_matrix,
    tokenize_words,
)

RANDOM_STATE = 42
FORMAT_VERSION = 2


def empty_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": pd.Series(dtype="string"),
            "label": pd.Series(dtype="int64"),
            "source": pd.Series(dtype="string"),
            "word_count": pd.Series(dtype="int64"),
        }
    )


def normalize_text(value: str) -> str:
    text = (
        str(value)
        .replace("\u00a0", " ")
        .replace("\u200b", " ")
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def training_chunks(
    text: str,
    min_words: int = 35,
    target_words: int = 150,
    max_words: int = 220,
) -> list[str]:
    text = normalize_text(text)

    if not text:
        return []

    sentences = split_sentences(text)

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    def flush() -> None:
        nonlocal current
        nonlocal current_words

        if current_words >= min_words:
            chunks.append(
                " ".join(current).strip()
            )

        current = []
        current_words = 0

    for sentence in sentences:
        words = tokenize_words(sentence)

        if not words:
            continue

        if len(words) > max_words:
            flush()

            raw_words = sentence.split()

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
                        tokenize_words(piece)
                    )
                    >= min_words
                ):
                    chunks.append(piece)

            continue

        if (
            current
            and current_words + len(words)
            > max_words
        ):
            flush()

        current.append(sentence)

        current_words += len(words)

        if current_words >= target_words:
            flush()

    flush()

    if (
        not chunks
        and len(
            tokenize_words(text)
        )
        >= 25
    ):
        chunks.append(text)

    return chunks


def clean_dataframe(
    dataframe: pd.DataFrame,
    source_name: str | None = None,
) -> pd.DataFrame:
    if dataframe.empty:
        return empty_training_frame()

    required = {
        "text",
        "label",
    }

    if not required.issubset(
        dataframe.columns
    ):
        raise ValueError(
            "Dataset harus memiliki kolom 'text' dan 'label'."
        )

    frame = dataframe.copy()

    frame["text"] = (
        frame["text"]
        .astype(str)
        .map(normalize_text)
    )

    frame["label"] = pd.to_numeric(
        frame["label"],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=[
            "text",
            "label",
        ]
    )

    frame = frame[
        frame["label"].isin(
            [0, 1]
        )
    ].copy()

    frame["label"] = (
        frame["label"]
        .astype("int64")
    )

    frame["word_count"] = (
        frame["text"]
        .map(
            lambda value: len(
                tokenize_words(value)
            )
        )
        .astype("int64")
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

    frame = frame[
        [
            "text",
            "label",
            "source",
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
        .astype("int64")
    )

    frame["source"] = (
        frame["source"]
        .astype(str)
    )

    frame["word_count"] = (
        frame["word_count"]
        .astype("int64")
    )

    return frame


def load_primary_dataset() -> pd.DataFrame:
    dataset = load_dataset(
        "shouwiku/detectai-indonesian",
        split="train",
    )

    frame = dataset.to_pandas()

    frame["source"] = np.where(
        frame["label"].astype(int)
        == 0,
        "wikipedia_id",
        "gemma4_id",
    )

    return clean_dataframe(frame)


def has_explicit_ai_marker(
    text: str,
) -> bool:
    lowered = text.lower()

    patterns = (
        r"\bchatgpt\b",
        r"\bgemini\b",
        r"\basisten ai\b",
        r"\bsebagai ai\b",
        r"\bmbak ai\b",
        r"\bmodel bahasa\b",
    )

    return any(
        re.search(
            pattern,
            lowered,
        )
        for pattern in patterns
    )


def get_stream(
    repo_id: str,
    config: str | None = None,
    columns: tuple[str, ...] | None = None,
):
    if config is None:
        dataset = load_dataset(
            repo_id,
            streaming=True,
        )

    else:
        dataset = load_dataset(
            repo_id,
            config,
            streaming=True,
        )

    if "train" in dataset:
        stream = dataset["train"]

    else:
        split_names = list(
            dataset.keys()
        )

        if not split_names:
            raise RuntimeError(
                f"Dataset {repo_id} tidak memiliki split yang dapat dipakai."
            )

        stream = dataset[
            split_names[0]
        ]

    if columns:
        available_columns = (
            stream.column_names
            or []
        )

        selected_columns = [
            column
            for column in columns
            if column in available_columns
        ]

        if selected_columns:
            stream = stream.select_columns(
                selected_columns
            )

    return stream


def extract_first_text(
    row: dict,
    fields: tuple[str, ...],
) -> str:
    for field in fields:
        value = row.get(field)

        if (
            isinstance(
                value,
                str,
            )
            and value.strip()
        ):
            return value

    return ""


def sample_human_public(
    limit: int,
) -> tuple[
    list[str],
    str | None,
]:
    candidates = (
        (
            "iqballx/indonesian_news_datasets",
            None,
            (
                "content",
                "text",
                "news_text",
            ),
        ),
        (
            "fahadh4ilyas/indonesian_news_datasets",
            None,
            (
                "content",
                "text",
                "news_text",
            ),
        ),
        (
            "ardimardiana/indonesian-political-news-clean",
            None,
            (
                "news_text",
                "content",
                "text",
            ),
        ),
        (
            "indonesian-nlp/wikipedia-id",
            None,
            (
                "text",
                "content",
            ),
        ),
    )

    errors: list[str] = []

    for (
        repo_id,
        config,
        fields,
    ) in candidates:
        collected: list[str] = []

        try:
            stream = get_stream(
                repo_id,
                config,
                columns=fields,
            )

            stream = stream.shuffle(
                seed=RANDOM_STATE,
                buffer_size=5000,
            )

            for row in stream:
                value = extract_first_text(
                    row,
                    fields,
                )

                if not value:
                    continue

                for chunk in training_chunks(
                    value
                ):
                    if has_explicit_ai_marker(
                        chunk
                    ):
                        continue

                    collected.append(
                        chunk
                    )

                    if (
                        len(collected)
                        >= limit
                    ):
                        return (
                            collected,
                            repo_id,
                        )

            if collected:
                return (
                    collected,
                    repo_id,
                )

            errors.append(
                f"{repo_id}: tidak menemukan teks yang dapat dipakai"
            )

        except Exception as exception:
            errors.append(
                f"{repo_id}: {exception}"
            )

    if errors:
        print(
            "Peringatan sumber human publik:"
        )

        for error in errors:
            print(
                f"- {error}"
            )

    return (
        [],
        None,
    )


def sample_ai_gemini(
    limit: int,
) -> tuple[
    list[str],
    str | None,
]:
    repo_id = (
        "kreasof-ai/percakapan-indo"
    )

    try:
        stream = get_stream(
            repo_id,
            columns=(
                "conversations",
            ),
        )

        stream = stream.shuffle(
            seed=RANDOM_STATE,
            buffer_size=5000,
        )

        collected: list[str] = []

        for row in stream:
            conversations = (
                row.get(
                    "conversations"
                )
                or []
            )

            if not isinstance(
                conversations,
                list,
            ):
                continue

            for message in conversations:
                if not isinstance(
                    message,
                    dict,
                ):
                    continue

                role = str(
                    message.get(
                        "role",
                        "",
                    )
                ).lower()

                if role != "assistant":
                    continue

                content = normalize_text(
                    message.get(
                        "content",
                        "",
                    )
                )

                if not content:
                    continue

                if has_explicit_ai_marker(
                    content
                ):
                    continue

                for chunk in training_chunks(
                    content,
                    min_words=30,
                    target_words=130,
                    max_words=220,
                ):
                    if has_explicit_ai_marker(
                        chunk
                    ):
                        continue

                    collected.append(
                        chunk
                    )

                    if (
                        len(collected)
                        >= limit
                    ):
                        return (
                            collected,
                            repo_id,
                        )

        if collected:
            return (
                collected,
                repo_id,
            )

        return (
            [],
            None,
        )

    except Exception as exception:
        print(
            "Peringatan sumber AI publik "
            f"{repo_id}: {exception}"
        )

        return (
            [],
            None,
        )


def load_public_augmentation(
    limit_per_class: int,
) -> pd.DataFrame:
    if limit_per_class <= 0:
        return empty_training_frame()

    human_texts, human_source = (
        sample_human_public(
            limit_per_class
        )
    )

    ai_texts, ai_source = (
        sample_ai_gemini(
            limit_per_class
        )
    )

    usable = min(
        len(human_texts),
        len(ai_texts),
        limit_per_class,
    )

    minimum = min(
        200,
        max(
            50,
            limit_per_class // 4,
        ),
    )

    if usable < minimum:
        print(
            "Peringatan: data augmentasi publik "
            "tidak cukup seimbang. "
            f"Human={len(human_texts)}, "
            f"AI={len(ai_texts)}, "
            f"minimum={minimum}."
        )

        return empty_training_frame()

    human_texts = (
        human_texts[
            :usable
        ]
    )

    ai_texts = (
        ai_texts[
            :usable
        ]
    )

    human = pd.DataFrame(
        {
            "text": human_texts,
            "label": np.zeros(
                usable,
                dtype=np.int64,
            ),
            "source": (
                human_source
                or "public_human_id"
            ),
        }
    )

    ai = pd.DataFrame(
        {
            "text": ai_texts,
            "label": np.ones(
                usable,
                dtype=np.int64,
            ),
            "source": (
                ai_source
                or "public_ai_id"
            ),
        }
    )

    print(
        "Augmentasi publik digunakan: "
        f"{human_source or 'human'}={usable}, "
        f"{ai_source or 'ai'}={usable}"
    )

    combined = pd.concat(
        [
            human,
            ai,
        ],
        ignore_index=True,
    )

    return clean_dataframe(
        combined
    )


def load_extra_csv(
    paths: list[str],
) -> pd.DataFrame:
    frames: list[
        pd.DataFrame
    ] = []

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

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    return clean_dataframe(
        combined
    )


def build_word_model() -> Pipeline:
    return Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="word",
                    ngram_range=(1, 3),
                    min_df=2,
                    max_df=0.995,
                    max_features=80000,
                    sublinear_tf=True,
                    strip_accents=None,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=2.0,
                    max_iter=4000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_char_model() -> Pipeline:
    return Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 6),
                    min_df=2,
                    max_df=0.999,
                    max_features=100000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=2.0,
                    max_iter=4000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_style_model() -> Pipeline:
    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=0.8,
                    max_iter=4000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def component_probabilities(
    models: dict,
    texts: list[str],
) -> np.ndarray:
    word_scores = (
        models["word"]
        .predict_proba(
            texts
        )[:, 1]
    )

    char_scores = (
        models["char"]
        .predict_proba(
            texts
        )[:, 1]
    )

    style_scores = (
        models["style"]
        .predict_proba(
            stylometry_matrix(
                texts
            )
        )[:, 1]
    )

    return np.column_stack(
        [
            word_scores,
            char_scores,
            style_scores,
        ]
    )


def select_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[
    float,
    float,
]:
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
        return (
            0.22,
            0.78,
        )

    ai_threshold = float(
        np.quantile(
            human_scores,
            0.98,
            method="higher",
        )
    )

    human_threshold = float(
        np.quantile(
            ai_scores,
            0.02,
            method="lower",
        )
    )

    ai_threshold = min(
        max(
            ai_threshold,
            0.72,
        ),
        0.95,
    )

    human_threshold = max(
        min(
            human_threshold,
            0.28,
        ),
        0.05,
    )

    if (
        human_threshold
        >= ai_threshold - 0.20
    ):
        human_threshold = 0.22
        ai_threshold = 0.78

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

    return {
        "accuracy_at_0_5": float(
            accuracy_score(
                y_true,
                binary,
            )
        ),
        "balanced_accuracy_at_0_5": float(
            balanced_accuracy_score(
                y_true,
                binary,
            )
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
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "brier": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
        "confusion_matrix_at_0_5": (
            confusion_matrix(
                y_true,
                binary,
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
    }


def source_metrics(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
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

    sources = sorted(
        frame["source"]
        .astype(str)
        .unique()
    )

    for source in sources:
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

        predicted = (
            scores
            >= 0.5
        ).astype(
            np.int64
        )

        item = {
            "samples": int(
                len(labels)
            ),
            "accuracy_at_0_5": float(
                accuracy_score(
                    labels,
                    predicted,
                )
            ),
            "mean_ai_score": float(
                np.mean(
                    scores
                )
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
            item["roc_auc"] = float(
                roc_auc_score(
                    labels,
                    scores,
                )
            )

        result[
            source
        ] = item

    return result


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default="models/detector.joblib",
    )

    parser.add_argument(
        "--public-augmentation",
        type=int,
        default=1200,
    )

    parser.add_argument(
        "--extra-csv",
        action="append",
        default=[],
    )

    arguments = parser.parse_args()

    print(
        "Memuat dataset utama..."
    )

    primary = load_primary_dataset()

    print(
        "Memuat data tambahan publik..."
    )

    augmentation = (
        load_public_augmentation(
            arguments.public_augmentation
        )
    )

    extras = load_extra_csv(
        arguments.extra_csv
    )

    frames = [
        frame
        for frame in (
            primary,
            augmentation,
            extras,
        )
        if not frame.empty
    ]

    dataframe = clean_dataframe(
        pd.concat(
            frames,
            ignore_index=True,
        )
    )

    conflicts = (
        dataframe
        .groupby(
            "text"
        )["label"]
        .nunique()
    )

    conflicting_texts = set(
        conflicts[
            conflicts > 1
        ].index
    )

    if conflicting_texts:
        dataframe = dataframe[
            ~dataframe[
                "text"
            ].isin(
                conflicting_texts
            )
        ].copy()

    dataframe = (
        dataframe
        .drop_duplicates(
            subset=[
                "text",
                "label",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    dataframe["label"] = (
        pd.to_numeric(
            dataframe["label"],
            errors="raise",
        )
        .astype(
            "int64"
        )
    )

    dataframe[
        "word_count"
    ] = (
        pd.to_numeric(
            dataframe[
                "word_count"
            ],
            errors="raise",
        )
        .astype(
            "int64"
        )
    )

    dataframe["text"] = (
        dataframe["text"]
        .astype(str)
    )

    dataframe["source"] = (
        dataframe["source"]
        .astype(str)
    )

    if len(dataframe) < 1000:
        raise ValueError(
            "Dataset terlalu kecil untuk versi detector ini."
        )

    class_counts = (
        dataframe["label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    if min(
        class_counts.get(
            0,
            0,
        ),
        class_counts.get(
            1,
            0,
        ),
    ) < 400:
        raise ValueError(
            "Masing-masing kelas harus memiliki setidaknya 400 sampel."
        )

    print(
        f"Total data: {len(dataframe)}"
    )

    print(
        f"Distribusi label: {class_counts}"
    )

    print(
        "Tipe label: "
        f"{dataframe['label'].dtype}"
    )

    print(
        "Sumber data:"
    )

    source_counts = (
        dataframe["source"]
        .value_counts()
        .to_dict()
    )

    for (
        source,
        count,
    ) in source_counts.items():
        print(
            f"- {source}: {count}"
        )

    train_frame, temp_frame = (
        train_test_split(
            dataframe,
            test_size=0.40,
            stratify=(
                dataframe["label"]
                .to_numpy(
                    dtype=np.int64
                )
            ),
            random_state=RANDOM_STATE,
        )
    )

    (
        calibration_frame,
        test_frame,
    ) = train_test_split(
        temp_frame,
        test_size=0.50,
        stratify=(
            temp_frame["label"]
            .to_numpy(
                dtype=np.int64
            )
        ),
        random_state=RANDOM_STATE,
    )

    train_texts = (
        train_frame["text"]
        .astype(str)
        .tolist()
    )

    calibration_texts = (
        calibration_frame[
            "text"
        ]
        .astype(str)
        .tolist()
    )

    test_texts = (
        test_frame["text"]
        .astype(str)
        .tolist()
    )

    y_train = (
        train_frame["label"]
        .to_numpy(
            dtype=np.int64
        )
    )

    y_calibration = (
        calibration_frame[
            "label"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    y_test = (
        test_frame["label"]
        .to_numpy(
            dtype=np.int64
        )
    )

    models = {
        "word": (
            build_word_model()
        ),
        "char": (
            build_char_model()
        ),
        "style": (
            build_style_model()
        ),
    }

    print(
        "Melatih model pola kata..."
    )

    models["word"].fit(
        train_texts,
        y_train,
    )

    print(
        "Melatih model pola karakter..."
    )

    models["char"].fit(
        train_texts,
        y_train,
    )

    print(
        "Melatih model gaya penulisan..."
    )

    models["style"].fit(
        stylometry_matrix(
            train_texts
        ),
        y_train,
    )

    calibration_components = (
        component_probabilities(
            models,
            calibration_texts,
        )
    )

    meta_model = (
        LogisticRegression(
            C=1.0,
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        )
    )

    meta_model.fit(
        meta_score_features(
            calibration_components
        ),
        y_calibration,
    )

    calibration_probabilities = (
        meta_model
        .predict_proba(
            meta_score_features(
                calibration_components
            )
        )[:, 1]
    )

    (
        human_threshold,
        ai_threshold,
    ) = select_thresholds(
        y_calibration,
        calibration_probabilities,
    )

    test_components = (
        component_probabilities(
            models,
            test_texts,
        )
    )

    test_probabilities = (
        meta_model
        .predict_proba(
            meta_score_features(
                test_components
            )
        )[:, 1]
    )

    metrics = evaluate(
        y_test,
        test_probabilities,
        human_threshold=human_threshold,
        ai_threshold=ai_threshold,
    )

    all_labels = (
        dataframe["label"]
        .to_numpy(
            dtype=np.int64
        )
    )

    metadata = {
        "format_version": (
            FORMAT_VERSION
        ),
        "model_name": (
            "DeteksiAI Hybrid ID"
        ),
        "model_version": "2.1",
        "trained_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "decision_thresholds": {
            "human": (
                human_threshold
            ),
            "ai": (
                ai_threshold
            ),
        },
        "samples": {
            "total": int(
                len(dataframe)
            ),
            "train": int(
                len(train_frame)
            ),
            "calibration": int(
                len(
                    calibration_frame
                )
            ),
            "test": int(
                len(test_frame)
            ),
            "human": int(
                np.sum(
                    all_labels == 0
                )
            ),
            "ai": int(
                np.sum(
                    all_labels == 1
                )
            ),
        },
        "sources": {
            str(key): int(value)
            for (
                key,
                value,
            ) in source_counts.items()
        },
        "evaluation": metrics,
        "test_source_breakdown": (
            source_metrics(
                test_frame,
                test_probabilities,
            )
        ),
    }

    output_path = Path(
        arguments.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "format_version": (
                FORMAT_VERSION
            ),
            "models": models,
            "meta_model": (
                meta_model
            ),
            "metadata": metadata,
        },
        output_path,
    )

    print(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        f"Model disimpan ke: {output_path}"
    )


if __name__ == "__main__":
    main()