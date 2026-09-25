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

    chunks = []
    current = []
    current_words = 0

    def flush():
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
        sentence_words = tokenize_words(
            sentence
        )

        if not sentence_words:
            continue

        if len(sentence_words) > max_words:
            flush()

            raw_words = sentence.split()

            for start in range(
                0,
                len(raw_words),
                target_words,
            ):
                piece = " ".join(
                    raw_words[
                        start:start + target_words
                    ]
                ).strip()

                if (
                    len(
                        tokenize_words(piece)
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
            + len(sentence_words)
            > max_words
        ):
            flush()

        current.append(
            sentence
        )

        current_words += len(
            sentence_words
        )

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
        chunks.append(
            text
        )

    return chunks


def clean_dataframe(
    dataframe: pd.DataFrame,
    source_name: str | None = None,
) -> pd.DataFrame:
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

    frame["label"] = (
        frame["label"]
        .astype(int)
    )

    frame = frame[
        frame["label"].isin(
            [0, 1]
        )
    ]

    frame["word_count"] = (
        frame["text"]
        .map(
            lambda value: len(
                tokenize_words(value)
            )
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
    ]

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
    ]

    return frame.reset_index(
        drop=True
    )


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

    return clean_dataframe(
        frame
    )


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


def sample_human_mc4(
    limit: int,
) -> list[str]:
    stream = None
    last_error = None

    for config in (
        "tiny",
        "full",
    ):
        try:
            stream = load_dataset(
                "indonesian-nlp/mc4-id",
                config,
                split="train",
                streaming=True,
            )

            break

        except Exception as exception:
            last_error = exception

    if stream is None:
        raise RuntimeError(
            "Tidak dapat memuat corpus manusia "
            f"mC4-ID: {last_error}"
        )

    stream = stream.shuffle(
        seed=RANDOM_STATE,
        buffer_size=10000,
    )

    collected = []

    for row in stream:
        value = row.get(
            "text",
            "",
        )

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
                return collected

    return collected


def sample_ai_gemini(
    limit: int,
) -> list[str]:
    stream = load_dataset(
        "kreasof-ai/percakapan-indo",
        split="train",
        streaming=True,
    )

    stream = stream.shuffle(
        seed=RANDOM_STATE,
        buffer_size=10000,
    )

    collected = []

    for row in stream:
        conversations = (
            row.get(
                "conversations"
            )
            or []
        )

        for message in conversations:
            if not isinstance(
                message,
                dict,
            ):
                continue

            if (
                str(
                    message.get(
                        "role",
                        "",
                    )
                ).lower()
                != "assistant"
            ):
                continue

            content = normalize_text(
                message.get(
                    "content",
                    "",
                )
            )

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
                collected.append(
                    chunk
                )

                if (
                    len(collected)
                    >= limit
                ):
                    return collected

    return collected


def load_public_augmentation(
    limit_per_class: int,
) -> pd.DataFrame:
    if limit_per_class <= 0:
        return pd.DataFrame(
            columns=[
                "text",
                "label",
                "source",
                "word_count",
            ]
        )

    try:
        human_texts = sample_human_mc4(
            limit_per_class
        )

        ai_texts = sample_ai_gemini(
            limit_per_class
        )

    except Exception as exception:
        print(
            "Peringatan: augmentasi publik "
            "dilewati karena gagal dimuat: "
            f"{exception}"
        )

        return pd.DataFrame(
            columns=[
                "text",
                "label",
                "source",
                "word_count",
            ]
        )

    minimum = min(
        200,
        max(
            50,
            limit_per_class // 4,
        ),
    )

    if (
        len(human_texts)
        < minimum
        or len(ai_texts)
        < minimum
    ):
        print(
            "Peringatan: augmentasi publik "
            "tidak cukup seimbang dan tidak digunakan."
        )

        return pd.DataFrame(
            columns=[
                "text",
                "label",
                "source",
                "word_count",
            ]
        )

    human = pd.DataFrame(
        {
            "text": human_texts,
            "label": 0,
            "source": "mc4_id_human",
        }
    )

    ai = pd.DataFrame(
        {
            "text": ai_texts,
            "label": 1,
            "source": "gemini_flash_id",
        }
    )

    return clean_dataframe(
        pd.concat(
            [
                human,
                ai,
            ],
            ignore_index=True,
        )
    )


def load_extra_csv(
    paths: list[str],
) -> pd.DataFrame:
    frames = []

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
        return pd.DataFrame(
            columns=[
                "text",
                "label",
                "source",
                "word_count",
            ]
        )

    return pd.concat(
        frames,
        ignore_index=True,
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
    texts: pd.Series | list[str],
) -> np.ndarray:
    text_values = list(
        texts
    )

    word_scores = (
        models["word"]
        .predict_proba(
            text_values
        )[:, 1]
    )

    char_scores = (
        models["char"]
        .predict_proba(
            text_values
        )[:, 1]
    )

    style_scores = (
        models["style"]
        .predict_proba(
            stylometry_matrix(
                text_values
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
        return 0.25, 0.75

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
    ).astype(int)

    selective = np.full(
        len(probabilities),
        -1,
        dtype=int,
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

    ai_false_positive_rate = float(
        np.sum(
            (selective == 1)
            & (y_true == 0)
        )
        / human_total
    )

    ai_detection_rate = float(
        np.sum(
            (selective == 1)
            & (y_true == 1)
        )
        / ai_total
    )

    human_false_negative_rate = float(
        np.sum(
            (selective == 0)
            & (y_true == 1)
        )
        / ai_total
    )

    selected_accuracy = None

    if np.any(
        selected_mask
    ):
        selected_accuracy = float(
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
            selected_accuracy
        ),
        "ai_false_positive_rate": (
            ai_false_positive_rate
        ),
        "ai_detection_rate": (
            ai_detection_rate
        ),
        "ai_missed_as_human_rate": (
            human_false_negative_rate
        ),
    }


def source_metrics(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
) -> dict:
    result = {}

    for source in sorted(
        frame["source"].unique()
    ):
        mask = (
            frame["source"]
            .to_numpy()
            == source
        )

        labels = (
            frame.loc[
                mask,
                "label",
            ]
            .to_numpy(
                dtype=int
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
        ).astype(int)

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

        result[source] = item

    return result


def main():
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

    dataframe = pd.concat(
        [
            primary,
            augmentation,
            extras,
        ],
        ignore_index=True,
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
            ~dataframe["text"].isin(
                conflicting_texts
            )
        ]

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

    if len(dataframe) < 1000:
        raise ValueError(
            "Dataset terlalu kecil untuk versi detector ini."
        )

    class_counts = (
        dataframe["label"]
        .value_counts()
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

    train_frame, temp_frame = (
        train_test_split(
            dataframe,
            test_size=0.40,
            stratify=dataframe[
                "label"
            ],
            random_state=RANDOM_STATE,
        )
    )

    (
        calibration_frame,
        test_frame,
    ) = train_test_split(
        temp_frame,
        test_size=0.50,
        stratify=temp_frame[
            "label"
        ],
        random_state=RANDOM_STATE,
    )

    models = {
        "word": build_word_model(),
        "char": build_char_model(),
        "style": build_style_model(),
    }

    print(
        "Melatih model pola kata..."
    )

    models["word"].fit(
        train_frame["text"],
        train_frame["label"],
    )

    print(
        "Melatih model pola karakter..."
    )

    models["char"].fit(
        train_frame["text"],
        train_frame["label"],
    )

    print(
        "Melatih model gaya penulisan..."
    )

    models["style"].fit(
        stylometry_matrix(
            train_frame["text"]
        ),
        train_frame["label"],
    )

    calibration_components = (
        component_probabilities(
            models,
            calibration_frame[
                "text"
            ],
        )
    )

    meta_model = LogisticRegression(
        C=1.0,
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
    )

    meta_model.fit(
        meta_score_features(
            calibration_components
        ),
        calibration_frame[
            "label"
        ],
    )

    calibration_probabilities = (
        meta_model
        .predict_proba(
            meta_score_features(
                calibration_components
            )
        )[:, 1]
    )

    y_calibration = (
        calibration_frame[
            "label"
        ]
        .to_numpy(
            dtype=int
        )
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
            test_frame["text"],
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

    y_test = (
        test_frame[
            "label"
        ]
        .to_numpy(
            dtype=int
        )
    )

    metrics = evaluate(
        y_test,
        test_probabilities,
        human_threshold=human_threshold,
        ai_threshold=ai_threshold,
    )

    metadata = {
        "format_version": (
            FORMAT_VERSION
        ),
        "model_name": (
            "DeteksiAI Hybrid ID"
        ),
        "model_version": "2.0",
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
                    dataframe[
                        "label"
                    ]
                    == 0
                )
            ),
            "ai": int(
                np.sum(
                    dataframe[
                        "label"
                    ]
                    == 1
                )
            ),
        },
        "sources": {
            key: int(value)
            for key, value
            in (
                dataframe[
                    "source"
                ]
                .value_counts()
                .to_dict()
                .items()
            )
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