import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from features import StylometryTransformer


def normalize_text(value: str) -> str:
    return " ".join(
        str(value)
        .replace("\u00a0", " ")
        .split()
    )


def load_training_data(csv_path: str | None) -> pd.DataFrame:
    if csv_path:
        dataframe = pd.read_csv(csv_path)
    else:
        dataset = load_dataset(
            "shouwiku/detectai-indonesian",
            split="train"
        )

        dataframe = dataset.to_pandas()

    required_columns = {
        "text",
        "label",
    }

    if not required_columns.issubset(dataframe.columns):
        raise ValueError(
            "Dataset harus memiliki kolom 'text' dan 'label'."
        )

    dataframe = dataframe.copy()

    dataframe["text"] = (
        dataframe["text"]
        .astype(str)
        .map(normalize_text)
    )

    dataframe["label"] = pd.to_numeric(
        dataframe["label"],
        errors="coerce"
    )

    dataframe = dataframe.dropna(
        subset=[
            "text",
            "label",
        ]
    )

    dataframe["label"] = (
        dataframe["label"]
        .astype(int)
    )

    dataframe = dataframe[
        dataframe["label"].isin([0, 1])
    ]

    dataframe["word_count_calc"] = (
        dataframe["text"]
        .str.split()
        .str.len()
    )

    dataframe = dataframe[
        dataframe["word_count_calc"] >= 25
    ]

    conflicting = (
        dataframe
        .groupby("text")["label"]
        .nunique()
    )

    conflicting = set(
        conflicting[
            conflicting > 1
        ].index
    )

    if conflicting:
        dataframe = dataframe[
            ~dataframe["text"].isin(conflicting)
        ]

    dataframe = (
        dataframe
        .drop_duplicates(
            subset=["text"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    return dataframe


def build_model() -> CalibratedClassifierCV:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.995,
                    max_features=50000,
                    sublinear_tf=True,
                    strip_accents=None,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=60000,
                    sublinear_tf=True,
                ),
            ),
            (
                "style",
                StylometryTransformer(),
            ),
        ]
    )

    classifier = LogisticRegression(
        C=2.0,
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    base_model = Pipeline(
        [
            (
                "features",
                features,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    return CalibratedClassifierCV(
        base_model,
        method="sigmoid",
        cv=5,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        default=None,
        help="CSV opsional dengan kolom text,label",
    )

    parser.add_argument(
        "--output",
        default="models/detector.joblib",
    )

    arguments = parser.parse_args()

    dataframe = load_training_data(
        arguments.csv
    )

    if len(dataframe) < 200:
        raise ValueError(
            "Dataset terlalu kecil. Gunakan setidaknya beberapa ratus sampel per kelas."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        dataframe["text"],
        dataframe["label"],
        test_size=0.2,
        stratify=dataframe["label"],
        random_state=42,
    )

    model = build_model()

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {
        "accuracy": float(
            accuracy_score(
                y_test,
                predictions
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                predictions
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_test,
                probabilities
            )
        ),
        "brier": float(
            brier_score_loss(
                y_test,
                probabilities
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
        ),
        "samples_total": int(
            len(dataframe)
        ),
        "samples_train": int(
            len(X_train)
        ),
        "samples_test": int(
            len(X_test)
        ),
        "human_samples": int(
            (
                dataframe["label"] == 0
            ).sum()
        ),
        "ai_samples": int(
            (
                dataframe["label"] == 1
            ).sum()
        ),
        "trained_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "decision_thresholds": {
            "human": 0.25,
            "ai": 0.75,
        },
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
            "model": model,
            "metadata": metrics,
        },
        output_path,
    )

    print(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        f"Model disimpan ke: {output_path}"
    )


if __name__ == "__main__":
    main()