import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from detector.features import meta_score_features, stylometry_matrix
from training.common import RANDOM_STATE
from training.weights import source_sample_weights


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
                    max_features=90000,
                    sublinear_tf=True,
                    strip_accents=None,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.8,
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
                    max_features=120000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.8,
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
                    C=0.7,
                    max_iter=4000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_models() -> dict[
    str,
    Pipeline,
]:
    return {
        "word": build_word_model(),
        "char": build_char_model(),
        "style": build_style_model(),
    }


def fit_models(
    models: dict[str, Pipeline],
    frame: pd.DataFrame,
) -> dict[str, Pipeline]:
    texts = (
        frame["text"]
        .astype(str)
        .tolist()
    )

    labels = (
        frame["label"]
        .to_numpy(
            dtype=np.int64
        )
    )

    weights = source_sample_weights(
        frame
    )

    models["word"].fit(
        texts,
        labels,
        classifier__sample_weight=weights,
    )

    models["char"].fit(
        texts,
        labels,
        classifier__sample_weight=weights,
    )

    models["style"].fit(
        stylometry_matrix(
            texts
        ),
        labels,
        classifier__sample_weight=weights,
    )

    return models


def component_probabilities(
    models: dict[str, Pipeline],
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


def out_of_fold_components(
    frame: pd.DataFrame,
    folds: int,
) -> np.ndarray:
    unique_groups = (
        frame["group_id"]
        .nunique()
    )

    effective_folds = min(
        folds,
        unique_groups,
    )

    if effective_folds < 2:
        raise ValueError(
            "Group terlalu sedikit untuk out-of-fold stacking."
        )

    labels = (
        frame["label"]
        .to_numpy(
            dtype=np.int64
        )
    )

    groups = (
        frame["group_id"]
        .astype(str)
        .to_numpy()
    )

    texts = (
        frame["text"]
        .astype(str)
        .to_numpy()
    )

    output = np.zeros(
        (
            len(frame),
            3,
        ),
        dtype=np.float64,
    )

    filled = np.zeros(
        len(frame),
        dtype=bool,
    )

    splitter = StratifiedGroupKFold(
        n_splits=effective_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for (
        fold_index,
        (
            train_index,
            validation_index,
        ),
    ) in enumerate(
        splitter.split(
            texts,
            labels,
            groups,
        ),
        start=1,
    ):
        train_frame = (
            frame
            .iloc[
                train_index
            ]
            .reset_index(
                drop=True
            )
        )

        validation_texts = (
            frame
            .iloc[
                validation_index
            ]["text"]
            .astype(str)
            .tolist()
        )

        models = fit_models(
            build_models(),
            train_frame,
        )

        output[
            validation_index
        ] = (
            component_probabilities(
                models,
                validation_texts,
            )
        )

        filled[
            validation_index
        ] = True

        print(
            f"Stacking fold "
            f"{fold_index}/"
            f"{effective_folds}: "
            f"train={len(train_index)}, "
            f"valid={len(validation_index)}"
        )

    if not np.all(filled):
        raise RuntimeError(
            "Tidak semua sampel development memperoleh prediksi out-of-fold."
        )

    return output


def build_meta_model() -> LogisticRegression:
    return LogisticRegression(
        C=0.8,
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
    )


def fit_meta_model(
    frame: pd.DataFrame,
    oof_components: np.ndarray,
) -> LogisticRegression:
    model = build_meta_model()

    labels = (
        frame["label"]
        .to_numpy(
            dtype=np.int64
        )
    )

    weights = source_sample_weights(
        frame
    )

    model.fit(
        meta_score_features(
            oof_components
        ),
        labels,
        sample_weight=weights,
    )

    return model


def final_probabilities(
    models: dict[str, Pipeline],
    meta_model: LogisticRegression,
    texts: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    components = (
        component_probabilities(
            models,
            texts,
        )
    )

    probabilities = (
        meta_model
        .predict_proba(
            meta_score_features(
                components
            )
        )[:, 1]
    )

    return (
        probabilities,
        components,
    )