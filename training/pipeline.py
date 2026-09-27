import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from detector.config import FORMAT_VERSION
from detector.features import STYLE_FEATURE_NAMES, meta_score_features
from training.augmentation import load_public_augmentation
from training.common import (
    canonical_fingerprint,
    clean_dataframe,
    remove_conflicts_and_duplicates,
)
from training.datasets import (
    load_external_test_csv,
    load_extra_csv,
    load_primary_dataset,
)
from training.metrics import evaluate, select_thresholds, source_metrics
from training.models import (
    build_models,
    final_probabilities,
    fit_meta_model,
    fit_models,
    out_of_fold_components,
)
from training.splitting import split_development_threshold_test


def _validate_dataset(
    frame: pd.DataFrame,
) -> None:
    if len(frame) < 1000:
        raise ValueError(
            "Dataset terlalu kecil. Minimal 1000 sampel setelah pembersihan."
        )

    class_counts = (
        frame["label"]
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

    if (
        frame["group_id"]
        .nunique()
        < 30
    ):
        raise ValueError(
            "Jumlah group_id terlalu sedikit untuk evaluasi group-aware."
        )


def _remove_external_overlap(
    external: pd.DataFrame,
    training: pd.DataFrame,
) -> pd.DataFrame:
    if external.empty:
        return external

    training_fingerprints = {
        canonical_fingerprint(
            text
        )
        for text in (
            training["text"]
            .astype(str)
        )
    }

    fingerprints = (
        external["text"]
        .astype(str)
        .map(
            canonical_fingerprint
        )
    )

    clean = external[
        ~fingerprints.isin(
            training_fingerprints
        )
    ].copy()

    return clean.reset_index(
        drop=True
    )


def _frame_summary(
    frame: pd.DataFrame,
) -> dict:
    return {
        "samples": int(
            len(frame)
        ),
        "groups": int(
            frame["group_id"]
            .nunique()
        ),
        "human": int(
            np.sum(
                frame["label"]
                .to_numpy(
                    dtype=np.int64
                )
                == 0
            )
        ),
        "ai": int(
            np.sum(
                frame["label"]
                .to_numpy(
                    dtype=np.int64
                )
                == 1
            )
        ),
    }


def train_detector(
    output: str,
    public_augmentation: int,
    extra_csv: list[str],
    external_test_csv: list[str],
    stack_folds: int,
    target_false_positive_rate: float,
    target_ai_missed_as_human_rate: float,
) -> dict:
    print(
        "Memuat dataset utama..."
    )

    primary = load_primary_dataset()

    print(
        "Memuat augmentasi publik..."
    )

    augmentation = (
        load_public_augmentation(
            public_augmentation
        )
    )

    extras = load_extra_csv(
        extra_csv
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

    dataframe = (
        remove_conflicts_and_duplicates(
            pd.concat(
                frames,
                ignore_index=True,
            )
        )
    )

    dataframe = clean_dataframe(
        dataframe
    )

    _validate_dataset(
        dataframe
    )

    source_counts = (
        dataframe["source"]
        .value_counts()
        .to_dict()
    )

    class_counts = (
        dataframe["label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print(
        f"Total data bersih: "
        f"{len(dataframe)}"
    )

    print(
        f"Distribusi label: "
        f"{class_counts}"
    )

    print(
        "Jumlah group: "
        f"{dataframe['group_id'].nunique()}"
    )

    print(
        "Sumber data:"
    )

    for (
        source,
        count,
    ) in source_counts.items():
        print(
            f"- {source}: {count}"
        )

    (
        development,
        threshold_frame,
        test_frame,
    ) = (
        split_development_threshold_test(
            dataframe
        )
    )

    print(
        "Split group-aware:"
    )

    print(
        "- development: "
        f"{_frame_summary(development)}"
    )

    print(
        "- threshold: "
        f"{_frame_summary(threshold_frame)}"
    )

    print(
        "- test: "
        f"{_frame_summary(test_frame)}"
    )

    print(
        "Membuat prediksi out-of-fold untuk stacking..."
    )

    oof_components = (
        out_of_fold_components(
            development,
            stack_folds,
        )
    )

    meta_model = fit_meta_model(
        development,
        oof_components,
    )

    oof_probabilities = (
        meta_model
        .predict_proba(
            meta_score_features(
                oof_components
            )
        )[:, 1]
    )

    print(
        "Melatih model dasar final pada development set..."
    )

    models = fit_models(
        build_models(),
        development,
    )

    threshold_texts = (
        threshold_frame["text"]
        .astype(str)
        .tolist()
    )

    threshold_labels = (
        threshold_frame["label"]
        .to_numpy(
            dtype=np.int64
        )
    )

    (
        threshold_probabilities,
        _,
    ) = (
        final_probabilities(
            models,
            meta_model,
            threshold_texts,
        )
    )

    (
        human_threshold,
        ai_threshold,
    ) = (
        select_thresholds(
            threshold_labels,
            threshold_probabilities,
            target_false_positive_rate=(
                target_false_positive_rate
            ),
            target_ai_missed_as_human_rate=(
                target_ai_missed_as_human_rate
            ),
        )
    )

    print(
        "Ambang keputusan: "
        f"human <= "
        f"{human_threshold:.4f}, "
        f"AI >= "
        f"{ai_threshold:.4f}"
    )

    test_texts = (
        test_frame["text"]
        .astype(str)
        .tolist()
    )

    y_test = (
        test_frame["label"]
        .to_numpy(
            dtype=np.int64
        )
    )

    (
        test_probabilities,
        _,
    ) = (
        final_probabilities(
            models,
            meta_model,
            test_texts,
        )
    )

    metrics = evaluate(
        y_test,
        test_probabilities,
        human_threshold,
        ai_threshold,
    )

    development_metrics = (
        evaluate(
            development["label"]
            .to_numpy(
                dtype=np.int64
            ),
            oof_probabilities,
            human_threshold,
            ai_threshold,
        )
    )

    external_evaluation = None

    if external_test_csv:
        external = (
            remove_conflicts_and_duplicates(
                load_external_test_csv(
                    external_test_csv
                )
            )
        )

        external = (
            _remove_external_overlap(
                external,
                dataframe,
            )
        )

        if (
            len(external) >= 2
            and external[
                "label"
            ].nunique()
            == 2
        ):
            (
                external_probabilities,
                _,
            ) = (
                final_probabilities(
                    models,
                    meta_model,
                    external["text"]
                    .astype(str)
                    .tolist(),
                )
            )

            external_evaluation = {
                "summary": evaluate(
                    external["label"]
                    .to_numpy(
                        dtype=np.int64
                    ),
                    external_probabilities,
                    human_threshold,
                    ai_threshold,
                ),
                "sources": source_metrics(
                    external,
                    external_probabilities,
                    human_threshold,
                    ai_threshold,
                ),
                "samples": int(
                    len(external)
                ),
            }

        elif external_test_csv:
            print(
                "Peringatan: external test tidak dievaluasi karena "
                "setelah pembersihan tidak memiliki kedua kelas."
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
        "model_version": "3.0",
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
        "threshold_targets": {
            "ai_false_positive_rate": (
                target_false_positive_rate
            ),
            "ai_missed_as_human_rate": (
                target_ai_missed_as_human_rate
            ),
        },
        "split_strategy": {
            "type": (
                "group-aware"
            ),
            "development_ratio": (
                0.75
            ),
            "threshold_ratio": (
                0.10
            ),
            "test_ratio": (
                0.15
            ),
            "stacking": (
                "out-of-fold"
            ),
            "stack_folds": (
                stack_folds
            ),
        },
        "samples": {
            "total": int(
                len(dataframe)
            ),
            "development": int(
                len(development)
            ),
            "threshold": int(
                len(
                    threshold_frame
                )
            ),
            "test": int(
                len(
                    test_frame
                )
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
            "groups": int(
                dataframe[
                    "group_id"
                ].nunique()
            ),
        },
        "sources": {
            str(key): int(value)
            for (
                key,
                value,
            ) in source_counts.items()
        },
        "style_features": (
            list(
                STYLE_FEATURE_NAMES
            )
        ),
        "evaluation": metrics,
        "development_oof_evaluation": (
            development_metrics
        ),
        "test_source_breakdown": (
            source_metrics(
                test_frame,
                test_probabilities,
                human_threshold,
                ai_threshold,
            )
        ),
        "external_evaluation": (
            external_evaluation
        ),
    }

    output_path = Path(
        output
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
        f"Model disimpan ke: "
        f"{output_path}"
    )

    return metadata