import numpy as np

from detector.config import (
    MAX_CHUNK_WORDS,
    MIN_INDONESIAN_SIGNAL,
    MIN_WORDS,
    SHORT_TEXT_WORDS,
    TARGET_CHUNK_WORDS,
)
from detector.features import (
    indonesian_signal,
    meta_score_features,
    style_snapshot,
    stylometry_matrix,
)
from detector.model_store import model_store
from detector.text import chunk_text, count_words, normalize_text


class AnalysisError(ValueError):
    pass


def predict_texts(
    texts: list[str],
    bundle: dict,
) -> tuple[np.ndarray, np.ndarray]:
    models = bundle["models"]

    word_scores = (
        models["word"]
        .predict_proba(texts)[:, 1]
    )

    char_scores = (
        models["char"]
        .predict_proba(texts)[:, 1]
    )

    style_scores = (
        models["style"]
        .predict_proba(
            stylometry_matrix(texts)
        )[:, 1]
    )

    component_scores = np.column_stack(
        [
            word_scores,
            char_scores,
            style_scores,
        ]
    )

    final_scores = (
        bundle["meta_model"]
        .predict_proba(
            meta_score_features(
                component_scores
            )
        )[:, 1]
    )

    return (
        final_scores,
        component_scores,
    )


def classify_score(
    score: float,
    human_threshold: float,
    ai_threshold: float,
) -> tuple[str, str]:
    if score >= ai_threshold:
        return "Indikasi AI", "ai"

    if score <= human_threshold:
        return "Cenderung manusia", "human"

    return "Belum pasti", "uncertain"


def weighted_average(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    return float(
        np.average(
            values,
            weights=weights,
        )
    )


def effective_thresholds(
    human_threshold: float,
    ai_threshold: float,
    word_count: int,
) -> tuple[float, float]:
    human = human_threshold
    ai = ai_threshold

    if word_count < SHORT_TEXT_WORDS:
        human -= 0.04
        ai += 0.04

    if word_count < 70:
        human -= 0.03
        ai += 0.03

    human = float(
        np.clip(
            human,
            0.03,
            0.45,
        )
    )

    ai = float(
        np.clip(
            ai,
            0.55,
            0.97,
        )
    )

    return human, ai


def confidence_label(
    verdict_key: str,
    overall_score: float,
    word_count: int,
    model_agreement: float,
    section_consistency: float,
    human_threshold: float,
    ai_threshold: float,
) -> tuple[str, int]:
    length_factor = float(
        np.clip(
            (
                word_count
                - MIN_WORDS
            )
            / 350,
            0.0,
            1.0,
        )
    )

    if verdict_key == "ai":
        decision_margin = float(
            np.clip(
                (
                    overall_score
                    - ai_threshold
                )
                / max(
                    1.0 - ai_threshold,
                    0.05,
                ),
                0.0,
                1.0,
            )
        )

    elif verdict_key == "human":
        decision_margin = float(
            np.clip(
                (
                    human_threshold
                    - overall_score
                )
                / max(
                    human_threshold,
                    0.05,
                ),
                0.0,
                1.0,
            )
        )

    else:
        center = (
            human_threshold
            + ai_threshold
        ) / 2

        half_width = max(
            (
                ai_threshold
                - human_threshold
            )
            / 2,
            0.05,
        )

        decision_margin = float(
            np.clip(
                1.0
                - abs(
                    overall_score
                    - center
                )
                / half_width,
                0.0,
                1.0,
            )
        )

    score = (
        0.35 * model_agreement
        + 0.25 * section_consistency
        + 0.20 * length_factor
        + 0.20 * decision_margin
    )

    if verdict_key == "uncertain":
        score = min(
            score,
            0.55,
        )

    value = int(
        round(
            score
            * 100
        )
    )

    if value >= 76:
        return "Tinggi", value

    if value >= 54:
        return "Sedang", value

    return "Rendah", value


def build_notes(
    text: str,
    word_count: int,
    chunks: list[str],
    final_scores: np.ndarray,
    component_scores: np.ndarray,
    human_threshold: float,
    ai_threshold: float,
    language_score: float,
) -> list[str]:
    notes: list[str] = []

    if word_count < 80:
        notes.append(
            "Teks masih pendek untuk deteksi kepengarangan. "
            "Sistem memakai ambang yang lebih ketat agar tidak mudah "
            "memberi keputusan pasti."
        )

    elif word_count < 150:
        notes.append(
            "Panjang teks cukup untuk pemeriksaan awal, tetapi hasil "
            "biasanya lebih stabil pada dokumen yang lebih panjang."
        )

    if language_score < MIN_INDONESIAN_SIGNAL:
        notes.append(
            "Teks tidak terlihat dominan berbahasa Indonesia, sehingga "
            "hasil diperlakukan sebagai belum pasti."
        )

    component_spread = float(
        np.mean(
            np.ptp(
                component_scores,
                axis=1,
            )
        )
    )

    if component_spread > 0.36:
        notes.append(
            "Model pola kata, karakter, dan gaya tulis memberi skor "
            "yang cukup berbeda."
        )

    if len(chunks) >= 2:
        ai_parts = int(
            np.sum(
                final_scores
                >= ai_threshold
            )
        )

        human_parts = int(
            np.sum(
                final_scores
                <= human_threshold
            )
        )

        if ai_parts > 0 and human_parts > 0:
            notes.append(
                "Dokumen memiliki bagian dengan pola yang saling "
                "bertentangan, sehingga keputusan keseluruhan dibuat "
                "lebih konservatif."
            )

    notes.append(
        "Indeks AI adalah skor klasifikasi model, bukan bukti "
        "kepengarangan dan bukan persentase kepastian absolut."
    )

    return notes


def analyze_text(text: str) -> dict:
    value = normalize_text(text)
    word_count = count_words(value)

    if word_count < MIN_WORDS:
        raise AnalysisError(
            f"Teks terlalu pendek. Masukkan minimal {MIN_WORDS} kata "
            "agar hasil tidak terlalu mudah berubah."
        )

    bundle = model_store.load()

    metadata = bundle.get(
        "metadata",
        {},
    )

    stored_thresholds = metadata.get(
        "decision_thresholds",
        {
            "human": 0.20,
            "ai": 0.80,
        },
    )

    human_threshold, ai_threshold = (
        effective_thresholds(
            float(
                stored_thresholds.get(
                    "human",
                    0.20,
                )
            ),
            float(
                stored_thresholds.get(
                    "ai",
                    0.80,
                )
            ),
            word_count,
        )
    )

    chunks = chunk_text(
        value,
        target_words=TARGET_CHUNK_WORDS,
        max_words=MAX_CHUNK_WORDS,
    )

    final_scores, component_scores = (
        predict_texts(
            chunks,
            bundle,
        )
    )

    weights = np.asarray(
        [
            max(
                count_words(chunk),
                1,
            )
            for chunk in chunks
        ],
        dtype=np.float64,
    )

    mean_score = weighted_average(
        final_scores,
        weights,
    )

    median_score = float(
        np.median(
            final_scores
        )
    )

    overall_score = (
        mean_score
        if len(final_scores) == 1
        else (
            0.50 * mean_score
            + 0.50 * median_score
        )
    )

    component_overall = np.asarray(
        [
            weighted_average(
                component_scores[:, index],
                weights,
            )
            for index in range(
                component_scores.shape[1]
            )
        ],
        dtype=np.float64,
    )

    component_spread = float(
        np.mean(
            np.ptp(
                component_scores,
                axis=1,
            )
        )
    )

    model_agreement = float(
        np.clip(
            1.0
            - component_spread
            / 0.65,
            0.0,
            1.0,
        )
    )

    section_std = (
        float(
            np.std(
                final_scores
            )
        )
        if len(final_scores) > 1
        else 0.0
    )

    section_consistency = float(
        np.clip(
            1.0
            - section_std
            / 0.30,
            0.0,
            1.0,
        )
    )

    ai_votes = int(
        np.sum(
            final_scores
            >= ai_threshold
        )
    )

    human_votes = int(
        np.sum(
            final_scores
            <= human_threshold
        )
    )

    uncertain_votes = (
        len(final_scores)
        - ai_votes
        - human_votes
    )

    if len(final_scores) == 1:
        verdict_label, verdict_key = (
            classify_score(
                overall_score,
                human_threshold,
                ai_threshold,
            )
        )

    else:
        minimum_votes = max(
            1,
            int(
                np.ceil(
                    len(final_scores)
                    * 0.60
                )
            ),
        )

        if (
            overall_score >= ai_threshold
            and ai_votes >= minimum_votes
        ):
            verdict_label = "Indikasi AI kuat"
            verdict_key = "ai"

        elif (
            overall_score <= human_threshold
            and human_votes >= minimum_votes
        ):
            verdict_label = "Cenderung ditulis manusia"
            verdict_key = "human"

        else:
            verdict_label = "Belum cukup bukti"
            verdict_key = "uncertain"

    language_score = indonesian_signal(
        value
    )

    mixed_sections = (
        ai_votes > 0
        and human_votes > 0
    )

    strong_disagreement = (
        component_spread > 0.45
    )

    if (
        language_score
        < MIN_INDONESIAN_SIGNAL
        or strong_disagreement
    ):
        verdict_label = "Belum cukup bukti"
        verdict_key = "uncertain"

    elif (
        mixed_sections
        and max(
            ai_votes,
            human_votes,
        )
        < int(
            np.ceil(
                len(final_scores)
                * 0.75
            )
        )
    ):
        verdict_label = "Belum cukup bukti"
        verdict_key = "uncertain"

    confidence, confidence_score = (
        confidence_label(
            verdict_key,
            overall_score,
            word_count,
            model_agreement,
            section_consistency,
            human_threshold,
            ai_threshold,
        )
    )

    sections: list[dict] = []

    for index, chunk in enumerate(chunks):
        section_label, section_key = (
            classify_score(
                float(
                    final_scores[index]
                ),
                human_threshold,
                ai_threshold,
            )
        )

        sections.append(
            {
                "index": index + 1,
                "text": chunk,
                "word_count": count_words(
                    chunk
                ),
                "score": int(
                    round(
                        float(
                            final_scores[index]
                        )
                        * 100
                    )
                ),
                "label": section_label,
                "label_key": section_key,
                "components": {
                    "word": int(
                        round(
                            float(
                                component_scores[
                                    index,
                                    0,
                                ]
                            )
                            * 100
                        )
                    ),
                    "char": int(
                        round(
                            float(
                                component_scores[
                                    index,
                                    1,
                                ]
                            )
                            * 100
                        )
                    ),
                    "style": int(
                        round(
                            float(
                                component_scores[
                                    index,
                                    2,
                                ]
                            )
                            * 100
                        )
                    ),
                },
            }
        )

    if verdict_key == "ai":
        summary = (
            "Skor keseluruhan melewati ambang AI dan mayoritas "
            "bagian menunjukkan pola yang searah."
        )

    elif verdict_key == "human":
        summary = (
            "Skor keseluruhan berada di bawah ambang manusia dan "
            "mayoritas bagian menunjukkan pola yang searah."
        )

    else:
        summary = (
            "Skor, bahasa, atau keputusan antarbagian belum cukup "
            "konsisten untuk menetapkan satu sisi secara aman."
        )

    notes = build_notes(
        value,
        word_count,
        chunks,
        final_scores,
        component_scores,
        human_threshold,
        ai_threshold,
        language_score,
    )

    evaluation = metadata.get(
        "evaluation",
        {},
    )

    samples = metadata.get(
        "samples",
        {},
    )

    return {
        "verdict": verdict_label,
        "verdict_key": verdict_key,
        "summary": summary,
        "ai_index": int(
            round(
                overall_score
                * 100
            )
        ),
        "confidence": confidence,
        "confidence_score": confidence_score,
        "model_agreement": int(
            round(
                model_agreement
                * 100
            )
        ),
        "section_consistency": int(
            round(
                section_consistency
                * 100
            )
        ),
        "word_count": word_count,
        "character_count": len(value),
        "section_count": len(chunks),
        "section_summary": {
            "ai": ai_votes,
            "human": human_votes,
            "uncertain": uncertain_votes,
        },
        "components": {
            "word": int(
                round(
                    float(
                        component_overall[0]
                    )
                    * 100
                )
            ),
            "char": int(
                round(
                    float(
                        component_overall[1]
                    )
                    * 100
                )
            ),
            "style": int(
                round(
                    float(
                        component_overall[2]
                    )
                    * 100
                )
            ),
        },
        "style": style_snapshot(value),
        "notes": notes,
        "sections": sections,
        "model": {
            "name": metadata.get(
                "model_name",
                "DeteksiAI Hybrid ID",
            ),
            "version": metadata.get(
                "model_version",
                "3.0",
            ),
            "sources": metadata.get(
                "sources",
                {},
            ),
            "test_samples": samples.get(
                "test"
            ),
            "selective_accuracy": evaluation.get(
                "selective_accuracy"
            ),
            "selective_coverage": evaluation.get(
                "selective_coverage"
            ),
            "ai_false_positive_rate": evaluation.get(
                "ai_false_positive_rate"
            ),
            "balanced_accuracy": evaluation.get(
                "balanced_accuracy_at_0_5"
            ),
            "roc_auc": evaluation.get(
                "roc_auc"
            ),
            "external_evaluation": metadata.get(
                "external_evaluation"
            ),
            "trained_at": metadata.get(
                "trained_at"
            ),
        },
        "thresholds": {
            "human": int(
                round(
                    human_threshold
                    * 100
                )
            ),
            "ai": int(
                round(
                    ai_threshold
                    * 100
                )
            ),
        },
    }