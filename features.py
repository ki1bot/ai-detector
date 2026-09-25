import math
import re
from collections import Counter
from typing import Iterable

import numpy as np

INDONESIAN_FUNCTION_WORDS = {
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "pada", "adalah", "ini",
    "itu", "dalam", "sebagai", "oleh", "karena", "atau", "juga", "akan", "dapat", "tidak",
    "tersebut", "lebih", "telah", "saat", "sehingga", "serta", "antara", "bagi", "terhadap",
    "namun", "jika", "maka", "setelah", "sebelum", "melalui", "tentang", "hingga", "masih",
    "sudah", "bisa", "menjadi", "merupakan", "agar", "bahwa", "secara", "setiap", "para",
    "sangat", "mereka", "kami", "kita", "saya", "anda", "ia", "dia", "ada", "sebuah",
    "suatu", "bukan", "belum", "hanya", "banyak",
}

CONNECTORS = (
    "selain itu",
    "oleh karena itu",
    "dengan demikian",
    "di sisi lain",
    "sementara itu",
    "secara keseluruhan",
    "pada dasarnya",
    "dalam konteks",
    "dapat disimpulkan",
    "lebih lanjut",
    "di samping itu",
    "meskipun demikian",
)

ENUMERATORS = (
    "pertama",
    "kedua",
    "ketiga",
    "keempat",
    "kelima",
    "selanjutnya",
    "terakhir",
)

STYLE_FEATURE_NAMES = (
    "log_word_count",
    "log_char_count",
    "log_sentence_count",
    "sentence_length_mean",
    "sentence_length_std",
    "sentence_length_cv",
    "sentence_length_q90_q10",
    "word_length_mean",
    "word_length_std",
    "lexical_diversity",
    "hapax_ratio",
    "function_word_ratio",
    "lexical_entropy",
    "punctuation_ratio",
    "comma_per_sentence",
    "semicolon_per_sentence",
    "colon_per_sentence",
    "question_per_sentence",
    "exclamation_per_sentence",
    "dash_per_sentence",
    "parenthesis_per_sentence",
    "quote_per_sentence",
    "digit_ratio",
    "uppercase_ratio",
    "newline_ratio",
    "connector_per_sentence",
    "enumerator_per_sentence",
    "repeated_bigram_ratio",
    "repeated_trigram_ratio",
    "sentence_start_repeat_ratio",
    "sentence_start_diversity",
    "log_paragraph_count",
    "paragraph_length_mean",
    "paragraph_length_std",
    "long_sentence_ratio",
    "short_sentence_ratio",
    "sentence_burstiness",
)


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", text.lower(), flags=re.UNICODE)


def split_sentences(text: str) -> list[str]:
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+|\n+", text)
        if item.strip()
    ]

    return sentences or ([text.strip()] if text.strip() else [])


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [
        item.strip()
        for item in re.split(r"\n\s*\n", text)
        if item.strip()
    ]

    return paragraphs or ([text.strip()] if text.strip() else [])


def _ngram_repeat_ratio(words: list[str], n: int) -> float:
    if len(words) < n:
        return 0.0

    grams = [
        tuple(words[index:index + n])
        for index in range(len(words) - n + 1)
    ]

    counts = Counter(grams)

    repeated = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    return repeated / max(len(grams), 1)


def _lexical_entropy(words: list[str]) -> float:
    if len(words) < 2:
        return 0.0

    counts = Counter(words)

    if len(counts) <= 1:
        return 0.0

    total = len(words)

    entropy = -sum(
        (count / total) * math.log(count / total)
        for count in counts.values()
    )

    return entropy / math.log(len(counts))


def _sentence_start_metrics(sentences: list[str]) -> tuple[float, float]:
    starts = []

    for sentence in sentences:
        words = tokenize_words(sentence)

        if words:
            starts.append(
                tuple(
                    words[:min(3, len(words))]
                )
            )

    if not starts:
        return 0.0, 0.0

    counts = Counter(starts)

    repeated = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    return (
        repeated / len(starts),
        len(counts) / len(starts),
    )


def extract_stylometry(text: str) -> list[float]:
    words = tokenize_words(text)
    sentences = split_sentences(text)
    paragraphs = split_paragraphs(text)

    word_count = max(len(words), 1)
    char_count = max(len(text), 1)
    sentence_count = max(len(sentences), 1)
    paragraph_count = max(len(paragraphs), 1)

    sentence_lengths = np.asarray(
        [
            len(tokenize_words(sentence))
            for sentence in sentences
        ] or [0],
        dtype=np.float64,
    )

    word_lengths = np.asarray(
        [
            len(word)
            for word in words
        ] or [0],
        dtype=np.float64,
    )

    paragraph_lengths = np.asarray(
        [
            len(tokenize_words(paragraph))
            for paragraph in paragraphs
        ] or [0],
        dtype=np.float64,
    )

    frequencies = Counter(words)

    unique_words = len(frequencies)

    hapax = sum(
        1
        for count in frequencies.values()
        if count == 1
    )

    function_words = sum(
        1
        for word in words
        if word in INDONESIAN_FUNCTION_WORDS
    )

    letters = [
        character
        for character in text
        if character.isalpha()
    ]

    uppercase = sum(
        character.isupper()
        for character in letters
    )

    punctuation = sum(
        character in ".,;:!?-—()[]{}\"'…"
        for character in text
    )

    digits = sum(
        character.isdigit()
        for character in text
    )

    lowered = text.lower()

    connector_hits = sum(
        lowered.count(connector)
        for connector in CONNECTORS
    )

    enumerator_hits = sum(
        len(
            re.findall(
                rf"\b{re.escape(item)}\b",
                lowered,
            )
        )
        for item in ENUMERATORS
    )

    start_repeat, start_diversity = _sentence_start_metrics(
        sentences
    )

    sentence_mean = float(
        sentence_lengths.mean()
    )

    sentence_std = float(
        sentence_lengths.std()
    )

    sentence_cv = (
        sentence_std
        / max(sentence_mean, 1.0)
    )

    sentence_q10 = float(
        np.quantile(
            sentence_lengths,
            0.10,
        )
    )

    sentence_q90 = float(
        np.quantile(
            sentence_lengths,
            0.90,
        )
    )

    if len(sentence_lengths) > 1:
        burstiness = (
            float(
                np.abs(
                    np.diff(sentence_lengths)
                ).mean()
            )
            / max(sentence_mean, 1.0)
        )
    else:
        burstiness = 0.0

    return [
        float(np.log1p(word_count)),
        float(np.log1p(char_count)),
        float(np.log1p(sentence_count)),
        sentence_mean,
        sentence_std,
        sentence_cv,
        sentence_q90 - sentence_q10,
        float(word_lengths.mean()),
        float(word_lengths.std()),
        unique_words / word_count,
        hapax / word_count,
        function_words / word_count,
        _lexical_entropy(words),
        punctuation / char_count,
        text.count(",") / sentence_count,
        text.count(";") / sentence_count,
        text.count(":") / sentence_count,
        text.count("?") / sentence_count,
        text.count("!") / sentence_count,
        (
            text.count("-")
            + text.count("—")
        ) / sentence_count,
        (
            text.count("(")
            + text.count(")")
        ) / sentence_count,
        (
            text.count('"')
            + text.count("“")
            + text.count("”")
            + text.count("'")
            + text.count("‘")
            + text.count("’")
        ) / sentence_count,
        digits / char_count,
        uppercase / max(len(letters), 1),
        text.count("\n") / char_count,
        connector_hits / sentence_count,
        enumerator_hits / sentence_count,
        _ngram_repeat_ratio(words, 2),
        _ngram_repeat_ratio(words, 3),
        start_repeat,
        start_diversity,
        float(np.log1p(paragraph_count)),
        float(paragraph_lengths.mean()),
        float(paragraph_lengths.std()),
        float(
            np.mean(
                sentence_lengths >= 25
            )
        ),
        float(
            np.mean(
                sentence_lengths <= 7
            )
        ),
        burstiness,
    ]


def stylometry_matrix(
    texts: Iterable[str],
) -> np.ndarray:
    return np.asarray(
        [
            extract_stylometry(
                str(text)
            )
            for text in texts
        ],
        dtype=np.float64,
    )


def meta_score_features(
    component_scores: np.ndarray,
) -> np.ndarray:
    scores = np.asarray(
        component_scores,
        dtype=np.float64,
    )

    if (
        scores.ndim != 2
        or scores.shape[1] != 3
    ):
        raise ValueError(
            "component_scores harus berbentuk matriks N x 3."
        )

    word_scores = scores[:, 0]
    char_scores = scores[:, 1]
    style_scores = scores[:, 2]

    return np.column_stack(
        [
            word_scores,
            char_scores,
            style_scores,
            np.abs(
                word_scores
                - char_scores
            ),
            np.abs(
                word_scores
                - style_scores
            ),
            np.abs(
                char_scores
                - style_scores
            ),
            scores.mean(axis=1),
            scores.std(axis=1),
        ]
    )


def indonesian_signal(text: str) -> float:
    words = tokenize_words(text)

    if not words:
        return 0.0

    hits = sum(
        1
        for word in words
        if word in INDONESIAN_FUNCTION_WORDS
    )

    return hits / len(words)


def style_snapshot(
    text: str,
) -> dict[str, float]:
    words = tokenize_words(text)

    sentences = split_sentences(text)

    lengths = np.asarray(
        [
            len(
                tokenize_words(sentence)
            )
            for sentence in sentences
        ] or [0],
        dtype=np.float64,
    )

    return {
        "avg_sentence_length": round(
            float(lengths.mean()),
            2,
        ),
        "sentence_length_variation": round(
            float(lengths.std()),
            2,
        ),
        "lexical_diversity": round(
            len(set(words))
            / max(len(words), 1),
            3,
        ),
        "repeated_bigram_ratio": round(
            _ngram_repeat_ratio(
                words,
                2,
            ),
            3,
        ),
        "indonesian_signal": round(
            indonesian_signal(text),
            3,
        ),
    }