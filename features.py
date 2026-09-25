import re
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

INDONESIAN_FUNCTION_WORDS = {
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "pada", "adalah", "ini",
    "itu", "dalam", "sebagai", "oleh", "karena", "atau", "juga", "akan", "dapat", "tidak",
    "tersebut", "lebih", "telah", "saat", "sehingga", "serta", "antara", "bagi", "terhadap",
    "namun", "jika", "maka", "setelah", "sebelum", "melalui", "tentang", "hingga", "masih",
    "sudah", "bisa", "menjadi", "merupakan", "agar", "bahwa", "secara", "setiap", "para"
}


def _words(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", text.lower(), flags=re.UNICODE)


def _sentences(text: str) -> list[str]:
    items = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
        if sentence.strip()
    ]
    return items or [text.strip()]


class StylometryTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: Iterable[str], y=None):
        return self

    def transform(self, X: Iterable[str]):
        rows = [self._extract(str(text)) for text in X]
        return csr_matrix(np.asarray(rows, dtype=np.float64))

    @staticmethod
    def _extract(text: str) -> list[float]:
        words = _words(text)
        sentences = _sentences(text)

        word_count = max(len(words), 1)
        char_count = max(len(text), 1)
        sentence_count = max(len(sentences), 1)

        sentence_lengths = np.asarray(
            [len(_words(sentence)) for sentence in sentences],
            dtype=np.float64
        )

        word_lengths = (
            np.asarray([len(word) for word in words], dtype=np.float64)
            if words
            else np.asarray([0.0])
        )

        unique_words = set(words)

        frequencies = {}

        for word in words:
            frequencies[word] = frequencies.get(word, 0) + 1

        hapax = sum(
            1
            for count in frequencies.values()
            if count == 1
        )

        punctuation_count = sum(
            1
            for character in text
            if character in ".,;:!?-—()[]{}\"'…"
        )

        digits = sum(
            character.isdigit()
            for character in text
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

        function_words = sum(
            1
            for word in words
            if word in INDONESIAN_FUNCTION_WORDS
        )

        connectors = (
            "selain itu",
            "oleh karena itu",
            "dengan demikian",
            "namun",
            "sementara itu",
            "secara keseluruhan",
            "pada dasarnya",
            "dalam konteks",
            "dapat disimpulkan"
        )

        connector_hits = sum(
            text.lower().count(connector)
            for connector in connectors
        )

        return [
            np.log1p(word_count),
            np.log1p(char_count),
            np.log1p(sentence_count),
            float(sentence_lengths.mean()) if sentence_lengths.size else 0.0,
            float(sentence_lengths.std()) if sentence_lengths.size else 0.0,
            float(word_lengths.mean()),
            float(word_lengths.std()),
            len(unique_words) / word_count,
            hapax / word_count,
            punctuation_count / char_count,
            digits / char_count,
            uppercase / max(len(letters), 1),
            text.count("\n") / char_count,
            text.count(",") / sentence_count,
            text.count(";") / sentence_count,
            (text.count("(") + text.count(")")) / sentence_count,
            (
                text.count('"')
                + text.count("“")
                + text.count("”")
            ) / sentence_count,
            function_words / word_count,
            connector_hits / sentence_count,
        ]