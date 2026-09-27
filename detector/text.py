import re
import unicodedata

WORD_PATTERN = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text))
    value = value.replace("\u00a0", " ").replace("\u200b", " ").replace("\ufeff", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def tokenize_words(text: str) -> list[str]:
    return WORD_PATTERN.findall(text.lower())


def count_words(text: str) -> int:
    return len(tokenize_words(text))


def split_sentences(text: str) -> list[str]:
    value = normalize_text(text)
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+|\n+", value)
        if item.strip()
    ]
    return sentences or ([value] if value else [])


def split_paragraphs(text: str) -> list[str]:
    value = normalize_text(text)
    paragraphs = [
        item.strip()
        for item in re.split(r"\n\s*\n", value)
        if item.strip()
    ]
    return paragraphs or ([value] if value else [])


def chunk_text(
    text: str,
    target_words: int = 150,
    max_words: int = 210,
) -> list[str]:
    value = normalize_text(text)
    paragraphs = split_paragraphs(value)
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words

        if current:
            chunk = " ".join(current).strip()
            if chunk:
                chunks.append(chunk)

        current = []
        current_words = 0

    for paragraph in paragraphs:
        sentences = split_sentences(paragraph) or [paragraph]

        for sentence in sentences:
            words = tokenize_words(sentence)

            if not words:
                continue

            if len(words) > max_words:
                flush()
                raw_words = sentence.split()

                for start in range(0, len(raw_words), target_words):
                    piece = " ".join(
                        raw_words[start : start + target_words]
                    ).strip()

                    if piece:
                        chunks.append(piece)

                continue

            if current and current_words + len(words) > max_words:
                flush()

            current.append(sentence)
            current_words += len(words)

            if current_words >= target_words:
                flush()

    flush()

    return chunks or ([value] if value else [])