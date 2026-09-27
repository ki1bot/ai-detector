from detector.text import chunk_text, count_words, normalize_text, split_sentences


def test_normalize_text() -> None:
    value = normalize_text(
        "Halo\u00a0  dunia.\n\n\nTes"
    )

    assert value == (
        "Halo dunia.\n\nTes"
    )


def test_count_words() -> None:
    assert (
        count_words(
            "Saya sedang menguji AI-detector versi baru."
        )
        == 6
    )


def test_split_sentences() -> None:
    sentences = split_sentences(
        "Satu kalimat. Dua kalimat! Tiga?"
    )

    assert len(sentences) == 3


def test_chunk_text_keeps_content() -> None:
    text = " ".join(
        [
            f"kata{i}"
            for i in range(350)
        ]
    )

    chunks = chunk_text(
        text,
        target_words=100,
        max_words=120,
    )

    assert len(chunks) >= 3

    assert (
        sum(
            count_words(chunk)
            for chunk in chunks
        )
        == 350
    )