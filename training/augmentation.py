import re

import pandas as pd
from datasets import load_dataset

from detector.text import normalize_text
from training.common import (
    RANDOM_STATE,
    clean_dataframe,
    empty_training_frame,
    stable_hash,
    training_chunks,
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
        r"\bmodel bahasa\b",
        r"\blarge language model\b",
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
    dataset = (
        load_dataset(
            repo_id,
            streaming=True,
        )
        if config is None
        else load_dataset(
            repo_id,
            config,
            streaming=True,
        )
    )

    if "train" in dataset:
        stream = dataset[
            "train"
        ]

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
            stream = (
                stream
                .select_columns(
                    selected_columns
                )
            )

    return stream


def extract_first_text(
    row: dict,
    fields: tuple[str, ...],
) -> str:
    for field in fields:
        value = row.get(
            field
        )

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
) -> pd.DataFrame:
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
            "indonesian-nlp/wikipedia-id",
            None,
            (
                "text",
                "content",
            ),
        ),
        (
            "indonesian-nlp/mc4-id",
            "full",
            ("text",),
        ),
    )

    records: list[dict] = []
    errors: list[str] = []

    quota = max(
        limit
        // max(
            len(candidates),
            1,
        ),
        100,
    )

    for (
        repo_id,
        config,
        fields,
    ) in candidates:
        if len(records) >= limit:
            break

        source_count = 0

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

            for row_index, row in enumerate(
                stream
            ):
                value = extract_first_text(
                    row,
                    fields,
                )

                if (
                    not value
                    or has_explicit_ai_marker(
                        value
                    )
                ):
                    continue

                group_id = (
                    f"human:{repo_id}:"
                    f"{stable_hash(value)}"
                )

                for chunk in training_chunks(
                    value
                ):
                    if has_explicit_ai_marker(
                        chunk
                    ):
                        continue

                    records.append(
                        {
                            "text": chunk,
                            "label": 0,
                            "source": (
                                f"human:{repo_id}"
                            ),
                            "group_id": group_id,
                        }
                    )

                    source_count += 1

                    if (
                        source_count >= quota
                        or len(records) >= limit
                    ):
                        break

                if (
                    source_count >= quota
                    or len(records) >= limit
                ):
                    break

                if row_index > 50000:
                    break

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
        clean_dataframe(
            pd.DataFrame(
                records
            )
        )
        if records
        else empty_training_frame()
    )


def sample_ai_public(
    limit: int,
) -> pd.DataFrame:
    repo_id = (
        "kreasof-ai/percakapan-indo"
    )

    records: list[dict] = []

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

        for row_index, row in enumerate(
            stream
        ):
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

            for (
                message_index,
                message,
            ) in enumerate(
                conversations
            ):
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

                if (
                    not content
                    or has_explicit_ai_marker(
                        content
                    )
                ):
                    continue

                group_id = (
                    f"ai:{repo_id}:"
                    f"{row_index}:"
                    f"{message_index}:"
                    f"{stable_hash(content)}"
                )

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

                    records.append(
                        {
                            "text": chunk,
                            "label": 1,
                            "source": (
                                "ai:gemini-flash-1.5"
                            ),
                            "group_id": group_id,
                        }
                    )

                    if len(records) >= limit:
                        return clean_dataframe(
                            pd.DataFrame(
                                records
                            )
                        )

        return (
            clean_dataframe(
                pd.DataFrame(
                    records
                )
            )
            if records
            else empty_training_frame()
        )

    except Exception as exception:
        print(
            "Peringatan sumber AI publik "
            f"{repo_id}: {exception}"
        )

        return empty_training_frame()


def load_public_augmentation(
    limit_per_class: int,
) -> pd.DataFrame:
    if limit_per_class <= 0:
        return empty_training_frame()

    human = sample_human_public(
        limit_per_class
    )

    ai = sample_ai_public(
        limit_per_class
    )

    usable = min(
        len(human),
        len(ai),
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
            "Peringatan: data augmentasi publik tidak cukup seimbang. "
            f"Human={len(human)}, "
            f"AI={len(ai)}, "
            f"minimum={minimum}."
        )

        return empty_training_frame()

    combined = pd.concat(
        [
            human.iloc[:usable],
            ai.iloc[:usable],
        ],
        ignore_index=True,
    )

    return clean_dataframe(
        combined
    )