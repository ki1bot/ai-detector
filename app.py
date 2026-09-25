import io
import re
from pathlib import Path
from typing import Optional

import joblib
from docx import Document
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pypdf import PdfReader

MODEL_PATH = Path(
    "models/detector.joblib"
)

MAX_FILE_BYTES = (
    5
    * 1024
    * 1024
)

ALLOWED_EXTENSIONS = {
    ".txt",
    ".md",
    ".docx",
    ".pdf",
}

app = FastAPI(
    title="DeteksiAI Indonesia",
    version="1.0.0",
)

app.mount(
    "/static",
    StaticFiles(
        directory="static"
    ),
    name="static",
)

templates = Jinja2Templates(
    directory="templates"
)

bundle = None


def get_bundle():
    global bundle

    if bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model belum tersedia. "
                    "Jalankan 'python train.py' terlebih dahulu."
                ),
            )

        bundle = joblib.load(
            MODEL_PATH
        )

    return bundle


def count_words(text: str) -> int:
    return len(
        re.findall(
            r"\b[\w'-]+\b",
            text,
            flags=re.UNICODE,
        )
    )


def normalize_text(text: str) -> str:
    text = text.replace(
        "\u00a0",
        " "
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


def extract_text(
    filename: str,
    content: bytes,
) -> str:
    suffix = (
        Path(filename)
        .suffix
        .lower()
    )

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Format file harus "
                "TXT, MD, DOCX, atau PDF."
            ),
        )

    try:
        if suffix in {
            ".txt",
            ".md",
        }:
            return content.decode(
                "utf-8",
                errors="replace",
            )

        if suffix == ".docx":
            document = Document(
                io.BytesIO(content)
            )

            return "\n\n".join(
                paragraph.text
                for paragraph in document.paragraphs
                if paragraph.text.strip()
            )

        reader = PdfReader(
            io.BytesIO(content)
        )

        return "\n\n".join(
            (
                page.extract_text()
                or ""
            )
            for page in reader.pages
        )

    except Exception as exception:
        raise HTTPException(
            status_code=400,
            detail=(
                "Dokumen tidak dapat dibaca: "
                f"{exception}"
            ),
        ) from exception


def chunk_text(
    text: str,
    target_words: int = 170,
    max_words: int = 230,
) -> list[str]:
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(
            r"\n\s*\n",
            text,
        )
        if paragraph.strip()
    ]

    if not paragraphs:
        paragraphs = [text]

    chunks = []

    current = []

    current_words = 0

    def flush():
        nonlocal current
        nonlocal current_words

        if current:
            chunks.append(
                "\n\n".join(
                    current
                ).strip()
            )

            current = []

            current_words = 0

    for paragraph in paragraphs:
        words = paragraph.split()

        if len(words) > max_words:
            flush()

            for start in range(
                0,
                len(words),
                target_words,
            ):
                piece = " ".join(
                    words[
                        start:
                        start + target_words
                    ]
                ).strip()

                if piece:
                    chunks.append(
                        piece
                    )

            continue

        if (
            current
            and current_words + len(words)
            > max_words
        ):
            flush()

        current.append(
            paragraph
        )

        current_words += len(
            words
        )

        if current_words >= target_words:
            flush()

    flush()

    return [
        chunk
        for chunk in chunks
        if chunk
    ]


def classify_score(
    score: float,
) -> tuple[str, str]:
    if score >= 0.75:
        return (
            "Kemungkinan AI",
            "ai",
        )

    if score <= 0.25:
        return (
            "Kemungkinan manusia",
            "human",
        )

    return (
        "Tidak meyakinkan / campuran",
        "uncertain",
    )


def reliability_label(
    word_count: int,
    chunk_scores: list[float],
) -> tuple[str, list[str]]:
    warnings = []

    if word_count < 80:
        warnings.append(
            "Teks sangat pendek; hasil detector cenderung tidak stabil."
        )

    elif word_count < 150:
        warnings.append(
            "Teks cukup pendek; gunakan hasil sebagai indikasi awal saja."
        )

    if len(chunk_scores) >= 2:
        spread = (
            max(chunk_scores)
            - min(chunk_scores)
        )

        if spread >= 0.55:
            warnings.append(
                "Skor antarbagian sangat berbeda; "
                "dokumen mungkin campuran atau gaya tulisnya tidak konsisten."
            )

    if (
        word_count >= 250
        and not warnings
    ):
        return (
            "Lebih baik",
            warnings,
        )

    if word_count >= 150:
        return (
            "Sedang",
            warnings,
        )

    return (
        "Rendah",
        warnings,
    )


def analyze(
    text: str,
) -> dict:
    text = normalize_text(
        text
    )

    words = count_words(
        text
    )

    if words < 25:
        raise HTTPException(
            status_code=400,
            detail=(
                "Teks terlalu pendek. "
                "Masukkan minimal 25 kata."
            ),
        )

    model_bundle = get_bundle()

    model = model_bundle[
        "model"
    ]

    metadata = model_bundle.get(
        "metadata",
        {},
    )

    chunks = chunk_text(
        text
    )

    scores = (
        model
        .predict_proba(
            chunks
        )[:, 1]
        .tolist()
    )

    weights = [
        max(
            count_words(chunk),
            1,
        )
        for chunk in chunks
    ]

    overall_score = (
        sum(
            score * weight
            for score, weight
            in zip(
                scores,
                weights,
            )
        )
        / sum(weights)
    )

    label, label_key = classify_score(
        overall_score
    )

    reliability, warnings = reliability_label(
        words,
        scores,
    )

    sections = []

    for index, (
        chunk,
        score,
    ) in enumerate(
        zip(
            chunks,
            scores,
        ),
        start=1,
    ):
        section_label, section_key = classify_score(
            score
        )

        sections.append(
            {
                "index": index,
                "text": chunk,
                "word_count": count_words(
                    chunk
                ),
                "ai_score": round(
                    score * 100,
                    2,
                ),
                "human_score": round(
                    (1 - score) * 100,
                    2,
                ),
                "label": section_label,
                "label_key": section_key,
            }
        )

    return {
        "label": label,
        "label_key": label_key,
        "ai_score": round(
            overall_score * 100,
            2,
        ),
        "human_score": round(
            (1 - overall_score) * 100,
            2,
        ),
        "word_count": words,
        "character_count": len(
            text
        ),
        "section_count": len(
            sections
        ),
        "reliability": reliability,
        "warnings": warnings,
        "sections": sections,
        "model_metrics": {
            "accuracy": metadata.get(
                "accuracy"
            ),
            "f1": metadata.get(
                "f1"
            ),
            "roc_auc": metadata.get(
                "roc_auc"
            ),
            "samples_test": metadata.get(
                "samples_test"
            ),
            "trained_at": metadata.get(
                "trained_at"
            ),
        },
        "disclaimer": (
            "Skor adalah estimasi model, "
            "bukan bukti pasti siapa penulis dokumen."
        ),
    }


@app.get(
    "/",
    response_class=HTMLResponse,
)
def home(
    request: Request,
):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.get(
    "/api/health"
)
def health():
    return {
        "status": "ok",
        "model_ready": MODEL_PATH.exists(),
    }


@app.post(
    "/api/analyze"
)
async def analyze_endpoint(
    text: Optional[str] = Form(
        default=None
    ),
    file: Optional[UploadFile] = File(
        default=None
    ),
):
    content = text or ""

    if (
        file
        and file.filename
    ):
        raw = await file.read(
            MAX_FILE_BYTES + 1
        )

        if len(raw) > MAX_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Ukuran file maksimal 5 MB."
                ),
            )

        extracted = extract_text(
            file.filename,
            raw,
        )

        if content.strip():
            content = (
                f"{content}\n\n"
                f"{extracted}"
            )
        else:
            content = extracted

    if not content.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "Masukkan teks atau unggah dokumen."
            ),
        )

    return analyze(
        content
    )