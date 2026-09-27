from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from detector import AnalysisError, analyze_text
from detector.config import BASE_DIR, MAX_FILE_BYTES, MODEL_PATH
from detector.documents import DocumentError, extract_text
from detector.model_store import ModelUnavailableError, model_store

app = FastAPI(title="AI Detector Indonesia", version="3.0.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/api/health")
def health():
    if not MODEL_PATH.exists():
        return {"status": "ok", "model_ready": False, "model_version": None}

    try:
        bundle = model_store.load()
        version = bundle.get("metadata", {}).get("model_version")
        return {"status": "ok", "model_ready": True, "model_version": version}
    except ModelUnavailableError as exception:
        return {
            "status": "ok",
            "model_ready": False,
            "model_version": None,
            "detail": str(exception),
        }


@app.post("/api/analyze")
async def analyze_endpoint(
    text: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
):
    content = text or ""

    if file and file.filename:
        raw = await file.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail="Ukuran file maksimal 8 MB.")

        try:
            extracted = extract_text(file.filename, raw)
        except DocumentError as exception:
            raise HTTPException(status_code=400, detail=str(exception)) from exception

        if not extracted.strip():
            raise HTTPException(
                status_code=400,
                detail=(
                    "Tidak ada teks yang dapat dibaca dari dokumen. "
                    "PDF hasil scan gambar belum didukung."
                ),
            )

        content = f"{content}\n\n{extracted}" if content.strip() else extracted

    if not content.strip():
        raise HTTPException(
            status_code=400,
            detail="Masukkan teks atau pilih dokumen terlebih dahulu.",
        )

    try:
        return analyze_text(content)
    except AnalysisError as exception:
        raise HTTPException(status_code=400, detail=str(exception)) from exception
    except ModelUnavailableError as exception:
        raise HTTPException(status_code=503, detail=str(exception)) from exception