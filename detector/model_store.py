from threading import Lock

import joblib

from detector.config import FORMAT_VERSION, MODEL_PATH


class ModelUnavailableError(RuntimeError):
    pass


class ModelStore:
    def __init__(self) -> None:
        self._bundle: dict | None = None
        self._lock = Lock()

    def load(self) -> dict:
        if self._bundle is not None:
            return self._bundle

        with self._lock:
            if self._bundle is not None:
                return self._bundle

            if not MODEL_PATH.exists():
                raise ModelUnavailableError(
                    "Model belum tersedia. "
                    "Jalankan python train.py terlebih dahulu."
                )

            bundle = joblib.load(MODEL_PATH)

            if bundle.get("format_version") != FORMAT_VERSION:
                raise ModelUnavailableError(
                    "Model tidak kompatibel dengan kode saat ini. "
                    "Jalankan python train.py lagi."
                )

            self._bundle = bundle
            return bundle

    def clear(self) -> None:
        with self._lock:
            self._bundle = None


model_store = ModelStore()