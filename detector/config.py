from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "detector.joblib"
MAX_FILE_BYTES = 8 * 1024 * 1024
ALLOWED_EXTENSIONS = {".txt", ".md", ".docx", ".pdf"}
FORMAT_VERSION = 3
MIN_WORDS = 50
SHORT_TEXT_WORDS = 100
TARGET_CHUNK_WORDS = 150
MAX_CHUNK_WORDS = 210
MIN_INDONESIAN_SIGNAL = 0.025