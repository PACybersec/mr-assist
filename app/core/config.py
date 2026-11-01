
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = APP_DIR / "data" / "knowledge"
DATABASE_DIR = APP_DIR / "data" / "chroma_db"

EMBEDDING_MODEL = "nomic-embed-text"
COLLECTION_NAME = "it_knowledge"
DEFAULT_MAX_DISTANCE = 0.35
DEFAULT_MAX_CHARS = 800
DEFAULT_N_RESULTS = 3
