import math
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_settings() -> dict:
    root = PROJECT_ROOT
    load_dotenv(root / ".env", override=False)
    values = {}
    for name, default in {
        "GROQ_API_KEY": "",
        "GROQ_MODEL": "openai/gpt-oss-120b",
        "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
        "MIN_SCORE": "0.5",
    }.items():
        value = os.getenv(name, default).strip()
        if not value or value.startswith("replace_with"):
            raise ValueError(f"Set {name} in .env or your shell.")
        values[name] = value

    try:
        min_score = float(values["MIN_SCORE"])
    except ValueError:
        raise ValueError("MIN_SCORE must be a number between -1 and 1.") from None
    if not math.isfinite(min_score) or not -1 <= min_score <= 1:
        raise ValueError("MIN_SCORE must be finite and between -1 and 1.")

    documents = root / "documents"
    if not documents.resolve().is_relative_to(root):
        raise ValueError("documents/ must stay inside the project.")
    return {
        "documents_dir": documents,
        "index_path": root / "storage" / "index.json",
        "api_key": values["GROQ_API_KEY"],
        "chat_model": values["GROQ_MODEL"],
        "embedding_model": values["EMBEDDING_MODEL"],
        "min_score": min_score,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 4,
    }
