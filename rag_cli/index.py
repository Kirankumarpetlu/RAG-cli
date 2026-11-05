import math
import os
import tempfile
from pathlib import Path

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import FastEmbedEmbeddings

from .config import PROJECT_ROOT
from .ingest import load_documents, split_documents


def build_index(settings) -> int:
    documents = load_documents(settings["documents_dir"])
    chunks = split_documents(documents, settings["chunk_size"], settings["chunk_overlap"])
    for chunk in chunks:
        chunk.metadata["embedding_model"] = settings["embedding_model"]
    embeddings = FastEmbedEmbeddings(
        model_name=settings["embedding_model"],
        cache_dir=str(PROJECT_ROOT / "storage" / "models"),
    )
    # shortcut: full-RAM linear search suits small corpora; replace when measured memory/latency requires it.
    store = InMemoryVectorStore(embeddings)
    store.add_documents(chunks)

    destination = settings["index_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
    os.close(descriptor)
    temporary = Path(name)
    try:
        store.dump(str(temporary))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return len(chunks)


def load_index(settings) -> InMemoryVectorStore:
    path = settings["index_path"]
    if not path.is_file():
        raise ValueError("Index missing. Run: python -m rag_cli ingest")
    embeddings = FastEmbedEmbeddings(
        model_name=settings["embedding_model"],
        cache_dir=str(PROJECT_ROOT / "storage" / "models"),
    )
    try:
        store = InMemoryVectorStore.load(str(path), embedding=embeddings)
        records = list(store.store.values())
        compatible = records and all(
            record["metadata"].get("embedding_model") == settings["embedding_model"]
            for record in records
        )
    except Exception as error:
        raise ValueError("Index unreadable or incompatible. Re-run ingestion.") from error
    if not compatible:
        raise ValueError("Index empty or embedding model changed. Re-run ingestion.")
    return store


def retrieve(question, store, top_k, min_score):
    if not question.strip():
        raise ValueError("Question cannot be blank.")
    if top_k < 1 or not math.isfinite(min_score) or not -1 <= min_score <= 1:
        raise ValueError("Invalid retrieval settings.")
    hits = store.similarity_search_with_score(question, k=top_k)
    return [(doc, score) for doc, score in hits if math.isfinite(score) and score >= min_score]
