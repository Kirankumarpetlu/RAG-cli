import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from rag_cli import index


class FixedEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0] if "refund" in text.lower() else [0.0, 1.0] for text in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


@pytest.fixture
def settings(tmp_path, monkeypatch):
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "policy.md").write_text("Refund requires a receipt.", encoding="utf-8")
    def local_embeddings(**kwargs):
        assert kwargs["model_name"] == "test-fixed" or kwargs["model_name"] == "another-model"
        assert "api_key" not in kwargs
        return FixedEmbeddings()

    monkeypatch.setattr(index, "FastEmbedEmbeddings", local_embeddings)
    return {
        "documents_dir": documents,
        "index_path": tmp_path / "storage" / "index.json",
        "embedding_model": "test-fixed",
        "chunk_size": 1000,
        "chunk_overlap": 100,
    }


def test_saved_store_and_model_compatibility(settings):
    assert index.build_index(settings) == 1
    store = index.load_index(settings)
    hits = index.retrieve("refund", store, 4, 0.9)
    assert hits[0][0].metadata["source"] == "policy.md"
    assert "receipt" in hits[0][0].page_content
    with pytest.raises(ValueError, match="model changed"):
        index.load_index({**settings, "embedding_model": "another-model"})


def test_failed_save_preserves_old_index(settings, monkeypatch):
    index.build_index(settings)
    original = settings["index_path"].read_bytes()

    def fail_dump(self, path):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(InMemoryVectorStore, "dump", fail_dump)
    with pytest.raises(OSError):
        index.build_index(settings)
    assert settings["index_path"].read_bytes() == original
    assert not list(settings["index_path"].parent.glob("*.tmp"))


def test_rebuild_drops_removed_file(settings):
    index.build_index(settings)
    (settings["documents_dir"] / "policy.md").unlink()
    (settings["documents_dir"] / "new.md").write_text("Delivery takes two days.", encoding="utf-8")
    index.build_index(settings)
    store = index.load_index(settings)
    assert {record["metadata"]["source"] for record in store.store.values()} == {"new.md"}


def test_retrieval_drops_unrelated_vectors():
    store = InMemoryVectorStore(FixedEmbeddings())
    store.add_documents([
        Document(page_content="Refund policy"), Document(page_content="Delivery policy")
    ])
    hits = index.retrieve("refund", store, 4, 0.5)
    assert [doc.page_content for doc, score in hits] == ["Refund policy"]


def test_failed_extraction_preserves_old_index(settings):
    index.build_index(settings)
    original = settings["index_path"].read_bytes()
    (settings["documents_dir"] / "broken.pdf").write_bytes(b"not a PDF")
    with pytest.raises(ValueError, match="broken.pdf"):
        index.build_index(settings)
    assert settings["index_path"].read_bytes() == original


def test_corrupt_index_is_preserved(settings):
    index.build_index(settings)
    settings["index_path"].write_text("broken JSON", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        index.load_index(settings)
    assert settings["index_path"].read_text(encoding="utf-8") == "broken JSON"
