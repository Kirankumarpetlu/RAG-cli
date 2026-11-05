import pytest
from langchain_core.documents import Document

from rag_cli import ingest


def test_markdown_and_suffix_filter(tmp_path):
    nested = tmp_path / "manuals"
    nested.mkdir()
    (nested / "POLICY.MD").write_text("Refunds require a receipt.", encoding="utf-8")
    (nested / "ignore.txt").write_text("Do not ingest this.", encoding="utf-8")
    documents = ingest.load_documents(tmp_path)
    assert len(documents) == 1
    assert documents[0].metadata == {"source": "manuals/POLICY.MD"}
    assert "receipt" in documents[0].page_content


def test_pdf_page_metadata_survives_splitting(tmp_path, monkeypatch):
    (tmp_path / "guide.PDF").write_bytes(b"mock PDF")

    class Loader:
        def __init__(self, path, mode):
            assert mode == "page"

        def load(self):
            return [Document(page_content="Refund policy. " * 20, metadata={"page": 2})]

    monkeypatch.setattr(ingest, "PyPDFLoader", Loader)
    chunks = ingest.split_documents(ingest.load_documents(tmp_path), 80, 10)
    assert len(chunks) > 1
    assert all(doc.metadata["page"] == 2 for doc in chunks)
    assert all(doc.metadata["source"] == "guide.PDF" for doc in chunks)
    assert len({doc.metadata["chunk_id"] for doc in chunks}) == len(chunks)


def test_textless_pdf_fails(tmp_path, monkeypatch):
    (tmp_path / "scan.pdf").write_bytes(b"mock PDF")

    class Loader:
        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            return [Document(page_content="  ", metadata={"page": 0})]

    monkeypatch.setattr(ingest, "PyPDFLoader", Loader)
    with pytest.raises(ValueError, match="OCR"):
        ingest.load_documents(tmp_path)


def test_escaping_file_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "escape.md"
    path.write_text("secret", encoding="utf-8")
    path_class = type(path)
    original_resolve = path_class.resolve

    def resolve(candidate, *args, **kwargs):
        if candidate == path:
            return tmp_path.parent / "outside.md"
        return original_resolve(candidate, *args, **kwargs)

    monkeypatch.setattr(path_class, "resolve", resolve)
    with pytest.raises(ValueError, match="escapes"):
        ingest.load_documents(tmp_path)


def test_invalid_chunk_overlap():
    with pytest.raises(ValueError):
        ingest.split_documents([Document(page_content="text")], 10, 10)


def test_blank_markdown_is_skipped(tmp_path):
    (tmp_path / "blank.md").write_text(" \n", encoding="utf-8")
    (tmp_path / "policy.md").write_text("Refunds require a receipt.", encoding="utf-8")
    assert [doc.metadata["source"] for doc in ingest.load_documents(tmp_path)] == ["policy.md"]


def test_real_pdf_extraction(tmp_path):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)}),
    })
    content = DecodedStreamObject()
    content.set_data(b"BT /F1 12 Tf 72 720 Td (Refunds require a receipt.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(content)
    writer.write(tmp_path / "policy.pdf")
    documents = ingest.load_documents(tmp_path)
    assert documents[0].metadata == {"source": "policy.pdf", "page": 0}
    assert "Refunds require a receipt." in documents[0].page_content


def test_upload_multiple_files_preserves_originals(tmp_path):
    sources = [tmp_path / "policy.MD", tmp_path / "guide.PDF"]
    for source in sources:
        source.write_bytes(b"original contents")
    destination = tmp_path / "documents"
    assert ingest.upload_documents(sources, destination) == 2
    for source in sources:
        assert source.read_bytes() == (destination / source.name).read_bytes()


@pytest.mark.parametrize("problem", ["unsupported", "missing", "existing", "duplicate"])
def test_upload_rejects_invalid_batch_before_copying(tmp_path, problem):
    source = tmp_path / "policy.md"
    source.write_text("new policy", encoding="utf-8")
    destination = tmp_path / "documents"
    destination.mkdir()
    bad = tmp_path / "missing.md"
    if problem == "unsupported":
        bad = tmp_path / "unsupported.txt"
        bad.write_text("unsupported", encoding="utf-8")
    elif problem == "existing":
        (destination / source.name).write_text("existing policy", encoding="utf-8")
        bad = source
    elif problem == "duplicate":
        bad = source
    with pytest.raises(ValueError):
        ingest.upload_documents([source, bad], destination)
    if problem == "existing":
        assert (destination / source.name).read_text(encoding="utf-8") == "existing policy"
    else:
        assert not list(destination.iterdir())


def test_failed_upload_removes_partial_copy(tmp_path, monkeypatch):
    source = tmp_path / "policy.md"
    source.write_bytes(b"original contents")
    destination = tmp_path / "documents"

    def fail_copy(incoming, output):
        output.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(ingest.shutil, "copyfileobj", fail_copy)
    with pytest.raises(OSError, match="disk full"):
        ingest.upload_documents([source], destination)
    assert not list(destination.iterdir())
    assert source.read_bytes() == b"original contents"
