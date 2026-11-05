import os
import shutil
import stat
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def upload_documents(paths, documents_dir: Path) -> int:
    sources = [Path(path).expanduser() for path in paths]
    targets = set()
    for source in sources:
        if source.suffix.lower() not in {".pdf", ".md"}:
            raise ValueError(f"Only PDF and Markdown files can be uploaded: {source.name}")
        if not source.is_file():
            raise ValueError(f"Upload file does not exist or is not a regular file: {source}")
        target = documents_dir / source.name
        if target in targets or target.exists() or target.is_symlink():
            raise ValueError(f"Document already exists or filename is repeated: {source.name}")
        targets.add(target)

    documents_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        target = documents_dir / source.name
        output = target.open("xb")
        try:
            with output, source.open("rb") as incoming:
                shutil.copyfileobj(incoming, output)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
    return len(sources)


def load_documents(documents_dir: Path) -> list[Document]:
    root = documents_dir.resolve()
    if not root.is_dir():
        raise ValueError("Create documents/ and add PDF or Markdown files.")
    documents = []
    skipped = 0

    def traversal_error(error):
        raise ValueError("Cannot read part of documents/; check permissions.") from error

    for directory, dirs, names in os.walk(root, followlinks=False, onerror=traversal_error):
        safe_dirs = []
        for name in sorted(dirs):
            child = Path(directory) / name
            attributes = child.lstat()
            reparse = getattr(attributes, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
            if child.is_symlink() or reparse:
                skipped += 1
            else:
                safe_dirs.append(name)
        dirs[:] = safe_dirs

        for name in sorted(names):
            path = Path(directory) / name
            suffix = path.suffix.lower()
            if suffix not in {".pdf", ".md"}:
                skipped += 1
                continue
            source = path.relative_to(root).as_posix()
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"Input link escapes documents/: {source}")
            if not path.is_file():
                raise ValueError(f"Input is not a regular file: {source}")
            try:
                loader = (
                    PyPDFLoader(str(path), mode="page")
                    if suffix == ".pdf"
                    else TextLoader(str(path), encoding="utf-8")
                )
                loaded = loader.load()
            except Exception as error:
                raise ValueError(f"Cannot extract text from {source}; check file/encoding.") from error

            useful = [doc for doc in loaded if doc.page_content.strip()]
            if not useful:
                if suffix == ".pdf":
                    raise ValueError(f"No readable text in {source}. OCR may be required.")
                continue
            if len(useful) != len(loaded):
                print(f"Warning: blank extracted pages skipped in {source}; inspect for scanned content.", file=sys.stderr)
            for doc in useful:
                metadata = {"source": source}
                if suffix == ".pdf":
                    page = doc.metadata.get("page")
                    if type(page) is not int or page < 0:
                        raise ValueError(f"PDF page metadata missing in {source}.")
                    metadata["page"] = page
                documents.append(Document(page_content=doc.page_content, metadata=metadata))

    if skipped:
        print(f"Skipped {skipped} unsupported files or linked directories.", file=sys.stderr)
    if not documents:
        raise ValueError("No readable PDF or Markdown documents found.")
    return documents


def split_documents(documents, chunk_size, chunk_overlap) -> list[Document]:
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Require chunk_size > chunk_overlap >= 0.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = [doc for doc in splitter.split_documents(documents) if doc.page_content.strip()]
    if not chunks:
        raise ValueError("No text chunks produced.")
    for number, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = str(number)
    return chunks
