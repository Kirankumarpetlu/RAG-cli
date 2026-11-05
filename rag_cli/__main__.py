import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from .answer import answer_question
from .config import PROJECT_ROOT, load_settings
from .index import build_index, load_index
from .ingest import upload_documents


def print_answer(result):
    print(result["answer"])
    if result["sources"]:
        print("\nSources:")
        for source in result["sources"]:
            print(f"[{source['id']}] {source['label']}")


def interactive() -> int:
    print("Document Q&A: paste a PDF or Markdown file path. Type exit to quit.")
    while True:
        path = input("File path: ").strip()
        if path.lower() in {"exit", "quit"}:
            return 0
        if len(path) >= 2 and path[0] == path[-1] and path[0] in {"'", '"'}:
            path = path[1:-1]
        with TemporaryDirectory(prefix="rag-cli-") as directory:
            try:
                upload_documents([path], Path(directory))
            except ValueError as error:
                print(f"Error: {error}", file=sys.stderr)
                continue
            settings = {
                **load_settings(),
                "documents_dir": Path(directory),
                "index_path": Path(directory) / "index.json",
            }
            print("Indexing your document...")
            count = build_index(settings)
            store = load_index(settings)
            print(f"Ready ({count} chunks). Ask questions, type /file to change the document, or exit to quit.")
            while True:
                question = input("Question: ").strip()
                if question.lower() in {"exit", "quit"}:
                    return 0
                if question == "/file":
                    break
                if not question:
                    continue
                print_answer(answer_question(question, store, settings))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Answer from local PDF and Markdown files. Run without arguments for interactive Q&A."
    )
    commands = parser.add_subparsers(dest="command")
    upload = commands.add_parser("upload", help="Copy PDF or Markdown files into documents/ without overwriting.")
    upload.add_argument("files", nargs="+", help="Paths to one or more PDF or Markdown files.")
    commands.add_parser("ingest", help="Rebuild the index from documents/.")
    ask = commands.add_parser("ask", help="Ask one question using the saved index.")
    ask.add_argument("question")
    args = parser.parse_args(argv)
    try:
        if args.command is None:
            return interactive()
        if args.command == "upload":
            documents_dir = PROJECT_ROOT / "documents"
            if not documents_dir.resolve().is_relative_to(PROJECT_ROOT):
                raise ValueError("documents/ must stay inside the project.")
            count = upload_documents(args.files, documents_dir)
            print(f"Uploaded {count} files to documents/. Run: python -m rag_cli ingest")
            return 0
        settings = load_settings()
        if args.command == "ingest":
            count = build_index(settings)
            print(f"Indexed {count} chunks.")
        else:
            if not args.question.strip():
                raise ValueError("Question cannot be blank.")
            result = answer_question(args.question, load_index(settings), settings)
            print_answer(result)
        return 0
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except OSError:
        print("File access failed; check paths, permissions, and free disk space.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    except EOFError:
        print("\nGoodbye.")
        return 0
    except Exception as error:
        print(f"Operation failed ({type(error).__name__}); check model access and connection.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
