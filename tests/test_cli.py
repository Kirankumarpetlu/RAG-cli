from rag_cli import __main__ as cli


def test_ingest_command(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_settings", lambda: {"configured": True})

    def build(settings):
        assert settings == {"configured": True}
        return 3

    monkeypatch.setattr(cli, "build_index", build)
    assert cli.main(["ingest"]) == 0
    assert capsys.readouterr().out == "Indexed 3 chunks.\n"


def test_ask_prints_cited_sources(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_settings", lambda: {})
    monkeypatch.setattr(cli, "load_index", lambda settings: "saved-store")

    def respond(question, store, settings):
        assert question == "What is required?"
        assert store == "saved-store"
        return {"answer": "A receipt. [1]", "sources": [{"id": 1, "label": "policy.md; chunk=0"}]}

    monkeypatch.setattr(cli, "answer_question", respond)
    assert cli.main(["ask", "What is required?"]) == 0
    assert capsys.readouterr().out == "A receipt. [1]\n\nSources:\n[1] policy.md; chunk=0\n"


def test_provider_error_does_not_print_sensitive_details(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_settings", lambda: {})

    def fail(settings):
        raise RuntimeError("sensitive request details")

    monkeypatch.setattr(cli, "build_index", fail)
    assert cli.main(["ingest"]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert "RuntimeError" in output.err
    assert "sensitive" not in output.err


def test_upload_does_not_require_provider_settings(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "policy.md"
    source.write_text("Refund policy", encoding="utf-8")

    def unexpected_settings():
        raise AssertionError("Upload must not require provider settings")

    monkeypatch.setattr(cli, "load_settings", unexpected_settings)
    assert cli.main(["upload", str(source)]) == 0
    assert (tmp_path / "documents" / "policy.md").read_text(encoding="utf-8") == "Refund policy"
    assert "python -m rag_cli ingest" in capsys.readouterr().out


def test_interactive_indexes_only_selected_file_and_reuses_store(tmp_path, monkeypatch, capsys):
    source = tmp_path / "policy with spaces.md"
    source.write_text("Refund policy", encoding="utf-8")
    persistent_index = tmp_path / "index.json"
    persistent_index.write_text("existing index", encoding="utf-8")
    inputs = iter([f'"{source}"', "", "What is required?", "What else?", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    monkeypatch.setattr(cli, "load_settings", lambda: {"index_path": persistent_index})
    builds = []
    loads = []

    def build(settings):
        builds.append(settings)
        assert list(settings["documents_dir"].iterdir()) == [settings["documents_dir"] / source.name]
        assert (settings["documents_dir"] / source.name).read_text(encoding="utf-8") == "Refund policy"
        assert settings["index_path"] != persistent_index
        return 1

    def load(settings):
        loads.append(settings)
        return "session-store"

    questions = []

    def respond(question, store, settings):
        questions.append(question)
        assert store == "session-store"
        return {"answer": "A receipt. [1]", "sources": [{"id": 1, "label": source.name}]}

    monkeypatch.setattr(cli, "build_index", build)
    monkeypatch.setattr(cli, "load_index", load)
    monkeypatch.setattr(cli, "answer_question", respond)
    assert cli.main([]) == 0
    assert len(builds) == len(loads) == 1
    assert questions == ["What is required?", "What else?"]
    assert persistent_index.read_text(encoding="utf-8") == "existing index"
    assert not builds[0]["documents_dir"].exists()
    assert capsys.readouterr().out.count("A receipt. [1]") == 2


def test_interactive_invalid_path_can_be_retried_and_exited(tmp_path, monkeypatch, capsys):
    inputs = iter([str(tmp_path / "missing.pdf"), "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    assert cli.main([]) == 0
    assert "does not exist" in capsys.readouterr().err


def test_interactive_file_switch_cleans_previous_session(tmp_path, monkeypatch):
    sources = [tmp_path / "first.md", tmp_path / "second.md"]
    for source in sources:
        source.write_text("Evidence", encoding="utf-8")
    inputs = iter([str(sources[0]), "/file", str(sources[1]), "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    monkeypatch.setattr(cli, "load_settings", lambda: {})
    sessions = []

    def build(settings):
        if sessions:
            assert not sessions[-1].exists()
        sessions.append(settings["documents_dir"])
        assert (sessions[-1] / sources[len(sessions) - 1].name).is_file()
        return 1

    monkeypatch.setattr(cli, "build_index", build)
    monkeypatch.setattr(cli, "load_index", lambda settings: None)
    assert cli.main([]) == 0
    assert len(sessions) == 2
    assert not any(path.exists() for path in sessions)


def test_interactive_eof_exits_cleanly(monkeypatch):
    def end_input(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", end_input)
    assert cli.main([]) == 0
