from pathlib import Path

import pytest

from rag_cli import config


@pytest.fixture
def configured_environment(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda *args, **kwargs: None)
    for name, value in {
        "GROQ_API_KEY": "test-key",
        "GROQ_MODEL": "test-chat",
        "EMBEDDING_MODEL": "test-embedding",
        "MIN_SCORE": "0.5",
    }.items():
        monkeypatch.setenv(name, value)


def test_settings_are_independent_of_working_directory(configured_environment, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = config.load_settings()
    assert settings["documents_dir"] == Path(config.__file__).resolve().parent.parent / "documents"
    assert settings["api_key"] == "test-key"
    assert settings["min_score"] == 0.5


@pytest.mark.parametrize("score", ["nan", "inf", "-inf", "1.1", "-1.1", "invalid"])
def test_invalid_score(configured_environment, monkeypatch, score):
    monkeypatch.setenv("MIN_SCORE", score)
    with pytest.raises(ValueError, match="MIN_SCORE"):
        config.load_settings()


@pytest.mark.parametrize("value", ["", "replace_with_your_own_key"])
def test_missing_or_placeholder_key(configured_environment, monkeypatch, value):
    monkeypatch.setenv("GROQ_API_KEY", value)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        config.load_settings()


def test_groq_defaults_need_no_openai_key(configured_environment, monkeypatch):
    for name in ("OPENAI_API_KEY", "GROQ_MODEL", "EMBEDDING_MODEL", "MIN_SCORE"):
        monkeypatch.delenv(name, raising=False)
    settings = config.load_settings()
    assert settings["chat_model"] == "openai/gpt-oss-120b"
    assert settings["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert settings["min_score"] == 0.5
