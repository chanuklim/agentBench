from scieval.runner.cli import _missing_judge_env_vars


def test_missing_judge_env_vars_reports_unset_known_provider(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    missing = _missing_judge_env_vars(["openai/gpt-5-mini", "anthropic/claude-x"])
    assert missing == ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]


def test_missing_judge_env_vars_empty_when_all_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    missing = _missing_judge_env_vars(["openai/gpt-5-mini"])
    assert missing == []


def test_missing_judge_env_vars_skips_unknown_provider(monkeypatch):
    monkeypatch.delenv("SOME_OTHER_KEY", raising=False)
    missing = _missing_judge_env_vars(["mycompany/internal-judge"])
    assert missing == []


def test_missing_judge_env_vars_dedupes_same_var(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    missing = _missing_judge_env_vars(["openai/gpt-5-mini", "openai/gpt-5"])
    assert missing == ["OPENAI_API_KEY"]


def test_missing_judge_env_vars_no_judges():
    assert _missing_judge_env_vars([]) == []
