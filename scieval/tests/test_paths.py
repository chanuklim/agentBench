from pathlib import Path

from scieval.paths import new_run_dir, scieval_home


def test_home_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SCIEVAL_HOME", str(tmp_path / "h"))
    assert scieval_home() == tmp_path / "h"


def test_home_default(monkeypatch):
    monkeypatch.delenv("SCIEVAL_HOME", raising=False)
    assert scieval_home() == Path.home() / ".scieval"


def test_new_run_dir(tmp_path):
    d = new_run_dir(tmp_path)
    assert d.is_dir() and d.parent == tmp_path / "runs"
    assert d != new_run_dir(tmp_path)
