import json
from pathlib import Path

from scieval.config import load_config
from scieval.runner.manifest import _git_sha, build_manifest, write_manifest
from scieval.tasks.catalog import CATALOG

FIXTURE = Path(__file__).parent / "fixtures" / "config"


def _manifest(monkeypatch):
    monkeypatch.setenv("TEST_BASE_URL", "http://localhost:8000/v1")
    cfg = load_config(FIXTURE)
    return build_manifest(
        run_id="r1", suite="toy", budget="small", profile="local",
        models=cfg.models, benchmarks=[CATALOG["gpqa_diamond"], CATALOG["hle_stem"]],
        judges={"hle_stem": "openai/gpt-4o-mini"}, config_dir=FIXTURE,
    )


def test_manifest_contents(monkeypatch):
    m = _manifest(monkeypatch)
    assert m["run_id"] == "r1"
    assert m["scaffold"] == "native-v1"
    assert m["versions"]["inspect_ai"]
    assert m["benchmarks"]["hle_stem"]["judge"] == "openai/gpt-4o-mini"
    assert m["benchmarks"]["hle_stem"]["dataset_revision"].startswith("5a81a4c")
    assert len(m["config_sha256"]) == 64


def test_manifest_no_absolute_paths(monkeypatch, tmp_path):
    m = _manifest(monkeypatch)
    text = json.dumps(m)
    assert str(Path.home()) not in text
    p = write_manifest(tmp_path, m)
    assert p == tmp_path / "manifest.json"
    assert json.loads(p.read_text())["run_id"] == "r1"


def test_git_sha_independent_of_caller_cwd(monkeypatch, tmp_path):
    """_git_sha() must resolve THIS package's repo sha regardless of the
    caller's process cwd -- it pins cwd to this package's own directory
    rather than trusting whatever directory the run happened to start in."""
    monkeypatch.chdir(tmp_path)  # tmp_path is not inside any git repo
    sha = _git_sha()
    assert sha != "unknown"
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)


def test_manifest_git_sha_present_when_invoked_from_unrelated_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    m = _manifest(monkeypatch)
    assert m["git_sha"] != "unknown"
    assert len(m["git_sha"]) == 40


def test_write_manifest_uses_utf8_encoding(monkeypatch, tmp_path):
    m = _manifest(monkeypatch)
    m["profile"] = "로컬-프로파일"  # non-ASCII, would mis-decode under a non-utf8 default
    p = write_manifest(tmp_path, m)
    raw = p.read_bytes()
    assert json.loads(raw.decode("utf-8"))["profile"] == "로컬-프로파일"
