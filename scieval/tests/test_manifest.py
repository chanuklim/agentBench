import json
from pathlib import Path

from scieval.config import load_config
from scieval.runner.manifest import build_manifest, write_manifest
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
