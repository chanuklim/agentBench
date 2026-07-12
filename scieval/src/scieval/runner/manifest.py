import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

from scieval.config import ModelConfig
from scieval.tasks.spec import BenchmarkSpec

SCAFFOLD_ID = "native-v1"  # M1: 벤치마크 네이티브 solver 사용


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _config_sha256(config_dir: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(config_dir.glob("*.yaml")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def build_manifest(
    run_id: str,
    suite: str,
    budget: str,
    profile: str,
    models: dict[str, ModelConfig],
    benchmarks: list[BenchmarkSpec],
    judges: dict[str, str],
    config_dir: Path,
) -> dict:
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite": suite,
        "budget": budget,
        "profile": profile,
        "scaffold": SCAFFOLD_ID,
        "models": {
            mid: {
                "inspect_model": m.inspect_model,
                "base_url_env": m.base_url_env,
                "api_key_env": m.api_key_env,
            }
            for mid, m in models.items()
        },
        "benchmarks": {
            b.id: {
                "task_name": b.task_name,
                "dataset_revision": b.dataset_revision,
                "judge": judges.get(b.id),
            }
            for b in benchmarks
        },
        "versions": {
            "python": sys.version.split()[0],
            "scieval": pkg_version("scieval"),
            "inspect_ai": pkg_version("inspect-ai"),
            "inspect_evals": pkg_version("inspect-evals"),
        },
        "git_sha": _git_sha(),
        "config_sha256": _config_sha256(config_dir),
    }


def write_manifest(run_dir: Path, m: dict) -> Path:
    p = run_dir / "manifest.json"
    p.write_text(json.dumps(m, indent=2, ensure_ascii=False))
    return p
