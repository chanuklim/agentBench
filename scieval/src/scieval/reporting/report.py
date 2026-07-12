import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from scieval.scoring.aggregate import AxisScore, IndexResult


@dataclass(frozen=True)
class ModelReport:
    axes: dict[str, AxisScore]
    index: IndexResult
    gates: dict[str, str]
    normalized: dict[str, float]


def render_report(run_dir: Path, results_df: pd.DataFrame,
                  per_model: dict[str, ModelReport]) -> Path:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"),
                      autoescape=True)
    html = env.get_template("report.html.j2").render(
        manifest=manifest,
        manifest_json=json.dumps(manifest, indent=2, ensure_ascii=False),
        per_model=per_model,
        rows=results_df.to_dict("records"),
    )
    p = run_dir / "report.html"
    p.write_text(html, encoding="utf-8")
    return p
