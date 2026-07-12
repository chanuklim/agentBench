import os
import secrets
from datetime import datetime, timezone
from pathlib import Path


def scieval_home() -> Path:
    env = os.environ.get("SCIEVAL_HOME")
    return Path(env) if env else Path.home() / ".scieval"


def new_run_dir(home: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    d = home / "runs" / f"{ts}-{secrets.token_hex(2)}"
    d.mkdir(parents=True, exist_ok=False)
    return d
