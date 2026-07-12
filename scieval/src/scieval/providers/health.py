from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class HealthReport:
    ok: bool
    usage_present: bool
    detail: str


def check_endpoint(
    base_url: str, api_key: str | None, model: str, client: httpx.Client | None = None
) -> HealthReport:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    own = client is None
    client = client or httpx.Client(base_url=base_url, timeout=30.0)
    try:
        resp = client.post(
            "/chat/completions",
            json={"model": model, "max_tokens": 1,
                  "messages": [{"role": "user", "content": "ping"}]},
            headers=headers,
        )
        if resp.status_code != 200:
            return HealthReport(False, False, f"HTTP {resp.status_code}: {resp.text[:200]}")
        body = resp.json()
        if not body.get("choices"):
            return HealthReport(False, False, "no choices in response")
        usage_present = "completion_tokens" in (body.get("usage") or {})
        return HealthReport(True, usage_present, "ok")
    except httpx.HTTPError as e:
        return HealthReport(False, False, f"connection error: {e}")
    finally:
        if own:
            client.close()
