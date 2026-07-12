import httpx

from scieval.providers.health import check_endpoint


def _client(payload: dict, status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(status, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler),
                        base_url="http://test/v1")


def test_healthy():
    r = check_endpoint("http://test/v1", None, "m", client=_client({
        "choices": [{"message": {"content": "hi"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
    }))
    assert r.ok and r.usage_present


def test_missing_usage():
    r = check_endpoint("http://test/v1", None, "m",
                       client=_client({"choices": [{"message": {"content": "x"}}]}))
    assert r.ok and not r.usage_present


def test_http_error():
    r = check_endpoint("http://test/v1", None, "m", client=_client({}, status=500))
    assert not r.ok
