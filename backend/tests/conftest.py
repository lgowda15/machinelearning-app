"""Shared pytest fixtures for the API integration tests.

`client` must enter TestClient as a context manager. Returned bare
(`TestClient(app)` with no `with`), each `client.post()`/`.get()` call opens
its own anyio blocking portal -- and thus its own asyncio event loop and
raw OS socketpair -- and tears it down again, for every single request.
Under Windows sandboxing that repeated socket creation intermittently raises
`PermissionError: [WinError 10013]`, unrelated to whatever the test is
actually checking. `scope="session"` enters the `with` block once for the
entire test run instead of once per test, cutting the number of socketpair()
attempts from ~1 per test to 1 total.

That single remaining attempt can still hit the same transient denial, so
retry only that specific failure -- a bounded loop around the
`TestClient.__enter__`, not a blanket rerun-on-any-failure decorator around
the tests themselves, which would also silently retry genuine assertion
failures.
"""
import time

import pytest
from starlette.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    last_exc = None
    for attempt in range(3):
        try:
            with TestClient(app) as c:
                yield c
                return
        except PermissionError as e:
            last_exc = e
            time.sleep(0.5 * (attempt + 1))
    raise last_exc
