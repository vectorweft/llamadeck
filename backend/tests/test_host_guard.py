"""The Host/Origin fence in front of an API with no authentication.

Two attacks reach a loopback-bound service from any web page the user visits,
and both were verified working against this app on 2026-09-07:

* DNS rebinding — `evil.com` re-resolves to 127.0.0.1 and the attacker's own
  JavaScript then reads and writes the local API as same-origin.
* Blind CSRF — a cross-origin form POST is *delivered* regardless of CORS, and
  most mutating endpoints here take no body, so delivery is the whole attack.

Chained with `PUT /api/settings` → `llama_bin` → preset argv, either one is
remote code execution as the user. These tests pin the invariant that stops
them: a request may only address LlamaDeck by literal address (or a name the
user explicitly listed), and a mutating request may only come from such a page.
"""
from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from lld.main import create_app
from lld.net_guard import hostname_of, host_is_allowed


@pytest.fixture()
def app(tmp_path, monkeypatch):
    """A fresh app whose settings live in a throwaway dir.

    net_guard reads settings.json directly (not via load_settings), so the
    module constant is what has to move, and its mtime cache has to be cleared
    between tests that write different files at the same path.
    """
    from lld import net_guard
    from lld import settings as settings_mod

    state = tmp_path / "llamadeck"
    state.mkdir()
    monkeypatch.setattr(settings_mod, "STATE_DIR", state)
    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", state / "settings.json")
    monkeypatch.setattr(settings_mod, "LOGS_DIR", state / "logs")
    monkeypatch.setattr(net_guard, "SETTINGS_PATH", state / "settings.json")
    monkeypatch.setattr(net_guard, "_extra_cache", None)
    monkeypatch.delenv("LLAMADECK_BIND_HOST", raising=False)
    return create_app()


def _client(app, host: str) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url=f"http://{host}")


# --- Host: the rebinding cut ------------------------------------------------

@pytest.mark.asyncio
async def test_a_rebound_domain_is_refused(app):
    """The attack's signature: the request arrives on loopback but carries the
    attacker's domain in Host, because that is what keeps their page
    same-origin with the response."""
    async with _client(app, "evil.com") as c:
        resp = await c.get("/health")
    assert resp.status_code == 403
    assert "evil.com" in resp.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.1:8770", "192.168.1.40:8770",
                                  "[::1]:8770", "localhost:8770", "lld.localhost"])
async def test_the_ways_people_actually_reach_it_still_work(app, host):
    """Loopback, LAN by address, IPv6 and localhost are all untouched — the
    guard must not cost anyone their existing access."""
    async with _client(app, host) as c:
        resp = await c.get("/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_a_name_the_user_listed_is_allowed(app, tmp_path):
    """The escape hatch for "I reach it at http://workstation:8770". Without
    it the guard would be a wall rather than a fence."""
    from lld import net_guard

    net_guard.SETTINGS_PATH.write_text(json.dumps({"allowed_hosts": ["workstation"]}))
    net_guard._extra_cache = None
    async with _client(app, "workstation:8770") as c:
        resp = await c.get("/health")
    assert resp.status_code == 200
    async with _client(app, "evil.com") as c:
        assert (await c.get("/health")).status_code == 403


@pytest.mark.asyncio
async def test_the_static_shell_and_mcp_mount_are_behind_the_guard(app):
    """The middleware is registered last so it wraps everything, including the
    SPA fallback and the /mcp mount. A rebinding page that could still fetch
    the shell would learn the app is there and go on to the API."""
    async with _client(app, "evil.com") as c:
        assert (await c.get("/")).status_code == 403
        assert (await c.get("/mcp")).status_code == 403
        assert (await c.put("/api/settings", json={})).status_code == 403


# --- Origin: the blind-CSRF cut ---------------------------------------------

@pytest.mark.asyncio
async def test_a_cross_origin_form_post_is_refused(app):
    """CORS never sees this one: the browser sends the request and simply
    withholds the response. `/api/system/restart` takes no body, so being sent
    is all the attacker needs."""
    async with _client(app, "127.0.0.1") as c:
        resp = await c.post("/api/system/restart", headers={"origin": "https://evil.com"})
    assert resp.status_code == 403
    assert "evil.com" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_an_opaque_origin_is_refused(app):
    """A sandboxed iframe or a file:// page sends `Origin: null`. Treating that
    as "no origin" would hand back the bypass."""
    async with _client(app, "127.0.0.1") as c:
        resp = await c.post("/api/system/restart", headers={"origin": "null"})
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_the_vite_dev_origin_still_posts(app):
    """`npm run dev` proxies to :8770 with changeOrigin, which rewrites Host
    but leaves Origin as http://localhost:5173. Refusing that would break
    frontend development outright."""
    async with _client(app, "127.0.0.1") as c:
        resp = await c.put("/api/settings", json={},
                           headers={"origin": "http://localhost:5173"})
    # 403 is the only answer this test rules out; the payload itself is empty,
    # so a validation error is a pass.
    assert resp.status_code != 403


@pytest.mark.asyncio
async def test_a_request_with_no_origin_is_left_alone(app):
    """curl, the CLI and MCP clients send no Origin. They are not browsers and
    cannot be driven by a hostile page, so the Host check is their whole
    boundary."""
    async with _client(app, "127.0.0.1") as c:
        resp = await c.get("/api/settings")
    assert resp.status_code == 200


# --- parsing ----------------------------------------------------------------

@pytest.mark.parametrize("raw,want", [
    ("127.0.0.1:8770", "127.0.0.1"),
    ("[::1]:8770", "::1"),
    ("::1", "::1"),
    ("Workstation.Local.", "workstation.local"),
    ("localhost", "localhost"),
])
def test_hostname_parsing(raw, want):
    assert hostname_of(raw) == want


def test_a_missing_host_header_is_not_a_browser():
    """HTTP/1.0 clients omit it. Rebinding cannot: the forged name is the
    mechanism, so there is nothing to catch here."""
    assert host_is_allowed("") is True
