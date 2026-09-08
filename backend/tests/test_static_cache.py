"""Cache headers on the built UI.

The bundler content-hashes everything under /_app, so those files can be
cached forever. The SPA shell that names those hashes must not be: with no
Cache-Control header browsers apply heuristic freshness to it and keep serving
the previous index.html after a UI rebuild. The app then requests chunk
filenames that no longer exist and dies with "Failed to fetch dynamically
imported module" — on a page that a normal reload does not repair.
"""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from lld import main as main_mod


@pytest.fixture
def built_ui(tmp_path, monkeypatch):
    """A minimal STATIC_DIR shaped like a real `vite build` output."""
    static = tmp_path / "static"
    (static / "_app" / "immutable" / "nodes").mkdir(parents=True)
    (static / "index.html").write_text(
        '<script type="module" src="/_app/immutable/nodes/0.abc123.js"></script>'
    )
    (static / "_app" / "immutable" / "nodes" / "0.abc123.js").write_text("export default 1")
    (static / "favicon.png").write_bytes(b"\x89PNG")
    monkeypatch.setattr(main_mod, "STATIC_DIR", static)
    return static


async def _get(app, path: str):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1") as c:
        return await c.get(path)


@pytest.mark.asyncio
async def test_shell_must_revalidate(built_ui):
    app = main_mod.create_app()
    for path in ("/", "/presets", "/models"):
        resp = await _get(app, path)
        assert resp.status_code == 200, path
        assert resp.headers["cache-control"] == "no-cache", path


@pytest.mark.asyncio
async def test_hashed_assets_are_immutable(built_ui):
    app = main_mod.create_app()
    resp = await _get(app, "/_app/immutable/nodes/0.abc123.js")
    assert resp.status_code == 200
    cc = resp.headers["cache-control"]
    assert "immutable" in cc
    assert "max-age=31536000" in cc


@pytest.mark.asyncio
async def test_unhashed_extras_revalidate(built_ui):
    """favicon/manifest keep their names across rebuilds, so pinning them
    would serve the old file indefinitely."""
    app = main_mod.create_app()
    resp = await _get(app, "/favicon.png")
    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "no-cache"


@pytest.mark.asyncio
async def test_unknown_api_path_is_json_not_the_shell(built_ui):
    """An old backend behind a new UI must not answer /api/* with HTML — the
    client would report a parse error instead of "restart the service"."""
    app = main_mod.create_app()
    resp = await _get(app, "/api/does-not-exist")
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("application/json")


# --- the SPA fallback is not a file server ----------------------------------

@pytest.mark.asyncio
async def test_a_traversal_does_not_escape_the_build_directory(built_ui, tmp_path):
    """`full_path` is whatever the client sent: uvicorn does not decode it and
    starlette does not normalise it, so "%2e%2e/%2e%2e/secret" arrives here as
    "../../secret". The old code joined that onto STATIC_DIR and served
    anything `is_file()` said yes to — every file the backend user can read.

    The percent-encoded form is the one that matters: httpx (like a browser)
    collapses a literal "/../" before it ever leaves the client, so only
    "%2e%2e" arrives intact. The plain spellings are here to say so — if a
    future client or proxy stops normalising them, this test already covers it.

    The answer is the SPA shell (200), not a 404: an unknown *page* path is
    exactly what client-side routing is for, and the point is that no file
    outside the build directory is ever the response body.
    """
    secret = tmp_path / "secret.txt"
    secret.write_text("SUPER-SECRET")

    app = main_mod.create_app()
    for path in ("/../secret.txt", "/%2e%2e/secret.txt",
                 "/../../etc/hostname", "/x/../../secret.txt"):
        resp = await _get(app, path)
        assert "SUPER-SECRET" not in resp.text, path
        assert resp.status_code == 200, path
        assert resp.headers["cache-control"] == "no-cache", path


@pytest.mark.asyncio
async def test_a_real_extra_file_is_still_served(built_ui):
    """The traversal check must not cost the unhashed extras their route —
    favicon and friends are served from here, not from the /_app mount."""
    app = main_mod.create_app()
    resp = await _get(app, "/favicon.png")
    assert resp.status_code == 200
    assert resp.content == b"\x89PNG"
