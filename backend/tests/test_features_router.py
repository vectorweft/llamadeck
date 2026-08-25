"""Local model (llama.cpp router) summarization + scan cleanup.

Two real failures on a box where What's New pointed at a llama.cpp ROUTER:

  * /props answers ``default_generation_settings.n_ctx = 0`` — a router has no
    single default context. probe_endpoint used to take that at face value and
    report ``n_ctx=None``, so the adaptive prompt fitting was skipped entirely
    and the server returned a raw 400 "request (24k tokens) exceeds the
    available context size (3072 tokens)".
  * /models serves a whole catalog, so an empty configured model id raised
    "pick one in Settings" even when exactly one model was loaded and ready.

select_model() prefers the loaded model, and probe_endpoint() now derives the
context from the loaded model's meta.n_ctx.
"""
from __future__ import annotations

import pytest

from lld import features
from lld.db import connect, init_db
from lld.features import FeatureError, FeatureTracker, select_model


# ---------- probe_endpoint on a router ----------

class _FakeResp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


def _router_props():
    return {"role": "router", "default_generation_settings": {"n_ctx": 0}}


def _models_payload(loaded_id, loaded_ctx, others):
    data = []
    for mid, ctx in others:
        data.append({"id": mid, "status": {"value": "unloaded"}})
    data.append({
        "id": loaded_id,
        "status": {"value": "loaded"},
        "meta": {"n_ctx": loaded_ctx},
    })
    return {"data": data}


def _fake_client_for(routes):
    """routes: dict url -> _FakeResp. Hands back a stand-in for
    httpx.AsyncClient that answers GETs from the map."""
    class _FakeClient:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            for route, resp in routes.items():
                if url.startswith(route):
                    return resp
            return _FakeResp(404, {})

    return _FakeClient


@pytest.mark.asyncio
async def test_probe_endpoint_derives_n_ctx_from_the_loaded_model(monkeypatch):
    """The router's own /props says n_ctx=0 — the context has to come from the
    loaded model's meta. Otherwise the fit-check is skipped and a 400 happens
    later."""
    routes = {
        "http://local/models": _FakeResp(200, _models_payload(
            "qwen3.8-27b", 132_096, [("a-1b", 3072), ("b-2b", 8192)])),
        "http://local/props": _FakeResp(200, _router_props()),
    }
    monkeypatch.setattr(features, "_probe_cache", {})
    monkeypatch.setattr(features.httpx, "AsyncClient", _fake_client_for(routes))

    info = await features.probe_endpoint("http://local")
    assert info["reachable"] and info["native"]
    assert info["loaded"] == "qwen3.8-27b"
    assert info["n_ctx"] == 132_096
    assert info["n_ctx_by_model"]["qwen3.8-27b"] == 132_096


@pytest.mark.asyncio
async def test_probe_endpoint_uses_props_n_ctx_when_present(monkeypatch):
    """A plain server still reports its own context; we must not prefer the
    loaded-model fallback over a real /props answer."""
    routes = {
        "http://local/models": _FakeResp(200, _models_payload("only", 8192, [])),
        "http://local/props": _FakeResp(200, {"n_ctx": 64000}),
    }
    monkeypatch.setattr(features, "_probe_cache", {})
    monkeypatch.setattr(features.httpx, "AsyncClient", _fake_client_for(routes))
    assert (await features.probe_endpoint("http://local"))["n_ctx"] == 64000


# ---------- select_model ----------

def test_select_model_prefers_the_loaded_model_when_auto():
    info = {"models": ["a-1b", "b-2b", "qwen3.8-27b"], "loaded": "qwen3.8-27b"}
    assert select_model("", info) == "qwen3.8-27b"


def test_select_model_explicit_name_wins_even_with_a_loaded_model():
    info = {"models": ["a-1b", "qwen3.8-27b"], "loaded": "qwen3.8-27b"}
    assert select_model("a-1b", info) == "a-1b"


def test_select_model_no_loaded_model_and_several_asks_for_one():
    info = {"models": ["a-1b", "b-2b"], "loaded": None}
    with pytest.raises(FeatureError, match="pick one"):
        select_model("", info)


def test_select_model_single_served_model_takes_it():
    info = {"models": ["only"], "loaded": "only"}
    assert select_model("", info) == "only"


# ---------- delete_scan / delete_scans ----------

@pytest.mark.asyncio
async def _seed_scan(status="failed"):
    await init_db()
    async with connect() as db:
        cur = await db.execute(
            "INSERT INTO feature_scans (created_at, from_commit, to_commit, "
            "build_number, status, error) VALUES (?, ?, ?, ?, ?, ?)",
            (1.0, "aaa", "bbb", 123, status, "boom"),
        )
        scan_id = cur.lastrowid
        await db.execute(
            "INSERT INTO release_features (scan_id, created_at, title_tr, what_tr, "
            "how_tr, why_tr, flags_json, architectures_json, source_urls_json, "
            "confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (scan_id, 1.0, "t", "w", "h", "y", "[]", "[]", "[]", "high"),
        )
        await db.execute(
            "INSERT INTO feature_ab_runs (feature_id, model_path, flags_json, "
            "created_at, status) VALUES ((SELECT id FROM release_features "
            "WHERE scan_id=?), ?, ?, ?, ?)",
            (scan_id, "/m/x.gguf", "{}", 1.0, "done"),
        )
        await db.commit()
    return scan_id


@pytest.mark.asyncio
async def test_delete_scan_removes_cards_and_ab_runs():
    scan_id = await _seed_scan()
    assert await FeatureTracker().delete_scan(scan_id)
    async with connect() as db:
        assert (await (await db.execute(
            "SELECT COUNT(*) AS n FROM feature_scans WHERE id=?", (scan_id,)
        )).fetchone())["n"] == 0
        assert (await (await db.execute(
            "SELECT COUNT(*) AS n FROM release_features WHERE scan_id=?", (scan_id,)
        )).fetchone())["n"] == 0
        assert (await (await db.execute(
            "SELECT COUNT(*) AS n FROM feature_ab_runs"
        )).fetchone())["n"] == 0


@pytest.mark.asyncio
async def test_delete_scans_filters_by_status():
    await _seed_scan("failed")
    await _seed_scan("summarized")
    await _seed_scan("failed")
    n = await FeatureTracker().delete_scans(status="failed")
    assert n == 2
    async with connect() as db:
        rem = await db.execute("SELECT status FROM feature_scans")
        statuses = [r["status"] for r in await rem.fetchall()]
    assert statuses == ["summarized"]


@pytest.mark.asyncio
async def test_deleting_a_scan_that_does_not_exist_says_so():
    """The API turns False into a 404. Returning True unconditionally made
    that branch unreachable, so deleting nothing reported success."""
    await init_db()
    assert await FeatureTracker().delete_scan(999_999) is False


# ---------- probe_endpoint: what "reachable" means ----------

@pytest.mark.asyncio
async def test_an_endpoint_with_an_empty_catalogue_is_still_reachable(monkeypatch):
    """A 200 from /models is proof something is listening. Treating an empty
    catalogue as unreachable is what turns "no models loaded" into "nothing
    answered — is the server running?" on the settings screen."""
    routes = {
        "http://local/models": _FakeResp(200, {"data": []}),
        "http://local/props": _FakeResp(404, {}),
    }
    monkeypatch.setattr(features, "_probe_cache", {})
    monkeypatch.setattr(features.httpx, "AsyncClient", _fake_client_for(routes))
    info = await features.probe_endpoint("http://local")
    assert info["reachable"] is True
    assert info["models"] == []


@pytest.mark.asyncio
async def test_probe_endpoint_reads_the_models_name_shape(monkeypatch):
    """Not every server answers the OpenAI `data[].id` shape; some list
    `models[].name`. Both have to be understood or the probe reports a live
    endpoint as serving nothing."""
    routes = {
        "http://local/models": _FakeResp(200, {"models": [
            {"name": "gemma-3-27b"}, {"name": "qwen3.8-27b"},
        ]}),
        "http://local/props": _FakeResp(404, {}),
    }
    monkeypatch.setattr(features, "_probe_cache", {})
    monkeypatch.setattr(features.httpx, "AsyncClient", _fake_client_for(routes))
    info = await features.probe_endpoint("http://local")
    assert info["models"] == ["gemma-3-27b", "qwen3.8-27b"]
    assert info["reachable"] is True


@pytest.mark.asyncio
async def test_a_model_list_is_never_collected_twice(monkeypatch):
    """Both candidate URLs answer here. Whichever one supplies the ids ends
    the loop, so no id is listed twice."""
    payload = _models_payload("only", 8192, [])
    routes = {
        "http://local/models": _FakeResp(200, payload),
        "http://local/v1/models": _FakeResp(200, payload),
        "http://local/props": _FakeResp(404, {}),
    }
    monkeypatch.setattr(features, "_probe_cache", {})
    monkeypatch.setattr(features.httpx, "AsyncClient", _fake_client_for(routes))
    assert (await features.probe_endpoint("http://local"))["models"] == ["only"]
