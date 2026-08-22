"""The INI on disk vs the table the router actually parsed.

Grounded in the 2026-08-22 incident: a preset went 96000 → 132000, LlamaDeck
rewrote router-models.ini the same second, and the router — which reads that
file only at startup — kept loading the model at 96000 for six hours while
every screen showed 132000. The fixtures are trimmed from that machine's real
INI and the real `status.args` the router published alongside it.
"""
from __future__ import annotations

from lld.router_drift import args_to_map, ini_drift, parse_ini

INI = """version = 1

; global defaults from router preset 'router-8085'
[*]
ctx-size = 32768
n-gpu-layers = 999
parallel = 2
flash-attn = on
cache-type-k = q8_0
jinja = true

; from preset 'qwen3.8-27b-R9700-vision'
[qwen3.8-27b-R9700-vision]
model = /m/Qwen3.8-27B-UD-Q6_K.gguf
ctx-size = 132000
cache-type-k = f16
n-gpu-layers = 999
parallel = 1
flash-attn = on
jinja = true
"""

# The router's own argv for that model, as returned by GET /models.
LIVE_96K = [
    "/b/llama-server", "--host", "127.0.0.1", "--jinja", "--metrics",
    "--port", "34401", "--slots", "--alias", "qwen3.8-27b-R9700-vision",
    "--ctx-size", "96000", "--cache-type-k", "f16",
    "--flash-attn", "on", "--model", "/m/Qwen3.8-27B-UD-Q6_K.gguf",
    "--n-gpu-layers", "999", "--parallel", "1",
]
LIVE_132K = [a if a != "96000" else "132000" for a in LIVE_96K]


def _models(args: list[str], model_id: str = "qwen3.8-27b-R9700-vision") -> list[dict]:
    return [{"id": model_id, "status": {"value": "loaded", "args": args}}]


def test_parse_ini_keeps_the_preamble_out_of_the_sections():
    """`version = 1` sits before any header — configparser refuses the file
    outright over it, which is why this parser exists."""
    global_, sections = parse_ini(INI)
    assert global_["ctx-size"] == "32768"
    assert "version" not in global_
    assert list(sections) == ["qwen3.8-27b-R9700-vision"]
    assert sections["qwen3.8-27b-R9700-vision"]["ctx-size"] == "132000"


def test_args_to_map_reads_a_valueless_flag_as_true():
    m = args_to_map(["--jinja", "--ctx-size", "8192", "--slots"])
    assert m == {"jinja": "true", "ctx-size": "8192", "slots": "true"}


def test_the_incident_is_reported_as_one_line():
    diffs = ini_drift(INI, _models(LIVE_96K))
    assert diffs == [{
        "model": "qwen3.8-27b-R9700-vision",
        "key": "ctx-size",
        "ini": "132000",
        "live": "96000",
    }]


def test_in_sync_router_reports_nothing():
    """Guards the alarm itself: a false positive here would put a permanent
    'reload me' banner on the page, which is how a real warning gets ignored."""
    assert ini_drift(INI, _models(LIVE_132K)) == []


def test_section_overrides_global_rather_than_conflicting_with_it():
    """The `[*]` ctx is 32768 and the section's is 132000. llama.cpp cascades
    the global under the section, so only the section value may be compared —
    comparing the global too would report drift on every overridden key."""
    assert not any(d["ini"] == "32768" for d in ini_drift(INI, _models(LIVE_132K)))


def test_global_only_key_is_still_compared():
    ini = INI.replace("parallel = 2", "parallel = 4")
    live = [a for a in LIVE_132K]
    # model section has parallel = 1, so change that too and leave the global
    # as the only source for a key the live args disagree with.
    ini = ini.replace("parallel = 1\n", "")
    diffs = ini_drift(ini, _models(live))
    assert {"model": "qwen3.8-27b-R9700-vision", "key": "parallel", "ini": "4", "live": "1"} in diffs


def test_injected_flags_are_never_drift():
    """host/port/alias come from the router, not the INI: the port is picked
    fresh at load time and would otherwise differ on every single poll."""
    ini = INI + "host = 0.0.0.0\nport = 9999\nalias = something-else\n"
    assert ini_drift(ini, _models(LIVE_132K)) == []


def test_boolean_and_numeric_spellings_match():
    ini = INI.replace("jinja = true", "jinja = yes").replace("n-gpu-layers = 999\nparallel = 1", "n-gpu-layers = 999.0\nparallel = 1")
    assert ini_drift(ini, _models(LIVE_132K)) == []


def test_a_key_added_to_the_ini_after_startup_shows_as_missing():
    ini = INI.replace("cache-type-k = f16", "cache-type-k = f16\nreasoning-budget = 4096")
    assert {
        "model": "qwen3.8-27b-R9700-vision",
        "key": "reasoning-budget",
        "ini": "4096",
        "live": "—",
    } in ini_drift(ini, _models(LIVE_132K))


def test_a_model_added_to_the_ini_after_startup_is_named():
    ini = INI + "\n[Brand-New-Model]\nmodel = /m/new.gguf\n"
    assert {
        "model": "Brand-New-Model",
        "key": "(model)",
        "ini": "present",
        "live": "missing",
    } in ini_drift(ini, _models(LIVE_132K))


def test_a_model_without_published_args_is_skipped_not_guessed():
    """An older router build publishes no args for a model it has never
    loaded. Unknown is not the same as different."""
    models = [{"id": "qwen3.8-27b-R9700-vision", "status": {"value": "unloaded"}}]
    assert ini_drift(INI, models) == []


def test_empty_payload_does_not_flag_every_model_as_missing():
    """A router that is up but has not answered /models yet (or a 502 that the
    caller swallowed) must not paint the whole table red."""
    assert ini_drift(INI, []) == []


def test_a_flag_deleted_from_the_preset_is_reported_too():
    """The reverse direction, and the reason it matters: `--tools` was removed
    from a preset, the INI lost the line, and the router kept spawning the
    model with `--tools true` — which is what made it unloadable in the first
    place. Only reporting INI-side changes would call that "in sync"."""
    live = [*LIVE_132K, "--tools", "true"]
    assert {
        "model": "qwen3.8-27b-R9700-vision",
        "key": "tools",
        "ini": "—",
        "live": "true",
    } in ini_drift(INI, _models(live))


def test_a_key_that_moved_between_sections_is_not_a_deletion():
    """ctx-size lives in both [*] and the model section here. Dropping it from
    the section leaves the global in force — the key is still in the file, so
    it is a value change at most, never a removal."""
    ini = INI.replace("model = /m/Qwen3.8-27B-UD-Q6_K.gguf\nctx-size = 132000",
                      "model = /m/Qwen3.8-27B-UD-Q6_K.gguf")
    assert not any(d["ini"] == "—" for d in ini_drift(ini, _models(LIVE_132K)))


def test_the_routers_own_control_flags_are_not_drift():
    """llama.cpp copies its base params into every model it spawns, so a child
    carries --metrics/--slots/--props that no INI ever set. Reporting those
    would put a permanent banner on the page, which is how a real warning
    stops being read."""
    live = [*LIVE_132K, "--metrics", "--slots", "--props", "--no-webui"]
    assert ini_drift(INI, _models(live)) == []


def test_drift_is_measured_against_the_ini_the_router_was_launched_with():
    """An adopted router carries its own `--models-preset`, which need not be
    the file LlamaDeck would write today. Comparing against the wrong file
    invents drift for every model at once."""
    from lld.api.router_api import INI_PATH, _ini_path_for

    assert str(_ini_path_for({"config": {"models_preset_path": "/srv/other.ini"}})) == "/srv/other.ini"
    assert _ini_path_for({"config": {}}) == INI_PATH
    assert _ini_path_for({}) == INI_PATH
