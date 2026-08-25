"""A fresh clone must boot quietly and leave nothing broken behind.

Three symptoms a first-time user hit, all from the same root cause — the seeded
`router-8085` preset is created before the user has chosen anything, so its
models_dir is the factory default `~/llama.cpp/models`:

  * on a box whose llama.cpp lives elsewhere that path does not exist, so boot
    auto-start raised and every single boot ended in a red ERROR for a preset
    nobody created;
  * on a box with a source checkout at ~/llama.cpp the path *does* exist (the
    repo ships a models/ dir of ggml-vocab test stubs), so the router came up
    serving stubs the user never asked for;
  * finishing the wizard did not help, because setting the models root never
    touched the preset — the ERROR came back on the next boot of a fully
    configured install.

The rule these tests pin: the factory default is the one models path that was
never a user decision, so it may be repointed or skipped, and anything the user
actually chose is left alone.
"""
from __future__ import annotations

from lld.presets import LlamaServerConfig, PresetRegistry
from lld.settings import factory_models_root


def _router(name: str, models_dir: str) -> LlamaServerConfig:
    return LlamaServerConfig(name=name, mode="router", models_dir=models_dir, port=8085)


def test_repoint_moves_only_the_untouched_factory_default(tmp_path, monkeypatch):
    from lld.api import setup_api

    registry = PresetRegistry(path=tmp_path / "presets.json")
    chosen = "/mnt/weights/gguf"
    factory = factory_models_root()
    registry.upsert(_router("router-8085", factory))
    registry.upsert(_router("router-mine", "/srv/models"))
    registry.upsert(
        LlamaServerConfig(name="chat", mode="single", model_path="/srv/models/a.gguf")
    )
    monkeypatch.setattr(setup_api, "PresetRegistry", lambda: registry)

    moved = setup_api._repoint_factory_routers(chosen, factory)

    assert moved == ["router-8085"]
    by_name = {c.name: c for c in registry.list()}
    assert by_name["router-8085"].models_dir == chosen
    # A models_dir the user typed in is a decision, not a leftover.
    assert by_name["router-mine"].models_dir == "/srv/models"


def test_repoint_is_a_noop_when_the_chosen_root_is_the_default(tmp_path, monkeypatch):
    """Someone whose models really do live in ~/llama.cpp/models picks that path
    in the wizard. Nothing to move, and nothing to report."""
    from lld.api import setup_api

    factory = factory_models_root()
    registry = PresetRegistry(path=tmp_path / "presets.json")
    registry.upsert(_router("router-8085", factory))
    monkeypatch.setattr(setup_api, "PresetRegistry", lambda: registry)

    assert setup_api._repoint_factory_routers(factory, factory) == []


def test_missing_factory_scan_root_is_not_a_warning(caplog):
    """The only scan root a fresh install has is one the user never chose.
    Warning about it greets a first boot with a problem that isn't one."""
    import logging

    from lld import models

    with caplog.at_level(logging.WARNING, logger="lld.models"):
        models.scan_roots([factory_models_root()])
    assert caplog.records == []

    with caplog.at_level(logging.WARNING, logger="lld.models"):
        models.scan_roots(["/mnt/usb-that-is-unplugged"])
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
