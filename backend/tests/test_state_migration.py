"""Carrying ~/.config/lsc over to ~/.config/llamadeck.

The project was renamed from LSC to LlamaDeck. An existing install keeps every
setting, preset and download token in the old directory, so getting this wrong
does not look like a bug — it looks like the app forgot who you are and came up
as a fresh install.
"""
from __future__ import annotations

import json

import pytest

from lld import settings as settings_mod


@pytest.fixture
def state(tmp_path, monkeypatch):
    """Point every state path at tmp_path and re-arm the one-shot migration."""
    new = tmp_path / "llamadeck"
    legacy = tmp_path / "lsc"
    monkeypatch.setattr(settings_mod, "STATE_DIR", new)
    monkeypatch.setattr(settings_mod, "LEGACY_STATE_DIR", legacy)
    monkeypatch.setattr(settings_mod, "DB_PATH", new / "llamadeck.db")
    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", new / "settings.json")
    monkeypatch.setattr(settings_mod, "PRESETS_PATH", new / "presets.json")
    monkeypatch.setattr(settings_mod, "LOGS_DIR", new / "logs")
    # The real guard is "STATE_DIR is where _state_dir() says it should be",
    # which is what keeps a test from moving the developer's own config into a
    # temporary directory. Keep the two consistent instead of disabling it.
    monkeypatch.setattr(settings_mod, "_state_dir", lambda: new)
    monkeypatch.setattr(settings_mod, "_migrated", False)
    return new, legacy


def _populate_legacy(legacy):
    (legacy / "logs").mkdir(parents=True)
    (legacy / "settings.json").write_text(json.dumps({"controller_bind_port": 9999}))
    (legacy / "presets.json").write_text(
        json.dumps([{"name": "router", "models_preset_path": str(legacy / "router-models.ini")}])
    )
    (legacy / "router-models.ini").write_text("[*]\nctx-size = 8192\n")
    (legacy / "lsc.db").write_bytes(b"sqlite-ish")
    (legacy / "lsc.db-wal").write_bytes(b"wal")


def test_the_old_directory_is_moved_not_copied(state):
    new, legacy = state
    _populate_legacy(legacy)

    settings_mod.ensure_state_dirs()

    assert (new / "settings.json").exists()
    assert (new / "router-models.ini").read_text().startswith("[*]")
    assert legacy.is_symlink(), "the old path must survive as a symlink, not a second copy"
    assert legacy.resolve() == new.resolve()


def test_the_settings_survive_the_move(state):
    new, legacy = state
    _populate_legacy(legacy)
    assert settings_mod.load_settings().controller_bind_port == 9999


def test_the_database_is_renamed_with_its_wal(state):
    new, legacy = state
    _populate_legacy(legacy)

    settings_mod.ensure_state_dirs()

    assert (new / "llamadeck.db").read_bytes() == b"sqlite-ish"
    assert (new / "llamadeck.db-wal").read_bytes() == b"wal"
    assert not (new / "lsc.db").exists()


def test_absolute_paths_stored_inside_the_state_are_rewritten(state):
    """A router preset pins its INI by absolute path. Left alone it would keep
    naming the old directory forever — working, through the symlink, but
    showing the pre-rename name back to the user in the editor."""
    new, legacy = state
    _populate_legacy(legacy)

    settings_mod.ensure_state_dirs()

    presets = json.loads((new / "presets.json").read_text())
    assert presets[0]["models_preset_path"] == str(new / "router-models.ini")


def test_nothing_happens_when_the_new_directory_already_exists(state):
    """Never merge two states. If both exist the new one is authoritative and
    the old one is left exactly where it is for the user to inspect."""
    new, legacy = state
    _populate_legacy(legacy)
    new.mkdir()
    (new / "settings.json").write_text(json.dumps({"controller_bind_port": 8770}))

    settings_mod.ensure_state_dirs()

    assert settings_mod.load_settings().controller_bind_port == 8770
    assert not legacy.is_symlink()
    assert (legacy / "settings.json").exists()


def test_a_fresh_install_just_creates_the_directory(state):
    new, legacy = state
    settings_mod.ensure_state_dirs()
    assert new.is_dir()
    assert not legacy.exists()


def test_migration_never_targets_a_redirected_state_dir(tmp_path, monkeypatch):
    """The guard that keeps a test suite from eating the developer's config:
    when STATE_DIR has been moved away from what _state_dir() computes,
    migration is off."""
    new = tmp_path / "somewhere-else"
    legacy = tmp_path / "lsc"
    legacy.mkdir()
    (legacy / "settings.json").write_text("{}")
    monkeypatch.setattr(settings_mod, "STATE_DIR", new)
    monkeypatch.setattr(settings_mod, "LEGACY_STATE_DIR", legacy)
    monkeypatch.setattr(settings_mod, "_state_dir", lambda: tmp_path / "the-real-one")
    monkeypatch.setattr(settings_mod, "_migrated", False)

    settings_mod.migrate_legacy_state()

    assert (legacy / "settings.json").exists()
    assert not new.exists()


def test_the_state_dir_can_be_overridden_by_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMADECK_STATE_DIR", str(tmp_path / "sandbox"))
    assert settings_mod._state_dir() == tmp_path / "sandbox"
