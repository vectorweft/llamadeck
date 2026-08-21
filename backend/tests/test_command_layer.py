"""The command box: one raw command line that is what actually runs.

Three promises are tested here, because breaking any of them turns the box
from a tool into a lie:

  1. render -> parse -> render is a fixed point (no drift on a round trip)
  2. deleting a flag in the box deletes it for real (absence is information)
  3. `argv_override` beats every field, and beats them in router mode too
"""
from __future__ import annotations

import pytest

from lld.argv import (
    command_argv,
    config_diff,
    from_argv,
    merge_command,
    split_command,
    to_argv,
    to_command,
)
from lld.flag_catalog import flags_missing_values, parse_help
from lld.model_profiles import Recipe, apply_recipe, detect_capabilities
from lld.settings import LlamaServerConfig

BIN = "/opt/llama.cpp/build/bin/llama-server"


def _preset(**kw) -> LlamaServerConfig:
    base = dict(
        name="qwen38-chat",
        model_path="/ml/models/Qwen3.8-27B-UD-Q6_K_XL.gguf",
        ctx_size=32768,
        jinja=True,
        extra_flags=["--n-cpu-moe", "24"],
    )
    base.update(kw)
    return LlamaServerConfig(**base)


# --- 1. the round trip -------------------------------------------------------

def test_rendered_command_parses_back_to_the_same_preset():
    cfg = _preset(reasoning="off", presence_penalty=1.5, devices=["CUDA0"])
    back, warnings = merge_command(cfg, to_command(cfg, BIN))
    assert config_diff(cfg, back) == []
    assert warnings == []


def test_round_trip_survives_wrapping_comments_and_quotes():
    text = (
        "llama-server \\\n"
        "  --model '/ml/a b.gguf' \\\n"
        "  # the drafter lives next to it\n"
        "  --chat-template-kwargs '{\"enable_thinking\": false}' \\\n"
        "  --ctx-size 4096\n"
    )
    cfg, _ = merge_command(_preset(), text)
    assert cfg.model_path == "/ml/a b.gguf"
    assert cfg.ctx_size == 4096
    assert '{"enable_thinking": false}' in cfg.extra_flags
    # A path with a space must come back out quoted, or the next parse splits it.
    assert "'/ml/a b.gguf'" in to_command(cfg, BIN)


def test_unbalanced_quote_is_an_error_not_a_truncated_command():
    with pytest.raises(ValueError):
        split_command("llama-server --api-key 'oops")


# --- 2. absence is information ----------------------------------------------

def test_deleting_a_boolean_flag_in_the_box_sticks():
    """--metrics/--slots/--jinja are emitted only when true, so a command
    without them means off. Defaulting to the dataclass (metrics on) would
    resurrect the flag on the very next render."""
    cfg = _preset(metrics=True, slots=True, jinja=True)
    stripped = "\n".join(
        line for line in to_command(cfg, BIN).splitlines()
        if "--metrics" not in line and "--slots" not in line and "--jinja" not in line
    )
    back, _ = merge_command(cfg, stripped)
    assert (back.metrics, back.slots, back.jinja) == (False, False, False)
    rendered = to_command(back, BIN)
    assert "--metrics" not in rendered and "--jinja" not in rendered


def test_metadata_survives_a_command_that_cannot_express_it():
    cfg = _preset(notes="ocr rig", ui_hidden=True, estimated_vram_mb=21000)
    back, _ = merge_command(cfg, "llama-server -m /ml/other.gguf")
    assert back.name == cfg.name
    assert back.notes == "ocr rig"
    assert back.ui_hidden is True
    assert back.estimated_vram_mb == 21000
    assert back.model_path == "/ml/other.gguf"


# --- 3. the override wins ----------------------------------------------------

def test_override_replaces_every_field():
    cfg = _preset(ctx_size=32768, argv_override="llama-server -m /ml/raw.gguf -c 999")
    argv = to_argv(cfg, BIN)
    assert argv == [BIN, "-m", "/ml/raw.gguf", "-c", "999"]
    assert "--ctx-size" not in argv          # the field lost, as promised
    assert "32768" not in argv


def test_override_wins_in_router_mode_too():
    cfg = _preset(mode="router", models_dir="/ml", argv_override="llama-server --port 9999")
    assert to_argv(cfg, BIN) == [BIN, "--port", "9999"]


def test_override_always_runs_the_configured_binary():
    """A saved command carries whatever path the user pasted. Honouring it
    would start yesterday's build after a rebuild moves the binary."""
    assert command_argv("/old/path/llama-server -c 8", BIN) == [BIN, "-c", "8"]
    assert command_argv("-c 8", BIN) == [BIN, "-c", "8"]        # no program at all
    assert command_argv("", BIN) == [BIN]


def test_empty_override_falls_back_to_the_fields():
    for empty in (None, "", "   \n"):
        cfg = _preset(argv_override=empty)
        assert "--ctx-size" in to_argv(cfg, BIN)


# --- reasoning / thinking ----------------------------------------------------

def test_reasoning_auto_emits_nothing():
    assert "--reasoning" not in to_argv(_preset(), BIN)


@pytest.mark.parametrize("mode", ["on", "off"])
def test_reasoning_is_emitted_and_parsed_back(mode):
    argv = to_argv(_preset(reasoning=mode), BIN)
    assert argv[argv.index("--reasoning") + 1] == mode
    assert from_argv(argv).reasoning == mode


def test_bare_reasoning_flag_means_on():
    """`--reasoning` takes an OPTIONAL value; a bare one is legal."""
    assert from_argv(["llama-server", "--reasoning", "--jinja"]).reasoning == "on"
    assert from_argv(["llama-server", "-rea", "off"]).reasoning == "off"


# --- router commands ---------------------------------------------------------

def test_router_command_parses_as_a_router_preset():
    cfg, warnings = merge_command(
        _preset(),
        "llama-server --models-dir /ml/models --models-max 2 --no-models-autoload --port 8085",
    )
    assert cfg.mode == "router"
    assert cfg.models_dir == "/ml/models"
    assert cfg.models_max == 2
    assert cfg.models_autoload is False
    assert cfg.extra_flags == []
    assert warnings == []


def test_router_flags_without_models_dir_warn():
    _, warnings = merge_command(_preset(), "llama-server --models-max 2")
    assert any("models-dir" in w for w in warnings)


def test_drafter_without_spec_type_is_read_as_speculative():
    cfg, warnings = merge_command(_preset(), "llama-server -m /a.gguf -md /draft.gguf")
    assert cfg.spec_type == "draft-simple"
    assert warnings


# --- rendering ---------------------------------------------------------------

def test_negative_values_stay_on_their_flag_line():
    cfg = _preset(threads=-1, extra_flags=["--reasoning-budget", "-1"])
    lines = to_command(cfg, BIN).splitlines()
    assert any(line.strip().startswith("--reasoning-budget -1") for line in lines), lines


def test_single_line_rendering_is_pasteable():
    cfg = _preset()
    one = to_command(cfg, BIN, multiline=False)
    assert "\\" not in one and "\n" not in one
    assert split_command(one) == to_argv(cfg, BIN)


def test_every_native_flag_can_be_parsed_back():
    """Every flag `to_argv` emits must be one `from_argv` recognizes, or the
    field silently degrades into extra_flags on the next round trip — which is
    how presence_penalty came back as None from its own rendered command."""
    from lld.argv import _FLAG_ALIASES, _FIELD_FLAGS, _canonical

    for field, flag, _kind in _FIELD_FLAGS:
        assert flag in _FLAG_ALIASES, f"{field} emits {flag}, which from_argv cannot parse"
        assert _canonical(flag) == flag


# --- flag catalog ------------------------------------------------------------

HELP = """----- common params -----

-t,    --threads N                      number of CPU threads to use during generation (default: -1)
                                        (env: LLAMA_ARG_THREADS)
--jinja, --no-jinja                     whether to use jinja template engine for chat (default: enabled)
-rea,  --reasoning [on|off|auto]        Use reasoning/thinking in the chat ('on', 'off', or 'auto')
-ot,   --override-tensor <tensor name pattern>=<buffer type>,...
                                        override tensor buffer type
"""


def test_catalog_reads_aliases_and_values():
    cat = parse_help(HELP)
    assert cat.available
    assert cat.known("-t") and cat.known("--threads")
    assert cat.get("-t").canonical == "--threads"
    assert cat.get("--threads").value_required
    assert cat.get("--reasoning").choices == ["on", "off", "auto"]
    assert not cat.get("--reasoning").value_required   # bare --reasoning is legal
    assert not cat.get("--jinja").takes_value


def test_catalog_prefers_the_positive_spelling():
    assert parse_help(HELP).get("--no-jinja").canonical == "--jinja"


def test_long_spec_keeps_its_value_placeholder():
    """A spec wider than the description column owns the whole line; splitting
    it at the column truncated `<tensor name pattern>` to `<tensor name pa`."""
    spec = parse_help(HELP).get("-ot")
    assert spec.takes_value
    assert spec.help == "override tensor buffer type"


def test_catalog_suggests_a_correction():
    assert "--threads" in parse_help(HELP).suggest("--thread")


def test_unqueryable_binary_claims_nothing():
    empty = parse_help("")
    assert not empty.available
    assert empty.suggest("--anything") == []


# --- model profiles ----------------------------------------------------------

def test_thinking_is_detected_from_the_chat_template():
    caps, why = detect_capabilities("{% if enable_thinking %}<think>{% endif %}")
    assert caps["thinking"]
    assert "enable_thinking" in why["thinking"]


def test_a_plain_template_advertises_nothing():
    caps, _ = detect_capabilities("{{ messages }}")
    assert not caps["thinking"] and not caps["tools"] and not caps["vision"]


def test_recipe_coerces_gguf_float_noise():
    """GGUF metadata is f32: top_k arrives as 20.0, which llama-server rejects
    as a value for `--top-k`."""
    cfg = apply_recipe(_preset(), Recipe(id="r", label="r", set={"top_k": 20.0, "top_p": 0.949999988}))
    assert cfg.top_k == 20 and isinstance(cfg.top_k, int)
    assert cfg.top_p == 0.95
    assert "--top-k" in to_argv(cfg, BIN)
    assert "20" in to_argv(cfg, BIN) and "20.0" not in to_argv(cfg, BIN)


def test_recipe_replaces_a_flag_instead_of_stacking_it():
    cfg = _preset(extra_flags=["--n-cpu-moe", "24", "--no-context-shift"])
    apply_recipe(cfg, Recipe(id="r", label="r", remove_flags=["--n-cpu-moe"], add_flags=["--n-cpu-moe", "32"]))
    assert cfg.extra_flags == ["--no-context-shift", "--n-cpu-moe", "32"]


# --- saving an override ------------------------------------------------------

def test_saving_an_override_makes_the_fields_mirror_the_command():
    """Everything outside argv rendering — the health probe, the port-conflict
    check, the VRAM estimate, the router INI — reads the FIELDS. If they keep
    the form's stale values while the command runs something else, the app
    reports 32K context for a process running 128K."""
    from lld.api.presets_api import _build_config

    cfg = _build_config({
        "name": "raw",
        "model_path": "/ml/old.gguf",
        "port": 8080,
        "ctx_size": 32768,
        "notes": "keep me",
        "argv_override": "llama-server -m /ml/new.gguf --port 8099 -c 131072",
    })
    assert cfg.model_path == "/ml/new.gguf"
    assert cfg.port == 8099
    assert cfg.ctx_size == 131072
    assert cfg.notes == "keep me"
    assert cfg.argv_override                      # still the thing that runs
    assert to_argv(cfg, BIN) == [BIN, "-m", "/ml/new.gguf", "--port", "8099", "-c", "131072"]


def test_an_unparseable_override_is_rejected_at_save():
    from fastapi import HTTPException

    from lld.api.presets_api import _build_config

    with pytest.raises(HTTPException) as e:
        _build_config({"name": "raw", "argv_override": "llama-server --api-key 'oops"})
    assert e.value.status_code == 400


def test_a_preset_without_an_override_is_untouched_by_the_sync():
    from lld.api.presets_api import _build_config

    cfg = _build_config({"name": "plain", "model_path": "/ml/a.gguf", "ctx_size": 8192})
    assert cfg.ctx_size == 8192
    assert cfg.argv_override is None


# --- shadowed flags ----------------------------------------------------------

def test_extra_flags_that_overrule_a_field_are_reported():
    """The state a hand-tuned preset actually ends up in: sampling copied off
    a terminal into extra_flags, silently beating the form's own values."""
    from lld.argv import shadowed_flags

    cfg = _preset(temperature=0.8, extra_flags=["--temp", "0.7", "--reasoning", "on"])
    rows = shadowed_flags(to_argv(cfg, BIN))
    assert [r["flag"] for r in rows] == ["--temp"]
    assert rows[0]["wins"] == "0.7"          # llama.cpp keeps the last one
    assert rows[0]["shadowed"] == ["0.8"]


def test_aliases_of_one_flag_count_as_the_same_flag():
    from lld.argv import shadowed_flags

    rows = shadowed_flags(["llama-server", "--temp", "0.8", "--temperature", "0.2"])
    assert len(rows) == 1
    assert set(rows[0]["spellings"]) == {"--temp", "--temperature"}


def test_a_repeated_value_or_boolean_is_not_worth_reporting():
    from lld.argv import shadowed_flags

    assert shadowed_flags(["llama-server", "--top-k", "20", "--top-k", "20"]) == []
    assert shadowed_flags(["llama-server", "--jinja", "--jinja"]) == []


def test_folding_the_rendered_command_resolves_the_shadowing():
    """The fix the editor offers: parse the command back, and the winning
    value lands in the field while extra_flags empties out."""
    from lld.argv import shadowed_flags

    cfg = _preset(temperature=0.8, extra_flags=["--temp", "0.7", "--reasoning", "on"])
    folded, _ = merge_command(cfg, to_command(cfg, BIN))
    assert folded.temperature == 0.7
    assert folded.reasoning == "on"
    assert folded.extra_flags == []
    assert shadowed_flags(to_argv(folded, BIN)) == []


def test_gguf_float_noise_is_cleaned_where_it_enters():
    """f32 metadata reads back as 0.949999988079071 / 20.0. Cleaned at the
    source so the editor, the recipes and API clients all see one tidy value —
    and so `--top-k` never renders as `20.0`, which llama-server rejects."""
    from lld.model_defaults import _clean_sampling

    assert _clean_sampling("top_p", 0.949999988079071) == 0.95
    assert _clean_sampling("top_k", 20.0) == 20
    assert isinstance(_clean_sampling("top_k", 20.0), int)
    assert _clean_sampling("temperature", 1.0) == 1.0


# --- What's New hints: only offer what can actually be applied ---------------

async def test_card_flags_are_classified_before_being_offered():
    """Feature cards list the flags a change *touches*, read out of release
    notes. A Vulkan diagnostics card naming --model/--device/--ctx-size has
    nothing to add to a preset: those are fields. Offering them appended
    `--model` with no value into extra_flags."""
    from lld.flag_catalog import classify_flags

    cfg = _preset()
    buckets = await classify_flags(cfg, BIN, ["--model", "--device", "--ctx-size", "--gpu-layers"])
    assert buckets["actionable"] == []
    assert set(buckets["managed"]) == {"--model", "--device", "--ctx-size", "--gpu-layers"}


async def test_a_real_new_flag_stays_actionable():
    from lld.flag_catalog import classify_flags

    buckets = await classify_flags(_preset(), BIN, ["--no-context-shift", "--n-cpu-moe"])
    # No catalog for a fake binary path, so nothing is rejected as unknown and
    # nothing is demoted for wanting a value — both stay offerable.
    assert "--no-context-shift" in buckets["actionable"]


async def test_a_flag_already_in_the_command_is_not_offered_again():
    from lld.flag_catalog import classify_flags

    cfg = _preset(extra_flags=["--no-context-shift"])
    buckets = await classify_flags(cfg, BIN, ["--no-context-shift"])
    assert buckets["present"] == ["--no-context-shift"]
    assert buckets["actionable"] == []


def test_a_card_with_no_architecture_matches_no_model():
    """"The summarizer did not scope this card" is unknown, not universal —
    treating it as universal put router and TTS cards on a text preset."""
    from lld.api.features_api import _arch_matches

    assert _arch_matches([], "qwen35") is False
    assert _arch_matches(["qwen3"], "qwen35") is True
    assert _arch_matches(["gemma4"], "qwen35") is False
    assert _arch_matches(["qwen35"], None) is False


def test_a_card_with_nothing_to_add_is_not_a_hint():
    """The editor's panel exists to add a flag. A card that only names fields
    (`--model`, `--ctx-size`) belongs on the What's New page — showing it here
    with an Add button was how a Qwen3-TTS note landed on a text preset."""
    import asyncio

    from lld.flag_catalog import classify_flags

    buckets = asyncio.run(classify_flags(_preset(), BIN, ["--model", "--ctx-size"]))
    assert buckets["actionable"] == []


# --- two GPUs: combinations that cancel each other out -----------------------

def test_sm_none_silently_defeats_an_ot_placement():
    """Measured on this box: `-sm none` loads on ONE GPU, so tensors `-ot`
    sends to the second card never arrive — and llama-server says nothing.
    `-ts 1,0` is the spelling that actually works."""
    from lld.argv import flag_conflicts

    rows = flag_conflicts(["llama-server", "-sm", "none", "-ot", "exps=Vulkan1"])
    assert [r["id"] for r in rows] == ["sm-none-vs-ot"]
    assert "-ts" in rows[0]["message"]


def test_a_working_two_gpu_layout_reports_no_conflict():
    from lld.argv import flag_conflicts

    assert flag_conflicts([
        "llama-server", "-dev", "CUDA0,Vulkan1", "-ts", "1,0", "-ot", "exps=Vulkan1",
    ]) == []
    # layer-split is the default and does not fight -ot
    assert flag_conflicts(["llama-server", "--split-mode", "layer", "-ot", "x=CPU"]) == []


async def test_pinning_one_card_under_two_backends_is_rejected():
    """CUDA0 and Vulkan2 are one 5090. Accepting both budgets 64 GB onto a
    32 GB card, which loads until it aborts out of memory."""
    from fastapi import HTTPException

    from lld.api import presets_api
    from lld.devices import LlamaDevice

    async def fake_probe(_binary):
        return [
            LlamaDevice(id="CUDA0", name="NVIDIA GeForce RTX 5090", total_mb=32000,
                        free_mb=32000, backend="CUDA"),
            LlamaDevice(id="Vulkan2", name="NVIDIA GeForce RTX 5090", total_mb=32000,
                        free_mb=32000, backend="Vulkan", duplicate_of="CUDA0"),
        ]

    presets_api.probe_devices = fake_probe
    try:
        with pytest.raises(HTTPException) as e:
            await presets_api._reject_aliased_devices("llama-server", ["CUDA0", "Vulkan2"])
        assert e.value.status_code == 400
        assert "same physical card" in e.value.detail
        # One id, or two genuinely different cards, stay allowed.
        await presets_api._reject_aliased_devices("llama-server", ["CUDA0"])
    finally:
        from lld.devices import probe_devices as real_probe
        presets_api.probe_devices = real_probe


# --- per-preset environment --------------------------------------------------

def test_env_renders_and_parses_back_like_a_shell():
    """`GGML_CUDA_DISABLE_GRAPHS=1` has no flag — a preset can only reach it
    through the environment, and the command box has to show it or the box
    stops being the whole truth."""
    cfg = _preset(env={"GGML_CUDA_DISABLE_GRAPHS": "1"})
    cmd = to_command(cfg, BIN)
    assert cmd.startswith("GGML_CUDA_DISABLE_GRAPHS=1")
    back, _ = merge_command(cfg, cmd)
    assert back.env == {"GGML_CUDA_DISABLE_GRAPHS": "1"}
    assert config_diff(cfg, back) == []


def test_env_is_not_passed_to_exec_as_an_argument():
    """A leading KEY=VALUE is environment, not argv — handing it to execve
    would make llama-server reject it as an unknown argument."""
    from lld.argv import command_argv, command_env

    assert command_argv("FOO=1 BAR=2 llama-server -c 8", BIN) == [BIN, "-c", "8"]
    assert command_env("FOO=1 BAR=2 llama-server -c 8") == {"FOO": "1", "BAR": "2"}


def test_only_the_leading_run_counts_as_environment():
    """`--api-key k=v` is an argument that happens to contain '='."""
    from lld.argv import command_argv, command_env

    text = "FOO=1 llama-server --api-key secret=v -m /a.gguf"
    assert command_env(text) == {"FOO": "1"}
    assert "secret=v" in command_argv(text, BIN)


def test_spawn_env_is_none_when_the_preset_sets_nothing():
    """Presets that set no variables must exec with the inherited environment
    untouched — same behaviour as before this field existed."""
    from lld.supervisor import ProcessHandle

    plain = ProcessHandle("p", _preset(), BIN)
    assert plain._spawn_env() is None

    with_env = ProcessHandle("p", _preset(env={"GGML_CUDA_DISABLE_GRAPHS": "1"}), BIN)
    merged = with_env._spawn_env()
    assert merged["GGML_CUDA_DISABLE_GRAPHS"] == "1"
    assert "PATH" in merged          # inherited, not replaced


def test_a_router_carries_the_env_of_every_model_it_serves():
    """A served preset's variables have nowhere else to land: the router loads
    its models inside its own process. Dropping them is how the 5090's
    GGML_CUDA_DISABLE_GRAPHS workaround stopped applying the day the model
    moved onto the router, and the card locked up (Xid 8) every ~20 minutes."""
    from lld.router_ini import router_env

    served = _preset(
        name="qwen38-5090",
        model_path="/ml/models/Qwen3.8-27B-UD-Q6_K.gguf",
        env={"GGML_CUDA_DISABLE_GRAPHS": "1"},
    )
    elsewhere = _preset(
        name="other-box",
        model_path="/somewhere/else/Other.gguf",
        env={"GGML_CUDA_FORCE_MMQ": "1"},
    )
    router = _preset(name="router-8085", mode="router", model_path=None,
                     models_dir="/ml/models")

    env, warnings = router_env("/ml/models", [served, elsewhere, router],
                               router_preset=router)
    assert env == {"GGML_CUDA_DISABLE_GRAPHS": "1"}   # not the one outside models_dir
    assert warnings == []


def test_the_router_process_env_reaches_the_spawn():
    """The merge is worthless if it does not make it into execve's environment."""
    from lld.supervisor import ProcessHandle

    served = _preset(name="qwen38-5090",
                     model_path="/ml/models/Qwen3.8-27B-UD-Q6_K.gguf",
                     env={"GGML_CUDA_DISABLE_GRAPHS": "1"})
    router = _preset(name="router-8085", mode="router", model_path=None,
                     models_dir="/ml/models")

    import lld.supervisor as supervisor
    real = supervisor.router_env
    supervisor.router_env = lambda md, router_preset=None: (
        real(md, [served, router], router_preset=router_preset)
    )
    try:
        merged = ProcessHandle("router-8085", router, BIN)._spawn_env()
    finally:
        supervisor.router_env = real
    assert merged["GGML_CUDA_DISABLE_GRAPHS"] == "1"
    assert "PATH" in merged          # inherited, not replaced


def test_two_presets_disagreeing_on_one_variable_is_reported():
    """One process, one value. Picking silently is how the losing preset's
    model runs with a setting its own panel says it has."""
    from lld.router_ini import router_env

    a = _preset(name="a-preset", model_path="/ml/models/A.gguf",
                env={"GGML_CUDA_DISABLE_GRAPHS": "1"})
    b = _preset(name="b-preset", model_path="/ml/models/B.gguf",
                env={"GGML_CUDA_DISABLE_GRAPHS": "0"})

    env, warnings = router_env("/ml/models", [a, b])
    assert env == {"GGML_CUDA_DISABLE_GRAPHS": "1"}   # first by name
    assert len(warnings) == 1
    assert "b-preset" in warnings[0] and "a-preset" in warnings[0]

    # The router's own value settles it, and says so.
    router = _preset(name="router-8085", mode="router", model_path=None,
                     models_dir="/ml/models", env={"GGML_CUDA_DISABLE_GRAPHS": "0"})
    env, warnings = router_env("/ml/models", [a, b, router], router_preset=router)
    assert env == {"GGML_CUDA_DISABLE_GRAPHS": "0"}
    assert any("router preset's own" in w for w in warnings)


def test_a_malformed_env_key_is_rejected_at_save():
    """execve rejects the whole environment, so one bad key would surface as
    'the preset will not start' with nothing pointing at the cause."""
    from fastapi import HTTPException

    from lld.api.presets_api import _reject_bad_env

    _reject_bad_env(_preset(env={"GGML_CUDA_DISABLE_GRAPHS": "1"}))   # fine
    for bad in ("2FOO", "has space", "has-dash", ""):
        with pytest.raises(HTTPException):
            _reject_bad_env(_preset(env={bad: "1"}))


# ── the mmproj ignores -dev: MTMD_BACKEND_DEVICE is injected at spawn ──────
# llama.cpp's multimodal encoder (tools/mtmd) does not see `--device`: it picks
# its GPU from the MTMD_BACKEND_DEVICE env var, defaulting to the first GPU
# backend in ggml's registry (CUDA0 on a CUDA+Vulkan build). Without this a
# vision preset pinned to Vulkan1 still drops the encoder on the NVIDIA card —
# the exact "part of my R9700 model landed on the 5090" report this guards.

def test_a_vision_preset_pinning_a_device_pins_the_mmproj_backend():
    from lld.supervisor import ProcessHandle

    vision = _preset(
        name="qwen-r9700-vision",
        model_path="/ml/models/Qwen3.8-27B-UD-Q6_K.gguf",
        mmproj_path="/ml/models/mmproj-BF16.gguf",
        devices=["Vulkan1"],
    )
    env = ProcessHandle("qwen-r9700-vision", vision, BIN)._spawn_env()
    assert env["MTMD_BACKEND_DEVICE"] == "Vulkan1"
    assert "PATH" in env          # inherited, not replaced


def test_an_explicit_mtmd_backend_device_wins_over_the_derived_one():
    from lld.supervisor import ProcessHandle

    vision = _preset(
        mmproj_path="/ml/models/mmproj-BF16.gguf",
        devices=["Vulkan1"],
        env={"MTMD_BACKEND_DEVICE": "CUDA0"},   # user really means CUDA0
    )
    env = ProcessHandle("qwen38-chat", vision, BIN)._spawn_env()
    assert env["MTMD_BACKEND_DEVICE"] == "CUDA0"


def test_no_mtmd_backend_device_without_mmproj_or_a_device_pin():
    from lld.supervisor import ProcessHandle

    # No projector → nothing to pin.
    assert ProcessHandle("p", _preset(devices=["Vulkan1"]), BIN)._spawn_env() is None
    # Projector but no pin → "let llama.cpp choose" stays untouched.
    assert ProcessHandle(
        "p", _preset(mmproj_path="/ml/models/mmproj-BF16.gguf"), BIN
    )._spawn_env() is None
    # Projector pinned to "none" → CPU-only intent is respected.
    assert ProcessHandle(
        "p",
        _preset(mmproj_path="/ml/models/mmproj-BF16.gguf", devices=["none"]),
        BIN,
    )._spawn_env() is None


def test_the_router_carries_the_derived_mmproj_backend_of_its_vision_models():
    """The router loads models in its own process, so the encoder pin must
    reach it through the router env, not the INI (`device =` is ignored by
    the encoder there just like -dev is)."""
    from lld.router_ini import router_env

    vision = _preset(
        name="qwen-r9700-vision",
        model_path="/ml/models/Qwen3.8-27B-UD-Q6_K.gguf",
        mmproj_path="/ml/models/mmproj-BF16.gguf",
        devices=["Vulkan1"],
    )
    router = _preset(name="router-8085", mode="router", model_path=None,
                     models_dir="/ml/models")

    env, warnings = router_env("/ml/models", [vision, router],
                               router_preset=router)
    assert env["MTMD_BACKEND_DEVICE"] == "Vulkan1"
    assert warnings == []


def test_router_conflicting_mmproj_backends_are_reported_like_any_env():
    """Two served vision presets pinning different GPUs cannot both win in one
    process; first by name takes it and the loser is named, exactly like the
    GGML_CUDA_DISABLE_GRAPHS conflict."""
    from lld.router_ini import router_env

    a = _preset(
        name="a-vision", model_path="/ml/models/A.gguf",
        mmproj_path="/ml/models/mmproj-A.gguf", devices=["Vulkan1"],
    )
    b = _preset(
        name="b-vision", model_path="/ml/models/B.gguf",
        mmproj_path="/ml/models/mmproj-B.gguf", devices=["CUDA0"],
    )

    env, warnings = router_env("/ml/models", [a, b])
    assert env["MTMD_BACKEND_DEVICE"] == "Vulkan1"   # first by name
    assert len(warnings) == 1
    assert "a-vision" in warnings[0] and "b-vision" in warnings[0]


# ── a flag that needs a value and was given none ──────────────────────
# llama-server refuses the WHOLE command line for this and exits before it has
# written anything a user would read as a reason. The preset just does not
# start. The catalog knows the placeholder, so this is knowable first.

_TOOLS_HELP = """
----- example -----

--tools TOOL1,TOOL2,...                 experimental: whether to enable built-in tools
--jinja, --no-jinja                     whether to use jinja template engine for chat
-s,    --seed SEED                      RNG seed
-ts,   --tensor-split N0,N1,N2,...      fraction of the model to offload per GPU
"""


def _tools_catalog():
    return parse_help(_TOOLS_HELP)


def test_a_value_taking_flag_with_nothing_after_it_is_reported():
    missing = flags_missing_values(_tools_catalog(), ["llama-server", "--tools"])
    assert [m["flag"] for m in missing] == ["--tools"]
    assert missing[0]["placeholder"] == "TOOL1,TOOL2,..."


def test_a_value_taking_flag_followed_by_another_flag_is_reported():
    """`--tools --jinja` would silently eat --jinja as the value if llama.cpp
    allowed it; it does not, and the user meant two flags."""
    missing = flags_missing_values(
        _tools_catalog(), ["llama-server", "--tools", "--jinja"]
    )
    assert [m["flag"] for m in missing] == ["--tools"]


@pytest.mark.parametrize("argv", [
    ["llama-server", "--tools", "all"],
    ["llama-server", "--tools=all"],
    ["llama-server", "--jinja"],
])
def test_a_flag_that_has_its_value_is_not_reported(argv):
    assert flags_missing_values(_tools_catalog(), argv) == []


def test_a_negative_value_is_a_value_not_a_missing_one():
    """`--seed -1` and `-ts -1,0` start with a dash. Reading them as "the next
    flag" would report a command that runs perfectly well."""
    assert flags_missing_values(
        _tools_catalog(), ["llama-server", "--seed", "-1", "-ts", "-1,0"]
    ) == []


def test_an_unqueryable_binary_accuses_nobody():
    """No catalog is not evidence that a flag is wrong."""
    from lld.flag_catalog import FlagCatalog

    assert flags_missing_values(FlagCatalog(available=False), ["llama-server", "--tools"]) == []


def test_a_placeholder_containing_commas_is_not_cut_in_half():
    """The comma separates spellings AND lives inside several placeholders.
    Dropping everything after the first one told the user to pass `<dev1` to
    --device and `TOOL1` to --tools."""
    cat = parse_help("""
----- example -----

-dev,  --device <dev1,dev2,..>          comma-separated list of devices
-ts,   --tensor-split N0,N1,N2,...      fraction per GPU
--tools TOOL1,TOOL2,...                 built-in tools
-ot,   --override-tensor <pattern>=<buffer>,...
                                        override tensor buffer type
-t,    --threads N                      CPU threads
""")
    assert cat.get("--device").placeholder == "<dev1,dev2,..>"
    assert cat.get("-ts").placeholder == "N0,N1,N2,..."
    assert cat.get("--tools").placeholder == "TOOL1,TOOL2,..."
    assert cat.get("-ot").placeholder == "<pattern>=<buffer>,..."
    # the ordinary case is untouched
    assert cat.get("--threads").names == ["-t", "--threads"]
    assert cat.get("--threads").placeholder == "N"


def test_a_dash_number_inside_a_placeholder_is_not_read_as_a_spelling():
    cat = parse_help("""
----- example -----

--range LO,-1                           a range whose upper bound is -1
""")
    assert cat.get("--range").names == ["--range"]
    assert cat.get("--range").placeholder == "LO,-1"
