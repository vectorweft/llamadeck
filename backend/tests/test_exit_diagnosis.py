"""Crash → explanation.

Every failed start used to read `exited with code 1`, which is the one thing
about the failure that carries no information. These are the real log tails
from a fresh Ubuntu box: a 96 GB model against a 32 GB card, a split model
whose part 1 was named differently from parts 2-4, and a build older than the
model it was pointed at.
"""
from __future__ import annotations

from lld.exit_diagnosis import diagnose_exit

OOM = [
    "0.00.230.375 I srv    load_model: loading model '/m/M-00001-of-00004.gguf'",
    "0.00.366.780 W common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 999, abort",
    "0.18.542.875 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 97809.00 MiB on device 0: cudaMalloc failed: out of memory",
    "0.18.542.881 E alloc_tensor_range: failed to allocate CUDA0 buffer of size 102560175104",
    "0.19.167.504 E llama_model_load: error loading model: unable to allocate CUDA0 buffer",
    "0.19.168.081 E srv  llama_server: exiting due to model loading error",
]

MISSING_SHARD = [
    "0.00.232.684 I srv    load_model: loading model '/m/M-0731-UD-IQ3_XXS-00001-of-00004.gguf'",
    "0.00.246.692 E gguf_init_from_file: failed to open GGUF file '/m/M-0731-UD-IQ3_XXS-00002-of-00004.gguf' (No such file or directory)",
    "0.00.247.382 E llama_model_load: error loading model: llama_model_loader: failed to load GGUF split from /m/M-0731-UD-IQ3_XXS-00002-of-00004.gguf",
    "0.00.260.596 E srv    load_model: failed to load model",
]

OLD_BUILD = [
    "0.00.100.000 I srv    load_model: loading model '/m/new.gguf'",
    "0.00.120.000 E llama_model_load: error loading model: unknown model architecture: 'deepseek4'",
]


def test_oom_names_the_size_and_the_fix():
    msg = diagnose_exit(1, OOM)
    assert "out of VRAM" in msg
    assert "95.5 GB" in msg          # 97809 MiB
    assert "--n-cpu-moe" in msg


def test_missing_shard_names_the_file():
    msg = diagnose_exit(1, MISSING_SHARD)
    assert "M-0731-UD-IQ3_XXS-00002-of-00004.gguf" in msg
    assert "part 1" in msg           # points at the real cause: the prefix


def test_old_build_says_rebuild():
    msg = diagnose_exit(1, OLD_BUILD)
    assert "deepseek4" in msg
    assert "rebuild" in msg.lower()


def test_unrecognised_error_is_quoted_not_swallowed():
    msg = diagnose_exit(1, [
        "0.00.1 I srv  starting",
        "0.00.2 E srv  some_future_failure: the disk went away",
    ])
    assert "some_future_failure: the disk went away" in msg


def test_no_error_lines_still_beats_an_exit_code():
    msg = diagnose_exit(9, ["0.00.1 I srv  loading", "0.00.2 I srv  halfway"])
    assert "halfway" in msg


def test_empty_log_falls_back_to_the_code():
    assert diagnose_exit(1, []) == "exited with code 1"


def test_autofit_probe_on_a_full_gpu_is_not_reported_as_a_model_too_big():
    """Measured 2026-08-20: a CPU-only preset (`-ngl 0`, `--device none`) died
    with "CUDA error: out of memory" while the 5090 was full with someone
    else's model. The model was never the problem — `--fit on` dry-runs a
    context on every device first, and a CUDA context needs VRAM of its own.
    `--fit off` started the same preset in 0.84 s holding zero VRAM."""
    from lld.exit_diagnosis import diagnose_exit

    log = [
        "W ggml_backend_cuda_device_get_memory: cudaMemGetInfo failed (out of memory), returning 0/0",
        "/ml/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:106: CUDA error",
        "E CUDA error: out of memory",
        "E   current device: 0, in function stream at .../ggml-cuda/common.cuh:1489",
        "E   cudaStreamCreateWithFlags(&streams[device][stream], 0x01)",
    ]
    msg = diagnose_exit(1, log)
    assert "--fit off" in msg
    assert "n_cpu_moe" not in msg          # not the "shrink your model" advice


def test_a_real_model_oom_still_says_shrink_the_model():
    from lld.exit_diagnosis import diagnose_exit

    msg = diagnose_exit(1, [
        "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 22000.00 MiB on device 0: "
        "cudaMalloc failed: out of memory",
    ])
    assert "--n-cpu-moe" in msg


# The real log from a preset that added `--tools` with no value. llama.cpp
# rejects arguments before its logger exists, so none of these lines carry the
# "E" severity field the error scanner looks for — and the usage block that
# follows ends in a line that says nothing at all.
BAD_ARGUMENT = [
    "WARNING: radv is not a conformant Vulkan implementation, testing use only.",
    'error while handling argument "--tools": expected value for argument',
    "",
    "usage:",
    "--tools TOOL1,TOOL2,...                 experimental: whether to enable built-in tools for AI agents - do not",
    "                                        enable in untrusted environments (default: no tools)",
    '                                        specify "all" to enable all tools',
    "",
    "to show complete usage, run with -h",
]


def test_a_rejected_argument_names_the_flag_not_the_usage_footer():
    """The last line of the log is "to show complete usage, run with -h", and
    quoting it back told the user nothing about which flag was wrong."""
    msg = diagnose_exit(1, BAD_ARGUMENT)
    assert "--tools" in msg
    assert "value" in msg
    assert "run with -h" not in msg


def test_other_argument_errors_are_quoted_with_their_reason():
    msg = diagnose_exit(1, [
        'error while handling argument "--reasoning": invalid value',
        "to show complete usage, run with -h",
    ])
    assert "--reasoning" in msg and "invalid value" in msg
