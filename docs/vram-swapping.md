# VRAM Swapping: Sharing One GPU Between an LLM and Other Workloads

A worked example of driving LlamaDeck over MCP so that an agent can time-share
a single GPU between a `llama-server` LLM and another GPU workload — image
generation (ComfyUI, SD-WebUI), TTS, training jobs, anything that needs the
VRAM your LLM is sitting on.

## The problem

Say you have a 32 GB GPU:

- Your chat model (a 27B quant with a long context) ≈ **24 GB** VRAM.
- An SDXL + ControlNet + upscale workflow ≈ **8–15 GB** VRAM.
- They don't fit together. One has to make room for the other.

The fix is a **swap**: the orchestrating agent stops the LLM when the other
workload needs the GPU, runs it, then brings the LLM back — without you
touching anything.

## The actors

| Actor | Role | Speaks |
|---|---|---|
| **Your agent** (Claude Code, or any MCP client) | Orchestrator | MCP client |
| **LlamaDeck** | llama-server process manager + VRAM estimates | MCP server (`http://127.0.0.1:8770/mcp`) |
| **The other workload** (e.g. ComfyUI) | Image/audio/whatever generation | Its own HTTP API or MCP |
| **llama-server** | The LLM process LlamaDeck manages | OpenAI-compatible HTTP |

## The flow, in seven steps

```
[1] Check VRAM          — is there already enough free?
[2] Stop the LLM        — stop_preset (~2-5 s, VRAM freed immediately)
[3] Verify              — enough free for the other workload now?
[4] Run the workload    — trigger the workflow, await the result
[5] Restart the LLM     — start_preset (spawns the process; model not loaded yet)
[6] Wait until ready    — wait_until_ready (model load, typically 30-60 s)
[7] Continue            — the LLM is serving again
```

## The MCP tools involved

| Tool | What it does | Typical latency |
|---|---|---|
| `get_vram()` | Total/used/free VRAM plus per-preset estimates | <100 ms |
| `get_server_statuses()` | Status + dynamic `vram_estimate.total_mb` per preset | <100 ms |
| `stop_preset(preset)` | SIGTERM/SIGKILL the llama-server; VRAM freed | 2–5 s |
| `start_preset(preset)` | Spawn llama-server; does **not** wait for the model to load | <1 s |
| `wait_until_ready(preset, timeout=120)` | Block until `/health` returns 200 | 30–60 s |
| `switch_preset(to, from)` | stop + start in one call (swap models on one port) | 30–60 s |
| `tail_logs(preset, n)` | Last n lines of the server log — your debugging window | <100 ms |

**The key distinction:** `start_preset` means *the process exists*;
`wait_until_ready` means *the model can actually take requests*. Always use
them together, or you'll fire requests at an empty port and collect 503s.

## Pseudocode

```python
async def run_image_workflow(workflow, llm_preset="chat-model"):
    # [1] Where are we?
    vram = await mcp.llamadeck.get_vram()
    statuses = await mcp.llamadeck.get_server_statuses()
    llm_running = statuses["presets"][llm_preset]["running"]
    needed_mb = 12_000   # estimate for this workflow

    # [2] Not enough room? Take the LLM down.
    swapped = False
    if vram["free_mb"] < needed_mb and llm_running:
        await mcp.llamadeck.stop_preset(llm_preset)
        swapped = True

    # [3] Confirm the VRAM actually came back
    vram = await mcp.llamadeck.get_vram()
    assert vram["free_mb"] >= needed_mb, "still not enough free VRAM"

    try:
        # [4] The other workload's turn
        result = await comfyui.run_workflow(workflow)
    finally:
        # [5+6] Bring the LLM back, whatever happened above
        if swapped:
            await mcp.llamadeck.start_preset(llm_preset)
            ready = await mcp.llamadeck.wait_until_ready(llm_preset, timeout=120)
            assert ready["ready"], f"LLM failed to come back: {ready.get('last_error')}"

    # [7] The caller can now use the LLM to evaluate the result
    return result
```

## Details that bite

### The other workload must actually release VRAM

Stopping your workflow isn't enough if the tool keeps models resident. For
ComfyUI specifically:

- start it with `--lowvram` / `--novram` → it unloads after every workflow, or
- POST its `/free` endpoint (ComfyUI 0.3+): `{"unload_models": true, "free_memory": true}`, or
- add a "Free GPU" node at the end of the workflow.

Other tools have their own equivalents — the pattern is the same: verify with
`get_vram()` rather than trusting the tool.

### Reload time is real

A large model with a long context takes **30–60 s** to load. That time is
spent blocked inside `wait_until_ready`. If the other workload runs often,
this cost compounds — batch N workflows into one swap instead of swapping
around each one.

### Failure modes

- `stop_preset` fails → the process may be adopted from outside LlamaDeck and
  not killable with your permissions. Check `tail_logs`.
- `start_preset` succeeds but `wait_until_ready` returns `ready=false` →
  read `tail_logs(preset, 200)`. Usual causes: VRAM not actually freed (the
  other tool didn't fully unload), model path missing, port collision.
- `wait_until_ready` fails fast (409) if the process exits early — you won't
  wait out the timeout on a crashed server.
- Some builds report healthy while still warming up. On critical paths, send
  a 1-token dummy completion before the real request.

### LLM-to-LLM swapping is a different feature

If what you want is several *LLMs* time-sharing one port (no ComfyUI
involved), use a router-mode preset instead — llama-server's router swaps
models without dropping the endpoint. It doesn't help for non-LLM workloads,
because the router itself keeps VRAM.

## Wiring it into your agent

MCP server registration (example `mcp.json`):

```json
{
  "mcpServers": {
    "llamadeck": {
      "url": "http://127.0.0.1:8770/mcp",
      "transport": "sse"
    }
  }
}
```

And a rule for the agent's system prompt, adapted to your sizes:

> When an image-generation request comes in, call `llamadeck.get_vram()`
> first. If `free_mb` is below the workflow's needs and the LLM is running,
> apply the swap pattern: `stop_preset → run workflow → start_preset →
> wait_until_ready`. Restore the LLM as soon as the workflow finishes, and
> tell the user what you're doing while they wait.

## Try it by hand first

Before wiring up the agent, verify the loop with curl (substitute your
preset name):

```bash
# Is LlamaDeck up?
curl http://127.0.0.1:8770/api/server/statuses | jq '.presets["chat-model"].running'

# Manual swap simulation
curl -X POST http://127.0.0.1:8770/api/server/stop/chat-model
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # should drop
# (run the other workload here)
curl -X POST http://127.0.0.1:8770/api/server/start/chat-model
curl "http://127.0.0.1:8770/api/server/wait_ready/chat-model?timeout=120"
# {"preset":"chat-model","ready":true,"elapsed_seconds":48.2,"attempts":97}
```

If this cycle works cleanly, the agent integration will too.

## Limitations

- LlamaDeck doesn't supervise the other workload's process — you start
  ComfyUI (or whatever) yourself. Automatic start/stop only covers
  llama-server. (The GPU broker's lease API can arbitrate more than that;
  this guide covers the simpler stop/start pattern.)
- There's no `ensure_free_vram(min_free_mb)` one-shot yet. With several LLMs
  running, the agent decides which one to stop.
