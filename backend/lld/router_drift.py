"""Does the router's in-memory model table still match the INI on disk?

llama-server's router reads `--models-preset` **once, at startup**. Editing a
preset in LlamaDeck rewrites the INI immediately (presets_api → write_ini), but
the running router keeps serving the table it parsed when it launched. Nothing
in either process reports the gap, so a changed ctx-size silently does nothing
until someone restarts the router — on 2026-08-22 a model ran for an hour at
ctx 96000 while every screen in the app showed the 132000 from the file.

The router does expose the fix: `GET /models?reload=1` re-reads the INI and
unloads any running model whose preset changed. What was missing is knowing
*when* to press it. This module answers that by comparing the INI text on disk
against the argv the router says it would use per model (`status.args`), which
is the router's parsed preset made visible.

Comparison is deliberately literal: our INI keys are llama-server long flags
without the dashes (`ctx-size` → `--ctx-size`), so no field mapping is needed
and a key we have never heard of still gets compared correctly.
"""
from __future__ import annotations

# Flags the router injects per model regardless of the INI: bind address, the
# per-model port it picks at load time, and the id it serves under. Comparing
# them would report drift on every model forever.
_INJECTED = {"host", "port", "alias"}

_TRUE = {"true", "yes", "on", "1", "enabled"}
_FALSE = {"false", "no", "off", "0", "disabled"}


def parse_ini(text: str) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Split the INI into (global `[*]` defaults, {section: {key: value}}).

    Hand-rolled rather than configparser: the file starts with a bare
    `version = 1` line before any section, which configparser rejects outright
    (MissingSectionHeaderError), and `[*]` is a legal section name here.
    """
    global_: dict[str, str] = {}
    sections: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith((";", "#")):
            continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip()
            if name == "*":
                current = global_
            else:
                current = sections.setdefault(name, {})
            continue
        if "=" not in line or current is None:
            continue  # preamble (`version = 1`) or a malformed line
        key, _, value = line.partition("=")
        current[key.strip()] = value.strip()
    return global_, sections


def args_to_map(args: list[str]) -> dict[str, str]:
    """argv → {flag-without-dashes: value}. A flag with no value is "true"."""
    out: dict[str, str] = {}
    i = 0
    while i < len(args):
        tok = args[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key = tok[2:]
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            out[key] = args[i + 1]
            i += 2
        else:
            out[key] = "true"
            i += 1
    return out


def _same(ini_val: str, live_val: str) -> bool:
    a, b = ini_val.strip(), live_val.strip()
    if a == b:
        return True
    la, lb = a.lower(), b.lower()
    if la in _TRUE and lb in _TRUE:
        return True
    if la in _FALSE and lb in _FALSE:
        return True
    # 999 vs 999.0, 0 vs -0: compare numerically when both sides are numbers.
    try:
        return float(a) == float(b)
    except ValueError:
        return False


def ini_drift(ini_text: str, models: list[dict]) -> list[dict]:
    """Per-model differences between the INI on disk and the router's table.

    `models` is the router's /models payload list; each entry needs `id` and
    `status.args`. Models the router has not published args for (never loaded
    under an older build, or absent from the payload) are skipped rather than
    reported — an unknown is not a difference.

    Returns `[{"model", "key", "ini", "live"}]`, sorted, empty when in sync.
    """
    if not models:
        # The router is up but told us nothing — a 502 the caller swallowed, or
        # a table that has not been published yet. Reporting every section as
        # "missing" here would paint the whole page red over a transient.
        return []
    global_, sections = parse_ini(ini_text)
    live_by_id: dict[str, list[str]] = {}
    for m in models:
        args = ((m.get("status") or {}).get("args")) or []
        if args:
            live_by_id[str(m.get("id"))] = args

    out: list[dict] = []
    for name, section in sections.items():
        args = live_by_id.get(name)
        if args is None:
            # A section the router does not serve yet: it was added to the INI
            # after startup. That IS drift, and the model name is the whole
            # message — there is no per-key comparison to make.
            if name not in {str(m.get("id")) for m in models}:
                out.append({"model": name, "key": "(model)", "ini": "present", "live": "missing"})
            continue
        live = args_to_map(args)
        merged = {**global_, **section}
        for key, val in merged.items():
            if key in _INJECTED:
                continue
            live_val = live.get(key)
            if live_val is None:
                out.append({"model": name, "key": key, "ini": val, "live": "—"})
            elif not _same(val, live_val):
                out.append({"model": name, "key": key, "ini": val, "live": live_val})
    out.sort(key=lambda d: (d["model"], d["key"]))
    return out
