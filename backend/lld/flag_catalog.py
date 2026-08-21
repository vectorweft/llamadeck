"""Every flag the *configured* llama-server binary actually accepts.

Parsed from `llama-server --help`, not from a table we maintain. That matters
because the command box lets a user type any flag they like: the only honest
way to tell "you typed a flag this build does not have" from "LlamaDeck does
not know that flag yet" is to ask the binary. A rebuild that adds flags makes
them known here on the next probe, with no code change.

Layout of the help output (llama.cpp, `common/arg.cpp`):

    ----- common params -----

    -t,    --threads N                      number of CPU threads to use ...
                                            (env: LLAMA_ARG_THREADS)
    --jinja, --no-jinja                     whether to use jinja template ...
    -rea,  --reasoning [on|off|auto]        Use reasoning/thinking in the chat ...
    --chat-template-file JINJA_TEMPLATE_FILE
                                            set a file ...

Column 40 starts the description; continuation lines are indented to it. A
spec longer than 40 characters pushes its description onto the next line.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path

from .procutil import run_capture

_DESC_COL = 40
_SECTION_RE = re.compile(r"^-{3,}\s*(?P<name>.+?)\s*-{3,}$")
_ENV_RE = re.compile(r"\(env:\s*(?P<env>[A-Z0-9_]+)\)")

# Placeholders that mean "this flag takes a value". A bracketed placeholder
# ("[on|off|auto]") is optional to llama.cpp — `-fa` alone is legal — so it is
# recorded separately; the tokenizer only consumes a following value when that
# token does not itself look like a flag.
_OPTIONAL_VALUE_RE = re.compile(r"^\[.*\]$")


@dataclass
class FlagSpec:
    """One row of `--help`: every spelling of a flag plus what it takes."""

    names: list[str]                       # ["-t", "--threads"] — all spellings
    placeholder: str | None = None         # "N", "STRING", "[on|off|auto]", None
    help: str = ""
    env: str | None = None
    section: str = ""

    @property
    def canonical(self) -> str:
        """Longest `--long-form`, falling back to the first spelling.

        Negations are skipped: `--jinja, --no-jinja` is one row, and naming it
        after `--no-jinja` would read as if enabling jinja were the odd case.
        """
        longs = [n for n in self.names if n.startswith("--") and not n.startswith("--no-")]
        if not longs:
            longs = [n for n in self.names if n.startswith("--")]
        return max(longs, key=len) if longs else self.names[0]

    @property
    def takes_value(self) -> bool:
        return self.placeholder is not None

    @property
    def value_required(self) -> bool:
        return self.takes_value and not _OPTIONAL_VALUE_RE.match(self.placeholder or "")

    @property
    def choices(self) -> list[str]:
        """Enum values from a `[on|off|auto]`-style placeholder, else []."""
        p = self.placeholder or ""
        if not _OPTIONAL_VALUE_RE.match(p):
            return []
        return [c for c in p.strip("[]").split("|") if c]

    def to_dict(self) -> dict:
        return {
            "names": self.names,
            "canonical": self.canonical,
            "placeholder": self.placeholder,
            "takes_value": self.takes_value,
            "value_required": self.value_required,
            "choices": self.choices,
            "help": self.help,
            "env": self.env,
            "section": self.section,
        }


@dataclass
class FlagCatalog:
    flags: list[FlagSpec] = field(default_factory=list)
    by_name: dict[str, FlagSpec] = field(default_factory=dict)
    # False when the binary could not be run at all. Callers must then treat
    # every flag as possibly-valid instead of reporting the user's command as
    # wrong — an unqueryable binary is not evidence against the command.
    available: bool = False

    def get(self, name: str) -> FlagSpec | None:
        return self.by_name.get(name)

    def known(self, name: str) -> bool:
        return name in self.by_name

    def suggest(self, name: str, n: int = 3) -> list[str]:
        """Closest known spellings for a flag that is not in the catalog."""
        if not self.by_name:
            return []
        return difflib.get_close_matches(name, list(self.by_name), n=n, cutoff=0.7)

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "flags": [f.to_dict() for f in self.flags],
        }


def parse_help(text: str) -> FlagCatalog:
    """Parse `llama-server --help` into a catalog. Pure — no I/O.

    Output with no recognizable flag rows yields an `available=False` catalog:
    a binary that printed something we cannot read is in exactly the same
    position as one that would not run, and neither may be used as evidence
    that a user's flag does not exist.
    """
    catalog = FlagCatalog()
    section = ""
    current: FlagSpec | None = None
    desc: list[str] = []

    def flush() -> None:
        nonlocal current, desc
        if current is None:
            return
        body = " ".join(part.strip() for part in desc if part.strip())
        current.help = body.strip()
        m = _ENV_RE.search(body)
        if m:
            current.env = m.group("env")
            current.help = _ENV_RE.sub("", body).strip()
        catalog.flags.append(current)
        for n in current.names:
            catalog.by_name.setdefault(n, current)
        current, desc = None, []

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        sec = _SECTION_RE.match(line.strip())
        if sec:
            flush()
            section = sec.group("name")
            continue
        if line[0].isspace():
            # Continuation of the current description (indented to _DESC_COL).
            if current is not None:
                desc.append(line)
            continue

        # The description starts at _DESC_COL only when the spec fits before
        # it — a longer spec (`-ot, --override-tensor <pattern>=<type>,...`)
        # owns the whole line and its description begins on the next one.
        # Splitting those at the column truncates the value placeholder.
        split_here = len(line) > _DESC_COL and line[_DESC_COL - 1] == " "
        spec_text = line[:_DESC_COL].strip() if split_here else line.strip()
        rest = line[_DESC_COL:].strip() if split_here else ""
        names, placeholder = _parse_spec(spec_text)
        if not names:
            # A stray line ("WARNING: radv is not a conformant …") — ignore it,
            # but do not let it swallow the previous flag's description.
            continue
        flush()
        current = FlagSpec(names=names, placeholder=placeholder, section=section)
        if rest:
            desc.append(rest)

    flush()
    catalog.available = bool(catalog.flags)
    return catalog


# A name is a dash followed by a letter. `-2` in a placeholder like "-1,-2"
# is not a flag, and must not be mistaken for one when splitting on commas.
_NAME_PIECE_RE = re.compile(r"^--?[A-Za-z]")


def _parse_spec(spec: str) -> tuple[list[str], str | None]:
    """"-t,    --threads N" -> (["-t", "--threads"], "N").

    The comma is both the separator between spellings AND a character inside
    several placeholders, which is why the split cannot simply be discarded
    once it stops producing names. It used to be, and the placeholders that
    matter most were the ones cut in half:

        -dev,  --device <dev1,dev2,..>     ->  "<dev1"
        -ts,   --tensor-split N0,N1,N2,... ->  "N0"
        --tools TOOL1,TOOL2,...            ->  "TOOL1"

    which the editor then showed as the value to type. Everything from the
    first piece that is not a name belongs to the placeholder.
    """
    names: list[str] = []
    placeholder: str | None = None
    pieces = spec.split(",")
    for i, piece in enumerate(pieces):
        stripped = piece.strip()
        if not _NAME_PIECE_RE.match(stripped):
            if placeholder is not None:
                tail = [p.strip() for p in pieces[i:]]
                placeholder = ",".join([placeholder, *tail])
            break
        head, _, rest = stripped.partition(" ")
        names.append(head)
        rest = rest.strip()
        if rest and placeholder is None:
            placeholder = rest
    return names, placeholder


# --- probing the configured binary -------------------------------------------
# `--help` output only changes when the binary does, so the cache key is the
# binary's identity. No TTL: unlike the device list there is nothing live here.
_cache: dict[tuple[str, int, int], FlagCatalog] = {}
_EMPTY = FlagCatalog(available=False)


def invalidate_cache() -> None:
    """Drop memoized catalogs — call after a rebuild swaps the binary."""
    _cache.clear()


async def get_flag_catalog(binary: str, timeout: float = 20.0) -> FlagCatalog:
    """Flags `binary` accepts, or an `available=False` catalog when it cannot
    be run. Never raises: a missing binary must not break the preset editor."""
    if not binary:
        return _EMPTY
    try:
        st = Path(binary).stat()
    except OSError:
        return _EMPTY
    key = (binary, st.st_size, st.st_mtime_ns)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    res = await run_capture([binary, "--help"], timeout=timeout)
    if not res.ok:
        return _EMPTY
    catalog = parse_help(res.text)
    if not catalog.flags:
        return _EMPTY
    _cache[key] = catalog
    return catalog


# --- can this flag actually be applied to this preset? -----------------------
# Feature cards (What's New) list the flags a llama.cpp change *touches*, as
# read out of release notes by an LLM. That is not the same as "flags you
# should add": a Vulkan diagnostics card lists --model, --device, --ctx-size
# because they appear in its repro command. Appending those to extra_flags
# produces `--model` with no value — an unstartable preset — or a second
# `--ctx-size` that shadows the field. So every advertised flag is classified
# before anything is offered to the user.


def _managed_flags() -> set[str]:
    """Flags LlamaDeck renders from preset fields. Adding one of these to
    extra_flags is never right: at best it duplicates the field, at worst it
    silently overrules it (llama.cpp keeps the last occurrence)."""
    from .argv import _FIELD_FLAGS, _ROUTER_VALUE_FLAGS

    managed = {flag for _f, flag, _k in _FIELD_FLAGS}
    managed |= set(_ROUTER_VALUE_FLAGS)
    managed |= {
        "-dev", "--device", "-ts", "--tensor-split", "--rpc",
        "--models-autoload", "--no-models-autoload",
        "--jinja", "--no-jinja", "--metrics", "--slots",
        "--cont-batching", "-cb", "--no-cont-batching",
    }
    return managed


def flags_missing_values(catalog: FlagCatalog, argv: list[str]) -> list[dict]:
    """Flags in `argv` that need a value and were given none.

    llama-server refuses the whole command line for this and exits before it
    opens a log the user would think to read:

        error while handling argument "--tools": expected value for argument

    which reaches the UI as nothing more than "the preset did not start". The
    catalog already knows every flag's placeholder, so this is knowable before
    anything is spawned.

    Deliberately conservative — a false positive would block a command that
    works. A following token counts as the value unless the catalog says it is
    itself a flag, so `--seed -1` and `--tensor-split -1,0` are not reported;
    only a flag followed by another known flag, or by nothing at all, is.
    Returns [] whenever the binary could not be queried.
    """
    if not catalog.available:
        return []
    out: list[dict] = []
    seen: set[str] = set()
    tokens = argv[1:]
    for i, tok in enumerate(tokens):
        if not tok.startswith("-") or tok == "-" or "=" in tok:
            continue
        spec = catalog.get(tok)
        if spec is None or not spec.value_required or tok in seen:
            continue
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if nxt is not None and not catalog.known(nxt.split("=", 1)[0]):
            continue
        seen.add(tok)
        out.append({
            "flag": tok,
            "placeholder": spec.placeholder or "",
            "help": spec.help,
        })
    return out


async def classify_flags(cfg, binary: str, flags: list[str]) -> dict[str, list[str]]:
    """Sort advertised flags into what can safely be applied to `cfg`.

    * `actionable`  — real in this build, absent from the command, and either
                      valueless or carrying its own value: safe to append.
    * `present`     — the command already passes it.
    * `managed`     — LlamaDeck owns it as a field; change the field instead.
    * `needs_value` — a real flag, but it takes a value we have no way to guess.
    * `unknown`     — this build has no such flag (empty when the binary could
                      not be queried, which is not evidence against the flag).
    """
    from .argv import _canonical, to_argv

    out: dict[str, list[str]] = {
        "actionable": [], "present": [], "managed": [], "needs_value": [], "unknown": [],
    }
    try:
        argv = to_argv(cfg, binary)
    except Exception:  # noqa: BLE001 — a broken override must not break hints
        argv = []
    in_command = {_canonical(t) or t for t in argv if t.startswith("-")}
    managed = _managed_flags()
    catalog = await get_flag_catalog(binary)

    for entry in flags:
        entry = (entry or "").strip()
        if not entry.startswith("-"):
            continue
        head, _, inline = entry.partition("=")
        head = head.split()[0]
        has_value = bool(inline) or len(entry.split()) > 1
        canon = _canonical(head) or head
        spec = catalog.get(head) or catalog.get(canon)
        if canon in managed or head in managed:
            out["managed"].append(entry)
        elif canon in in_command or head in in_command:
            out["present"].append(entry)
        elif catalog.available and spec is None:
            out["unknown"].append(entry)
        elif spec is not None and spec.value_required and not has_value:
            out["needs_value"].append(entry)
        else:
            out["actionable"].append(entry)
    return out
