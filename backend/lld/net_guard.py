"""Host and Origin validation — the fence around an unauthenticated control API.

LlamaDeck has no login by design: it is a local tool that starts and stops
processes on the machine it runs on. That is defensible only while "local" is
actually enforced, and binding to 127.0.0.1 does **not** enforce it. Two holes
remain open to any web page the user happens to visit:

* **DNS rebinding.** `evil.com` resolves to the attacker's server on the first
  lookup and to `127.0.0.1` on the second. The browser keeps treating the page
  as same-origin with `http://evil.com`, so the attacker's JavaScript can both
  send requests to the local backend *and read the replies*. A firewall,
  loopback binding and the CORS allow-list all sit this one out.
* **Blind CSRF.** A cross-origin `<form>` POST needs no CORS approval to be
  *delivered*; the attacker cannot read the response but does not need to. Most
  of this API's mutating endpoints take no body at all — restart, stop, adopt,
  rebuild — so firing them is the whole attack.

Chained with `PUT /api/settings` (which sets `llama_bin`) and the preset
`argv_override`, either hole is remote code execution as the user. Hence this
middleware.

The rule is the one property both attacks cannot satisfy:

* **Host** must be an IP literal, `localhost`/`*.localhost`, the configured
  bind host, or a name the user listed in `allowed_hosts`. A rebinding attack
  is defined by the browser sending the attacker's *domain* in `Host` — that is
  what keeps the page same-origin with itself — so requiring a literal address
  cuts it, while access by IP (loopback and LAN alike) is untouched.
* **Origin**, on mutating methods, must be a host that would itself pass the
  Host rule. A request with no `Origin` is not a browser form post (curl, the
  CLI, an MCP client), and is left alone; `Origin: null` — sandboxed iframe,
  `file://` — is refused, since that is the obvious way to launder a bad one.

Anything rejected gets a 403 that names the header and says how to allow it, so
a legitimate "I reach it at http://workstation:8770" is one setting away rather
than an unexplained failure.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import os
from urllib.parse import urlsplit

from .settings import SETTINGS_PATH

log = logging.getLogger("lld")

#: Methods that can change state. GET/HEAD/OPTIONS are left to the Host check
#: plus CORS: a cross-origin GET is delivered but its response is unreadable.
_MUTATING = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def hostname_of(value: str) -> str:
    """Strip the port (and IPv6 brackets) from a Host-header-shaped string.

    Accepts what actually turns up: `127.0.0.1:8770`, `[::1]:8770`, `::1`,
    `workstation.local.`, or a bare name. Returns it lowercased and without a
    trailing dot so comparisons are stable.
    """
    v = value.strip().lower()
    if v.startswith("["):
        end = v.find("]")
        return v[1:end] if end != -1 else v[1:]
    # One colon means host:port. More than one and no brackets is a bare IPv6
    # literal, which has no port to strip.
    if v.count(":") == 1:
        v = v.split(":", 1)[0]
    return v.rstrip(".")


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


#: (settings.json mtime_ns, extra host names). The fast path — an IP literal or
#: localhost — never touches this, so the common request reads no file at all.
_extra_cache: tuple[int, frozenset[str]] | None = None


def _extra_hosts() -> frozenset[str]:
    """Host names the user has legitimised: the bind hosts and `allowed_hosts`.

    Read straight from settings.json rather than through `load_settings()`:
    this runs on the request path, and `load_settings()` writes a default file
    and renames a corrupt one aside — side effects that have no business firing
    because someone sent an odd Host header. Cached on mtime, so editing the
    setting takes effect on the next request without a restart.
    """
    global _extra_cache
    try:
        mtime = SETTINGS_PATH.stat().st_mtime_ns
    except OSError:
        mtime = 0
    cached = _extra_cache
    if cached is not None and cached[0] == mtime:
        return cached[1]

    names: set[str] = set()
    try:
        data = json.loads(SETTINGS_PATH.read_text())
    except (OSError, ValueError):
        data = {}
    if isinstance(data, dict):
        for key in ("controller_bind_host", "mcp_bind_host"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                names.add(hostname_of(value))
        listed = data.get("allowed_hosts")
        if isinstance(listed, list):
            for value in listed:
                if isinstance(value, str) and value.strip():
                    names.add(hostname_of(value))
    # `llamadeck serve --host NAME` overrides the settings value for this run.
    env_host = os.environ.get("LLAMADECK_BIND_HOST")
    if env_host:
        names.add(hostname_of(env_host))

    names.discard("")
    _extra_cache = (mtime, frozenset(names))
    return _extra_cache[1]


def host_is_allowed(host: str) -> bool:
    """True if `host` (already stripped to a hostname) may address this app."""
    if not host:
        # HTTP/1.0 with no Host header. Not something a browser sends, and the
        # rebinding attack depends on the header being present and forged.
        return True
    if _is_ip_literal(host):
        return True
    if host == "localhost" or host.endswith(".localhost"):
        return True
    return host in _extra_hosts()


def _origin_is_allowed(origin: str) -> bool:
    o = origin.strip()
    if not o:
        return True
    if o.lower() == "null":
        # Opaque origin: a sandboxed iframe or a file:// page. Nothing in
        # LlamaDeck's own UI produces it, and allowing it would hand back the
        # bypass this middleware exists to close.
        return False
    try:
        host = urlsplit(o).hostname
    except ValueError:
        return False
    if not host:
        return False
    return host_is_allowed(hostname_of(host))


class HostOriginGuard:
    """Raw ASGI middleware, deliberately not `BaseHTTPMiddleware`.

    It has to sit outside everything — the API routers, the /mcp mount and the
    static SPA — and it has to not touch response bodies, because the download
    and build pages stream SSE through here for as long as a job runs.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        host_header: str | None = None
        origin_header: str | None = None
        for key, value in scope.get("headers") or ():
            if key == b"host":
                host_header = value.decode("latin-1")
            elif key == b"origin":
                origin_header = value.decode("latin-1")

        host = hostname_of(host_header or "")
        if not host_is_allowed(host):
            await self._deny(
                scope, send,
                f"Host header '{host}' is not allowed. LlamaDeck only answers to "
                "an IP address, localhost, or a name listed in the 'allowed_hosts' "
                "setting — this blocks DNS-rebinding attacks from web pages. Add "
                f"'{host}' to allowed_hosts in Settings if you reach LlamaDeck by "
                "that name.",
                "host", host,
            )
            return

        method = scope.get("method", "")
        if method in _MUTATING and origin_header is not None:
            if not _origin_is_allowed(origin_header):
                await self._deny(
                    scope, send,
                    f"Cross-origin {method} from '{origin_header.strip()}' is refused. "
                    "LlamaDeck has no authentication, so a state-changing request may "
                    "only come from a page it served itself.",
                    "origin", origin_header.strip(),
                )
                return

        await self.app(scope, receive, send)

    async def _deny(self, scope, send, detail: str, kind: str, value: str) -> None:
        log.warning(
            "blocked %s %s: bad %s %r",
            scope.get("method", "?"), scope.get("path", "?"), kind, value,
        )
        body = json.dumps({"detail": detail}).encode()
        await send({
            "type": "http.response.start",
            "status": 403,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})
