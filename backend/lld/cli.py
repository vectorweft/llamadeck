from __future__ import annotations

import argparse
import os

import uvicorn

from .settings import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llamadeck", description="LlamaDeck — the control deck for llama.cpp"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="Run the controller daemon")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("--reload", action="store_true", help="Auto-reload on code change")

    args = parser.parse_args()

    if args.cmd == "serve":
        s = load_settings()
        host = args.host or s.controller_bind_host
        port = args.port or s.controller_bind_port
        # The app module loads settings on its own and cannot see these flags —
        # pass the effective bind through so its startup banner is accurate.
        os.environ["LLAMADECK_BIND_HOST"] = str(host)
        os.environ["LLAMADECK_BIND_PORT"] = str(port)
        uvicorn.run(
            "lld.main:app",
            host=host,
            port=port,
            reload=args.reload,
            log_level="info",
            # Without a cap, "graceful" means "forever": the dashboard holds an
            # SSE metrics stream open for as long as its tab is, and uvicorn
            # waits for every open connection before it will exit. A SIGTERM
            # from systemd, a Ctrl-C in the launcher terminal, or the restart
            # button therefore hung indefinitely whenever anyone had the UI
            # open — which is essentially always. Give live connections ten
            # seconds to drain, then go.
            timeout_graceful_shutdown=10,
        )


if __name__ == "__main__":
    main()
