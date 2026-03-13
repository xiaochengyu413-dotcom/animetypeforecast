from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SITE_DIR = SCRIPT_DIR / "web_dashboard"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the interactive dashboard as a local website.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--site-dir", type=Path, default=DEFAULT_SITE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.site_dir.exists():
        raise FileNotFoundError(
            f"Site directory does not exist: {args.site_dir}. Build the dashboard first."
        )

    handler = partial(SimpleHTTPRequestHandler, directory=str(args.site_dir))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    address = f"http://{args.host}:{args.port}/"
    print(f"serving dashboard at {address}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
