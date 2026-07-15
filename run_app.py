"""
Launch the Path Planning Benchmark web app.

    python run_app.py                         # http://127.0.0.1:8177, opens browser
    python run_app.py --port 9000
    python run_app.py --no-browser            # headless (e.g. on a VM)
    python run_app.py --host 0.0.0.0 --no-browser   # reachable on the machine's network

On a remote VM, prefer an SSH tunnel (`ssh -L 8177:localhost:8177 user@host`) over
binding to 0.0.0.0 — the built-in server has no authentication. See the README.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description='Path Planning Benchmark app')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8177)
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not open the browser automatically.')
    args = parser.parse_args()

    from benchmark_app.server import main as serve
    serve(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == '__main__':
    main()
