"""SophiaAgent Web server entry point.

Usage:
    python run_web.py              # Start on default port 8080
    python run_web.py --port 3000  # Custom port
"""

import argparse
import logging
import uvicorn

from sophia.config import Config
from sophia.lifecycle import install_process_lifecycle_hooks
from sophia.web import create_app


def main():
    install_process_lifecycle_hooks()
    parser = argparse.ArgumentParser(description="SophiaAgent Web Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = Config.load(args.config)
    app = create_app(config)

    print(f"\nSophiaAgent Web Server")
    print(f"  Model: {config.model.name}")
    print(f"  Workspace: {config.session.workspace}")
    print(f"  URL: http://{args.host}:{args.port}")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
