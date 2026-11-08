"""Launch the KAIROS API and UI."""

import argparse

import uvicorn


def main() -> None:
    p = argparse.ArgumentParser(description="Launch the KAIROS decision engine")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8300)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()
    uvicorn.run("api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()