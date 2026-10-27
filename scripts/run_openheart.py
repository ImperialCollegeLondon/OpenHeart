#!/usr/bin/env python3
"""Run `openheart run` with either the HCM or the DCM model weights.

Usage:
    python run_openheart.py cases/ -o out/ --weights hcm --folds 1
    python run_openheart.py cases/ -o out/ --weights dcm --realign --jobs 4
    python run_openheart.py cases/ -o out/ --weights dcm --dry-run

Every argument except --weights and --dry-run is passed unchanged to
`openheart run`.

Set the two model folders below (or with env OPENHEART_HCM_MODEL /
OPENHEART_DCM_MODEL, which take precedence).
"""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

# Folder holding each model's weights (the one passed to `openheart run --model`).
MODELS = {
    "hcm": "/path/to/HCM_model",
    "dcm": "/path/to/DCM_model",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="openheart run with a choice of HCM or DCM model weights",
        epilog="Any other arguments are forwarded to `openheart run`.",
    )
    parser.add_argument("--weights", choices=sorted(MODELS), required=True,
                        help="which model to segment with")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the command without running it")
    args, passthrough = parser.parse_known_args()

    if "--model" in passthrough:
        parser.error("use --weights instead of --model")

    model = Path(os.environ.get(f"OPENHEART_{args.weights.upper()}_MODEL")
                 or MODELS[args.weights])
    if not args.dry_run and not model.is_dir():
        sys.exit(f"error: {args.weights.upper()} model folder not found: {model}")

    cmd = ["openheart", "run", *passthrough, "--model", str(model)]
    print(f"[{args.weights.upper()}] " + shlex.join(cmd), flush=True)
    if args.dry_run:
        return 0
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
