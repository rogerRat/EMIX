#!/usr/bin/env python3
"""Single entry point for supported experiment protocols."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = {"multi-v0": "run_multi_v0.py", "multi-v1-scratch": "run_multi_v1_scratch.py", "v0-pi05-paramsearch": "run_v0_pi05_paramsearch.py", "v0-pi05-100k": "run_v0_pi05_100k.py", "v0-pi05-continue-400k": "run_v0_pi05_continue_400k.py", "v0-pi05-accum300k": "run_v0_pi05_accum300k.py", "pi05-baai-baseline": "run_pi05_baai_baseline.py"}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", choices=PROTOCOLS)
    parser.add_argument("--mode", required=True, choices=("prepare", "mock", "run"))
    args = parser.parse_args()
    os.chdir(ROOT)
    os.execv(sys.executable, [sys.executable, "-u", str(ROOT / "scripts" / PROTOCOLS[args.protocol]), "--mode", args.mode])

if __name__ == "__main__":
    main()
