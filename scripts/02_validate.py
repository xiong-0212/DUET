from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.io import load_config
from duet_clean.validation import validate_manifests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/protocol.json")
    args = parser.parse_args()
    report = validate_manifests(load_config(args.config))
    print("validation OK" if report["ok"] else "validation FAILED")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
