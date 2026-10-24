from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.io import load_config
from duet_clean.pipeline import build_manifests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/protocol.json")
    args = parser.parse_args()
    config = load_config(args.config)
    summary = build_manifests(config)
    print(json.dumps(summary["manifests"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
