from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def run(script: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)


def main() -> int:
    run("01_build_dedup.py")
    run("02_validate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
