from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.identity import IdentityEngine
from duet_clean.io import load_config
from duet_clean.io import write_json
from duet_clean.large_protocol import (
    LARGE_MANIFEST,
    build_large_protocol,
    cluster_leakage_audit,
    write_large_protocol,
)


def main() -> int:
    config = load_config("configs/protocol.json")
    manifest = build_large_protocol(
        config,
        name=LARGE_MANIFEST,
        threshold=0.8,
        n_folds=5,
    )
    output = write_large_protocol(manifest)
    leakage = cluster_leakage_audit(
        manifest,
        0.8,
        IdentityEngine.from_config(config),
    )
    leakage_path = (
        ROOT / "results" / "large_protocol" / f"{LARGE_MANIFEST}_leakage.json"
    )
    leakage_path.write_text(
        json.dumps(leakage, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if leakage["n_violations"]:
        raise ValueError(
            f"{leakage['n_violations']} cross-fold homology violations"
        )
    environment = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import numpy
        import scipy
        import torch

        environment.update(
            {
                "numpy": numpy.__version__,
                "scipy": scipy.__version__,
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
            }
        )
    except ImportError as error:
        environment["dependency_error"] = str(error)
    write_json("results/large_protocol/environment.json", environment)
    print(
        json.dumps(
            {
                "manifest": str(output.relative_to(ROOT)),
                "counts": manifest["counts"],
                "clusters": manifest["audit"]["cluster_count"],
                "largest_cluster": manifest["audit"]["largest_cluster"],
                "fold_sizes": manifest["audit"]["fold_sizes"],
                "leakage_violations": leakage["n_violations"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
