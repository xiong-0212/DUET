from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "duet_model"))

from duet_clean.cobra import CoBRAHead
from duet_clean.large_results import (
    output_path,
    save_standard_result,
    standardize_result,
)
from duet_clean.metrics import compute_metrics, safe_torch_load
from rgcn_model import build_model


EXTERNAL_SPLITS = ["external_jiang", "time_dependent", "hariboss_test"]
DEFAULT_SEEDS = [42, 0, 1, 2, 3, 7, 11, 13, 17, 23]
GRAPH_METHODS = {
    "duet_graph": "graph_rnafm",
    "duet_tcn": "dual_rinalmo",
    "duet_dense": "dual_rinalmo",
    "graph_seq3d": "seq3d_rnafm",
    "graph_seqss3d": "seqss3d_rnafm",
    "rgcn_ernie": "seq3d_ernie",
}
GRASSP_METHODS = {
    "grassp_ernie": ("grassp_ernie", 768),
    "grassp_rnafm": ("grassp_rnafm", 640),
}
METHODS = ["cobra", *GRAPH_METHODS, *GRASSP_METHODS]


def predict_cobra(
    checkpoint_path: Path,
    split_records: dict[str, dict[str, Any]],
    embeddings: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    checkpoint = safe_torch_load(checkpoint_path, device)
    model = CoBRAHead().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    output = {}
    with torch.no_grad():
        for rna_id, record in split_records.items():
            x = embeddings[rna_id].float().to(device)
            probability = torch.softmax(model(x), dim=-1)[:, 1].cpu().numpy()
            output[rna_id] = {
                "prob": probability.tolist(),
                "label": list(record["labels"]),
            }
    return output


def predict_graph(
    checkpoint_path: Path,
    graphs: list[Any],
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    checkpoint = safe_torch_load(checkpoint_path, device)
    config = dict(checkpoint["config"]["model"])
    config["in_dim"] = int(graphs[0].x.size(1))
    config["num_relations"] = int(checkpoint["num_relations"])
    model = build_model(config, int(checkpoint["num_relations"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    output = {}
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            logits = model(
                graph.x,
                graph.edge_index,
                graph.edge_type,
                torch.zeros(graph.x.size(0), dtype=torch.long, device=device),
                edge_weight=(
                    graph.edge_weight
                    if getattr(graph, "edge_weight", None) is not None
                    else None
                ),
            )
            if isinstance(logits, tuple):
                logits = logits[0]
            output[graph.rna_id] = {
                "prob": torch.sigmoid(logits).float().cpu().numpy().tolist(),
                "label": graph.y.cpu().numpy().astype(int).tolist(),
            }
    return output


def predict_grassp(
    checkpoint_path: Path,
    graphs: list[Any],
    rna_dim: int,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    from grassp_model.RNABP import HybridRNABindingSiteModel

    checkpoint = safe_torch_load(checkpoint_path, device)
    model_config = dict(checkpoint.get("config", {}).get("model", {}))
    model = HybridRNABindingSiteModel(
        rna_dim=rna_dim,
        ss_dim=4,
        hidden=int(model_config.get("hidden", 86)),
        dropout=float(model_config.get("dropout", 0.6)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    output = {}
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
            logits = model(graph.x, graph.ss_emb, graph.edge_index, batch)
            output[graph.rna_id] = {
                "prob": torch.sigmoid(logits).float().cpu().numpy().tolist(),
                "label": graph.y.cpu().numpy().astype(int).tolist(),
            }
    return output


def split_chain_metrics(
    per_chain: dict[str, dict[str, Any]],
    threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = np.concatenate(
        [np.asarray(value["label"], dtype=int) for value in per_chain.values()]
    )
    probabilities = np.concatenate(
        [np.asarray(value["prob"], dtype=float) for value in per_chain.values()]
    )
    return (
        compute_metrics(labels, probabilities, threshold),
        compute_metrics(labels, probabilities, 0.5),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    external = torch.load(
        ROOT / "data" / "processed" / "smartpocket_external_graphs.pt",
        map_location="cpu",
        weights_only=False,
    )
    rnafm_payload = torch.load(
        ROOT / "data" / "external" / "emb_rnafm.pt",
        map_location="cpu",
        weights_only=False,
    )
    rinalmo_payload = torch.load(
        ROOT / "data" / "external" / "emb_rinalmo.pt",
        map_location="cpu",
        weights_only=False,
    )
    rnafm = rnafm_payload.get("embeddings", rnafm_payload)
    rinalmo = rinalmo_payload.get("embeddings", rinalmo_payload)
    external_records = json.loads(
        (ROOT / "data" / "external" / "external_sets.json").read_text(
            encoding="utf-8"
        )
    )["splits"]

    for method in args.methods:
        for seed in args.seeds:
            final = json.loads(
                output_path("final", method, seed).read_text(encoding="utf-8")
            )
            checkpoint = ROOT / final["checkpoint"]
            threshold = float(final["threshold"])
            test = {}
            test_05 = {}
            per_chain_all = {}
            for split in EXTERNAL_SPLITS:
                if method == "cobra":
                    per_chain = predict_cobra(
                        checkpoint,
                        external_records[split],
                        rinalmo,
                        device,
                    )
                elif method in GRAPH_METHODS:
                    graph_name = GRAPH_METHODS[method]
                    graphs = [
                        graph
                        for graph in external["graphs"][graph_name]
                        if graph.external_set == split
                    ]
                    per_chain = predict_graph(checkpoint, graphs, device)
                else:
                    graph_name, rna_dim = GRASSP_METHODS[method]
                    graphs = [
                        graph
                        for graph in external["graphs"][graph_name]
                        if graph.external_set == split
                    ]
                    per_chain = predict_grassp(
                        checkpoint,
                        graphs,
                        rna_dim,
                        device,
                    )
                test[split], test_05[split] = split_chain_metrics(
                    per_chain,
                    threshold,
                )
                per_chain_all[split] = per_chain
            result = {
                "method": method,
                "seed": seed,
                "threshold": threshold,
                "test": test,
                "test_at_0.5": test_05,
                "per_chain": per_chain_all,
            }
            path = (
                ROOT
                / "results"
                / "large_protocol"
                / "external"
                / f"{method}_seed{seed}.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                f"[external] {method} seed={seed} "
                + " ".join(
                    f"{split}={test[split]['MCC']:.3f}"
                    for split in ["external_jiang", "time_dependent"]
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
