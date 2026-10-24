from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))


SOURCE = ROOT / "data" / "processed" / "graph_seq3d_large_p80_cv5.pt"
OUTPUT = ROOT / "data" / "processed" / "graph_seq3d_shuffle_large_p80_cv5.pt"
AUDIT = ROOT / "results" / "followup" / "shuffled3d" / "graph_audit.json"
SEED_BASE = 20261024


def stable_seed(rna_id: str) -> int:
    digest = hashlib.sha256(rna_id.encode("ascii")).digest()
    return SEED_BASE + int.from_bytes(digest[:4], "little")


def shuffled_structure_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    seed: int,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    sources = edge_index[0].cpu().numpy()
    output = []
    for source in sorted(set(sources.tolist())):
        count = int((sources == source).sum())
        candidates = np.asarray(
            [target for target in range(num_nodes) if target != source],
            dtype=int,
        )
        replace = count > len(candidates)
        targets = rng.choice(candidates, size=count, replace=replace)
        output.extend((source, int(target)) for target in targets)
    return torch.tensor(output, dtype=torch.long).t().contiguous()


def main() -> int:
    payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    relation_names = list(payload["rel_names"])
    structure_relation = relation_names.index("struct3d")
    output = dict(payload)
    output["graphs"] = {}
    output["source_asset"] = str(SOURCE.relative_to(ROOT))
    output["shuffle"] = {
        "degree_preserving": True,
        "self_loops": False,
        "seed_base": SEED_BASE,
        "seed_rule": "sha256(rna_id)[:4] little-endian + seed_base",
    }
    audit = {
        "source": str(SOURCE.relative_to(ROOT)),
        "output": str(OUTPUT.relative_to(ROOT)),
        "splits": {},
        "errors": [],
        "duplicate_targets_required": 0,
    }
    for split, graphs in payload["graphs"].items():
        rewritten = []
        degree_checks = 0
        sequence_checks = 0
        for graph in graphs:
            structure_mask = graph.edge_type == structure_relation
            sequence_mask = graph.edge_type != structure_relation
            original_structure = graph.edge_index[:, structure_mask]
            original_sequence = graph.edge_index[:, sequence_mask].clone()
            shuffled_structure = shuffled_structure_edges(
                original_structure,
                int(graph.num_nodes),
                stable_seed(str(graph.rna_id)),
            )
            if shuffled_structure.size(1) > 0:
                pair_counts = Counter(
                    zip(
                        shuffled_structure[0].tolist(),
                        shuffled_structure[1].tolist(),
                    )
                )
                if max(pair_counts.values()) > 1:
                    audit["duplicate_targets_required"] += 1
            original_degrees = Counter(original_structure[0].tolist())
            shuffled_degrees = Counter(shuffled_structure[0].tolist())
            if original_degrees != shuffled_degrees:
                audit["errors"].append(
                    f"{split}/{graph.rna_id}: out-degree mismatch"
                )
            degree_checks += 1
            if bool((shuffled_structure[0] == shuffled_structure[1]).any()):
                audit["errors"].append(
                    f"{split}/{graph.rna_id}: shuffled self-loop"
                )
            if not torch.equal(
                original_sequence,
                graph.edge_index[:, sequence_mask],
            ):
                audit["errors"].append(
                    f"{split}/{graph.rna_id}: sequence edge mismatch"
                )
            sequence_checks += 1
            rewritten.append(
                Data(
                    x=graph.x,
                    edge_index=torch.cat(
                        [
                            original_sequence,
                            shuffled_structure,
                        ],
                        dim=1,
                    ),
                    edge_type=torch.cat(
                        [
                            graph.edge_type[sequence_mask],
                            torch.full(
                                (shuffled_structure.size(1),),
                                structure_relation,
                                dtype=torch.long,
                            ),
                        ]
                    ),
                    y=graph.y,
                    rna_id=graph.rna_id,
                    num_nodes=graph.num_nodes,
                    num_relations=graph.num_relations,
                )
            )
        output["graphs"][split] = rewritten
        audit["splits"][split] = {
            "graphs": len(graphs),
            "nodes": int(sum(graph.num_nodes for graph in graphs)),
            "edges": int(
                sum(graph.edge_index.size(1) for graph in rewritten)
            ),
            "degree_checks": degree_checks,
            "sequence_checks": sequence_checks,
        }
    if audit["errors"]:
        raise ValueError("; ".join(audit["errors"][:20]))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, OUTPUT)
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit["splits"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
