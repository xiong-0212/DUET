from __future__ import annotations

import sys
from pathlib import Path

import torch
import numpy as np
from torch_geometric.data import Data

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.experiments import manifest_splits
from duet_clean.grassp import load_processed_chains, to_grassp_candidates
from duet_clean.grassp import PROCESSED_PATHS
from duet_clean.io import project_path


SPLITS = ["TE18", "TL12", "JL10", "val", "train"]
SOURCE = "data/processed/graphs_graph_homology_p80.pt"


def structure_graph_for(rna_id: str, chains: dict[str, Data]) -> Data:
    for candidate in to_grassp_candidates(rna_id):
        if candidate in chains:
            return chains[candidate]
    raise KeyError(f"missing structural graph for {rna_id}")


def shuffled_edges(edge_index: torch.Tensor, num_nodes: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    sources = edge_index[0].cpu().numpy()
    targets = edge_index[1].cpu().numpy()
    out = []
    for source in sorted(set(sources.tolist())):
        k = int((sources == source).sum())
        candidates = [value for value in range(num_nodes) if value != source]
        chosen = rng.choice(
            candidates,
            size=k,
            replace=k > len(candidates),
        )
        out.extend((source, int(target)) for target in chosen)
    return torch.tensor(out, dtype=torch.long).t().contiguous()


def main() -> int:
    source = torch.load(project_path(SOURCE), map_location="cpu", weights_only=False)
    records, manifest = manifest_splits("homology_p80")
    structure = {
        split: load_processed_chains(PROCESSED_PATHS[split])
        for split in SPLITS
    }

    variants = {
        "seq3d": ["seq", "struct3d"],
        "seqss3d": ["seq", "ss", "struct3d"],
        "seq3d_shuffle": ["seq", "struct3d"],
        "seq3d_ernie": ["seq", "struct3d"],
    }
    for tag, relation_names in variants.items():
        output = dict(source)
        output["graphs"] = {}
        output["rel_names"] = relation_names
        output["num_relations"] = len(relation_names)
        output["structure_source"] = "GRASSP released processed 3D kNN graphs"
        output["clean_manifest"] = "homology_p80"
        output["manifest_sha256"] = manifest["manifest_sha256"]

        for split in SPLITS:
            by_id = {graph.rna_id: graph for graph in source["graphs"][split]}
            selected = []
            for record in records[split]:
                graph = by_id.get(record.rna_id)
                if graph is None:
                    raise KeyError(f"missing RNA-FM graph {split}/{record.rna_id}")
                structural = structure_graph_for(record.rna_id, structure[split])
                if structural.x.size(0) != graph.x.size(0):
                    raise ValueError(f"{record.rna_id}: structural node mismatch")

                edge_parts = []
                type_parts = []
                for relation_name in relation_names:
                    if relation_name == "seq":
                        mask = graph.edge_type == source["rel_names"].index("seq")
                        edge_part = graph.edge_index[:, mask]
                    elif relation_name == "ss":
                        mask = graph.edge_type == source["rel_names"].index("ss")
                        edge_part = graph.edge_index[:, mask]
                    elif relation_name == "struct3d":
                        edge_part = structural.edge_index
                        if tag == "seq3d_shuffle":
                            edge_part = shuffled_edges(
                                edge_part,
                                graph.x.size(0),
                                seed=20261024 + record.ordinal,
                            )
                    else:
                        raise ValueError(relation_name)
                    edge_parts.append(edge_part)
                    type_parts.append(
                        torch.full(
                            (edge_part.size(1),),
                            relation_names.index(relation_name),
                            dtype=torch.long,
                        )
                    )
                edge_index = torch.cat(edge_parts, dim=1)
                edge_type = torch.cat(type_parts)
                selected.append(
                    Data(
                        x=structural.x if tag == "seq3d_ernie" else graph.x,
                        edge_index=edge_index,
                        edge_type=edge_type,
                        y=graph.y,
                        rna_id=graph.rna_id,
                        num_nodes=graph.x.size(0),
                        num_relations=len(relation_names),
                    )
                )
            output["graphs"][split] = selected
            print(f"[{tag}] {split}: {len(selected)}")

        out_path = project_path(f"data/processed/graphs_graph_{tag}_homology_p80.pt")
        torch.save(output, out_path)
        print(f"[written] {out_path.relative_to(project_path('.'))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
