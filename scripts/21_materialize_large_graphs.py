from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.grassp import PROCESSED_PATHS, load_processed_chains
from duet_clean.large_protocol import LARGE_MANIFEST, load_large_manifest
from duet_clean.models import Record


SPLIT_ORDER = ["TE18", "TL12", "JL10", "RB9", "pool"]
STRUCTURE_SPLITS = ["TE18", "TL12", "JL10", "pool"]
OUTPUT_PREFIX = "large_p80_cv5"
RNAFM_SOURCE = "data/processed/graphs_seq_ss_clean_source.pt"
DUAL_SOURCE = "data/processed/graphs_dual_clean_source.pt"
RAW_SPLITS = [
    "train",
    "val",
    "TE18",
    "TL12",
    "JL10",
    "RB9",
    "hariofficial_test",
]


def _graph_index(source: dict[str, Any]) -> dict[str, list[Data]]:
    index: dict[str, list[Data]] = defaultdict(list)
    for split in RAW_SPLITS:
        for graph in source["graphs"].get(split, []):
            index[str(graph.rna_id)].append(graph)
    return dict(index)


def _select_graph(record: Record, index: dict[str, list[Data]]) -> Data:
    candidates = index.get(record.rna_id, [])
    for graph in candidates:
        if graph.x.size(0) != len(record.sequence):
            continue
        expected = torch.tensor(
            [int(value) for value in record.labels],
            dtype=graph.y.dtype,
        )
        if torch.equal(graph.y.view(-1).cpu(), expected):
            return graph
    raise KeyError(
        f"no matching graph for {record.rna_id} "
        f"({len(record.sequence)} residues, {len(candidates)} candidates)"
    )


def _selected_splits(
    records: dict[str, list[Record]],
    index: dict[str, list[Data]],
) -> dict[str, list[Data]]:
    return {
        split: [_select_graph(record, index) for record in records[split]]
        for split in SPLIT_ORDER
    }


def _structure_index() -> dict[str, Data]:
    index: dict[str, Data] = {}
    for split, path in PROCESSED_PATHS.items():
        for name, graph in load_processed_chains(path).items():
            if name in index:
                if graph.x.size(0) != index[name].x.size(0):
                    raise ValueError(f"conflicting structural graph size for {name}")
                continue
            index[name] = graph
    return index


def _to_grassp_candidates(rna_id: str) -> list[str]:
    pdb, _, chain = rna_id.partition("_")
    candidates = [f"{pdb}{chain}".upper()]
    if chain.upper() == "A":
        candidates.append(pdb.upper())
    return candidates


def _select_structure(record: Record, index: dict[str, Data]) -> Data:
    for candidate in _to_grassp_candidates(record.rna_id):
        graph = index.get(candidate)
        if graph is None:
            continue
        if graph.x.size(0) != len(record.sequence):
            raise ValueError(f"{record.rna_id}: structural length mismatch")
        expected = torch.tensor(
            [int(value) for value in record.labels],
            dtype=graph.y.dtype,
        )
        if not torch.equal(graph.y.view(-1).cpu(), expected):
            raise ValueError(f"{record.rna_id}: structural labels mismatch")
        return graph
    raise KeyError(f"missing structural graph for {record.rna_id}")


def _relation_edges(
    graph: Data,
    source_rel_names: list[str],
    relation_names: list[str],
    structural: Data | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    edges = []
    types = []
    for relation_name in relation_names:
        if relation_name == "struct3d":
            if structural is None:
                raise ValueError("struct3d requires a structural graph")
            edge_part = structural.edge_index
        else:
            relation_index = source_rel_names.index(relation_name)
            mask = graph.edge_type == relation_index
            edge_part = graph.edge_index[:, mask]
        edges.append(edge_part)
        types.append(
            torch.full(
                (edge_part.size(1),),
                relation_names.index(relation_name),
                dtype=torch.long,
            )
        )
    return torch.cat(edges, dim=1), torch.cat(types)


def _save_object(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> int:
    records, manifest = load_large_manifest(LARGE_MANIFEST)
    rnafm_source = torch.load(
        ROOT / RNAFM_SOURCE,
        map_location="cpu",
        weights_only=False,
    )
    dual_source = torch.load(
        ROOT / DUAL_SOURCE,
        map_location="cpu",
        weights_only=False,
    )
    rnafm_splits = _selected_splits(records, _graph_index(rnafm_source))
    dual_splits = _selected_splits(records, _graph_index(dual_source))
    structures = _structure_index()
    structural_splits = {
        split: [_select_structure(record, structures) for record in records[split]]
        for split in STRUCTURE_SPLITS
    }

    common = {
        "clean_manifest": LARGE_MANIFEST,
        "manifest_sha256": manifest["manifest_sha256"],
        "source_asset": RNAFM_SOURCE,
    }
    outputs = {}
    outputs["graph"] = {
        **rnafm_source,
        **common,
        "graphs": rnafm_splits,
        "in_dim": 640,
    }
    outputs["dual"] = {
        **dual_source,
        "clean_manifest": LARGE_MANIFEST,
        "manifest_sha256": manifest["manifest_sha256"],
        "source_asset": DUAL_SOURCE,
        "graphs": dual_splits,
        "in_dim": 1920,
    }

    structure_variants = {
        "seq3d": {
            "relations": ["seq", "struct3d"],
            "features": "rnafm",
            "structural_ss": False,
        },
        "seqss3d": {
            "relations": ["seq", "ss", "struct3d"],
            "features": "rnafm",
            "structural_ss": False,
        },
        "seq3d_ernie": {
            "relations": ["seq", "struct3d"],
            "features": "ernie",
            "structural_ss": False,
        },
        "grassp_rnafm": {
            "relations": ["struct3d"],
            "features": "rnafm",
            "structural_ss": True,
        },
        "grassp_ernie": {
            "relations": ["struct3d"],
            "features": "ernie",
            "structural_ss": True,
        },
    }
    for tag, spec in structure_variants.items():
        split_graphs = {}
        for split in STRUCTURE_SPLITS:
            graphs = []
            for record, source_graph, structural in zip(
                records[split],
                rnafm_splits[split],
                structural_splits[split],
            ):
                edge_index, edge_type = _relation_edges(
                    source_graph,
                    list(rnafm_source["rel_names"]),
                    spec["relations"],
                    structural,
                )
                x = (
                    source_graph.x
                    if spec["features"] == "rnafm"
                    else structural.x
                )
                payload = {
                    "x": x,
                    "edge_index": edge_index,
                    "edge_type": edge_type,
                    "y": source_graph.y,
                    "rna_id": record.rna_id,
                    "num_nodes": x.size(0),
                    "num_relations": len(spec["relations"]),
                }
                if spec["structural_ss"]:
                    payload["ss_emb"] = structural.ss_emb
                graphs.append(Data(**payload))
            split_graphs[split] = graphs
        outputs[tag] = {
            **common,
            "graphs": split_graphs,
            "in_dim": 640 if spec["features"] == "rnafm" else 768,
            "rel_names": spec["relations"],
            "num_relations": len(spec["relations"]),
            "feature_source": spec["features"],
        }

    file_tags = {
        "graph": "graph",
        "dual": "dual",
        "seq3d": "graph_seq3d",
        "seqss3d": "graph_seqss3d",
        "seq3d_ernie": "graph_seq3d_ernie",
        "grassp_rnafm": "grassp_rnafm",
        "grassp_ernie": "grassp_ernie",
    }
    audit = {
        "manifest": LARGE_MANIFEST,
        "manifest_sha256": manifest["manifest_sha256"],
        "counts": {split: len(rnafm_splits[split]) for split in SPLIT_ORDER},
        "outputs": {},
    }
    for tag, payload in outputs.items():
        path = ROOT / "data" / "processed" / f"{file_tags[tag]}_{OUTPUT_PREFIX}.pt"
        _save_object(path, payload)
        audit["outputs"][tag] = {
            "path": str(path.relative_to(ROOT)),
            "in_dim": payload["in_dim"],
            "rel_names": payload.get("rel_names"),
            "counts": {
                split: len(payload["graphs"][split])
                for split in payload["graphs"]
            },
            "nodes": {
                split: int(sum(graph.x.size(0) for graph in payload["graphs"][split]))
                for split in payload["graphs"]
            },
            "edges": {
                split: int(sum(graph.edge_index.size(1) for graph in payload["graphs"][split]))
                for split in payload["graphs"]
            },
        }
        print(f"[written] {path.relative_to(ROOT)}", flush=True)

    relation_counts = {
        "seq3d": Counter(),
        "seqss3d": Counter(),
        "seq3d_ernie": Counter(),
    }
    for tag in relation_counts:
        for graph in outputs[tag]["graphs"]["pool"]:
            relation_counts[tag].update(
                int(value)
                for value in torch.bincount(
                    graph.edge_type,
                    minlength=graph.num_relations,
                ).tolist()
            )
    audit["pool_relation_edge_counts"] = {
        tag: dict(sorted(counts.items()))
        for tag, counts in relation_counts.items()
    }
    output_path = ROOT / "results" / "large_protocol" / "graphs_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
