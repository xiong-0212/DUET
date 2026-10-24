from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.io import load_manifest_records, project_path


SOURCES = {
    "graph": "data/processed/graphs_seq_ss_clean_source.pt",
    "dual": "data/processed/graphs_dual_clean_source.pt",
}
SPLITS = ["TE18", "TL12", "JL10", "RB9", "val", "train"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifests",
        nargs="+",
        default=["exact_sequence_unique", "homology_p80"],
    )
    args = parser.parse_args()

    source_objects = {
        key: torch.load(project_path(path), map_location="cpu", weights_only=False)
        for key, path in SOURCES.items()
    }

    for manifest_name in args.manifests:
        raw_splits, manifest = load_manifest_records(manifest_name)
        for source_name, source in source_objects.items():
            output = dict(source)
            output_graphs = {}
            for split in SPLITS:
                by_id = {graph.rna_id: graph for graph in source["graphs"][split]}
                selected = []
                for record in raw_splits[split]:
                    graph = by_id.get(record.rna_id)
                    if graph is None:
                        raise KeyError(
                            f"{manifest_name}/{source_name}: missing graph {split}/{record.rna_id}"
                        )
                    if graph.x.size(0) != len(record.sequence):
                        raise ValueError(
                            f"{record.rna_id}: graph length {graph.x.size(0)} "
                            f"!= sequence length {len(record.sequence)}"
                        )
                    expected = torch.tensor(
                        [int(value) for value in record.labels],
                        dtype=graph.y.dtype,
                    )
                    if not torch.equal(graph.y.cpu(), expected):
                        raise ValueError(f"{record.rna_id}: graph labels do not match manifest")
                    selected.append(graph)
                output_graphs[split] = selected

            output["graphs"] = output_graphs
            output["clean_manifest"] = manifest_name
            output["manifest_sha256"] = manifest["manifest_sha256"]
            output["source_asset"] = SOURCES[source_name]
            out_path = project_path(
                f"data/processed/graphs_{source_name}_{manifest_name}.pt"
            )
            torch.save(output, out_path)
            print(
                f"[written] {out_path.relative_to(project_path('.'))} "
                + " ".join(f"{split}={len(output_graphs[split])}" for split in SPLITS),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
