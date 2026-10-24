from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_manifest(name: str) -> dict:
    return json.loads(
        (ROOT / "data" / "clean" / "manifests" / f"{name}.json").read_text(
            encoding="utf-8"
        )
    )


class FollowupDataTests(unittest.TestCase):
    def test_size_curve_is_nested_and_cluster_aware(self) -> None:
        small = load_manifest("followup_size_80_cv5")
        medium = load_manifest("followup_size_160_cv5")
        self.assertEqual(small["counts"]["pool"], 80)
        self.assertEqual(medium["counts"]["pool"], 160)
        small_clusters = {cluster["cluster_id"] for cluster in small["clusters"]}
        medium_clusters = {cluster["cluster_id"] for cluster in medium["clusters"]}
        self.assertTrue(small_clusters.issubset(medium_clusters))
        for manifest in [small, medium]:
            for cluster in manifest["clusters"]:
                containing = [
                    fold["fold"]
                    for fold in manifest["folds"]
                    if set(cluster["rna_ids"])
                    & set(fold["validation_ids"])
                ]
                self.assertEqual(containing, [cluster["fold"]])

    def test_repeated_cv_assignments_are_balanced(self) -> None:
        old_range = max([558, 1253, 1063, 744, 853]) - min(
            [558, 1253, 1063, 744, 853]
        )
        for seed in [101, 102, 103]:
            manifest = load_manifest(f"followup_cvrep_{seed}_cv5")
            positives = manifest["audit"]["fold_positive_residues"]
            self.assertEqual(manifest["audit"]["fold_sizes"], [58] * 5)
            self.assertLess(max(positives) - min(positives), old_range / 4)

    def test_shuffled_graph_preserves_degrees_and_sequence_edges(self) -> None:
        true_path = ROOT / "data" / "processed" / "graph_seq3d_large_p80_cv5.pt"
        shuffled_path = (
            ROOT
            / "data"
            / "processed"
            / "graph_seq3d_shuffle_large_p80_cv5.pt"
        )
        if not true_path.exists() or not shuffled_path.exists():
            self.skipTest("requires generated large-protocol graph assets")
        true_obj = torch.load(
            true_path,
            map_location="cpu",
            weights_only=False,
        )
        shuffled_obj = torch.load(
            shuffled_path,
            map_location="cpu",
            weights_only=False,
        )
        for split in true_obj["graphs"]:
            true_graphs = {
                graph.rna_id: graph for graph in true_obj["graphs"][split]
            }
            for shuffled in shuffled_obj["graphs"][split]:
                true = true_graphs[shuffled.rna_id]
                true_struct = true.edge_index[:, true.edge_type == 1]
                shuffled_struct = shuffled.edge_index[:, shuffled.edge_type == 1]
                self.assertTrue(
                    torch.equal(
                        torch.bincount(true_struct[0]),
                        torch.bincount(shuffled_struct[0]),
                    )
                )
                self.assertEqual(
                    int((shuffled_struct[0] == shuffled_struct[1]).sum()),
                    0,
                )


if __name__ == "__main__":
    unittest.main()
