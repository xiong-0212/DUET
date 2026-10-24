from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class StructureGraphTests(unittest.TestCase):
    def test_shuffle_preserves_degree_and_removes_self_loops(self) -> None:
        true_path = ROOT / "data" / "processed" / "graphs_graph_seq3d_homology_p80.pt"
        shuffled_path = (
            ROOT
            / "data"
            / "processed"
            / "graphs_graph_seq3d_shuffle_homology_p80.pt"
        )
        if not true_path.exists() or not shuffled_path.exists():
            self.skipTest("requires generated structure graph assets")
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
        true_graphs = {graph.rna_id: graph for graph in true_obj["graphs"]["train"]}
        shuffled_graphs = {
            graph.rna_id: graph for graph in shuffled_obj["graphs"]["train"]
        }
        self.assertEqual(set(true_graphs), set(shuffled_graphs))
        for rna_id, true_graph in true_graphs.items():
            shuffled = shuffled_graphs[rna_id]
            true_struct = true_graph.edge_index[:, true_graph.edge_type == 1]
            shuffled_struct = shuffled.edge_index[:, shuffled.edge_type == 1]
            self.assertEqual(true_struct.size(1), shuffled_struct.size(1))
            self.assertTrue(torch.equal(true_struct[0], shuffled_struct[0]))
            self.assertEqual(
                int((shuffled_struct[0] == shuffled_struct[1]).sum()),
                0,
            )


if __name__ == "__main__":
    unittest.main()
