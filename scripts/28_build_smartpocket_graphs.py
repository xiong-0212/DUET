from __future__ import annotations

import io
import json
import math
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
from Bio import Align
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from safetensors.torch import load_file
from torch_geometric.data import Data
from transformers import BertConfig, BertModel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))


EXTERNAL = ROOT / "data" / "external"
SETS_PATH = EXTERNAL / "external_sets.json"
SS_PATH = EXTERNAL / "ss_external.tsv"
CIF_CACHE = EXTERNAL / "cif_cache.json"
OUTPUT = ROOT / "data" / "processed" / "smartpocket_external_graphs.pt"
AUDIT = ROOT / "results" / "large_protocol" / "smartpocket_graphs_audit.json"
ERNIE_DIR = Path(
    os.environ.get(
        "ERNIE_RNA_DIR",
        str(ROOT / "data" / "external" / "ernierna-ss"),
    )
)
ERNIE_DISPLAY = (
    "ERNIE_RNA_DIR"
    if os.environ.get("ERNIE_RNA_DIR")
    else "data/external/ernierna-ss"
)
STANDARD = {"A", "C", "G", "U"}
ALIGNER = Align.PairwiseAligner()
ALIGNER.mode = "global"
ALIGNER.match_score = 1.0
ALIGNER.mismatch_score = -1.0
ALIGNER.open_gap_score = -4.0
ALIGNER.extend_gap_score = -1.0


def pairs_from_dotbracket(value: str) -> list[tuple[int, int]]:
    stack = []
    pairs = []
    for index, char in enumerate(value):
        if char == "(":
            stack.append(index)
        elif char == ")" and stack:
            pairs.append((stack.pop(), index))
    return pairs


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return f"${path.name}"


def dotbracket_features(value: str) -> torch.Tensor:
    paired = np.zeros(len(value), dtype=np.float32)
    opening = np.asarray([char == "(" for char in value], dtype=np.float32)
    closing = np.asarray([char == ")" for char in value], dtype=np.float32)
    unpaired = np.asarray([char == "." for char in value], dtype=np.float32)
    for left, right in pairs_from_dotbracket(value):
        paired[left] = 1.0
        paired[right] = 1.0
    return torch.tensor(
        np.column_stack([paired, opening, closing, unpaired]),
        dtype=torch.float,
    )


def undirected(pairs: list[tuple[int, int]]) -> torch.Tensor:
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    directed = list(pairs) + [(right, left) for left, right in pairs]
    return torch.tensor(directed, dtype=torch.long).t().contiguous()


def sequence_edges(length: int) -> torch.Tensor:
    return undirected([(index, index + 1) for index in range(length - 1)])


def topk_structure_edges(coords: np.ndarray, k: int = 8) -> torch.Tensor:
    if len(coords) <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    differences = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt(np.square(differences).sum(axis=-1))
    np.fill_diagonal(distances, np.inf)
    k_eff = min(k, len(coords) - 1)
    edges = []
    for source in range(len(coords)):
        targets = np.argsort(distances[source])[:k_eff]
        edges.extend((source, int(target)) for target in targets)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def fetch_cif(pdb_id: str, cache: dict[str, str]) -> str:
    key = pdb_id.upper()
    if key in cache:
        return cache[key]
    url = f"https://files.rcsb.org/download/{key}.cif"
    with urllib.request.urlopen(url, timeout=90) as response:
        text = response.read().decode("utf-8", "replace")
    cache[key] = text
    return text


def parse_coordinates(cif_text: str, chain: str) -> tuple[str, np.ndarray]:
    data = MMCIF2Dict(io.StringIO(cif_text))
    required = (
        "_atom_site.auth_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_atom_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"CIF missing fields: {missing}")
    residues: dict[str, dict[str, Any]] = {}
    for auth_chain, seq_id, comp, atom, x, y, z in zip(
        data["_atom_site.auth_asym_id"],
        data["_atom_site.label_seq_id"],
        data["_atom_site.label_comp_id"],
        data["_atom_site.label_atom_id"],
        data["_atom_site.Cartn_x"],
        data["_atom_site.Cartn_y"],
        data["_atom_site.Cartn_z"],
    ):
        if auth_chain != chain or comp not in STANDARD:
            continue
        entry = residues.setdefault(
            str(seq_id),
            {"base": comp, "atoms": {}},
        )
        entry["atoms"][atom] = (float(x), float(y), float(z))
    ordered = sorted(residues, key=lambda value: int(value))
    sequence = "".join(residues[key]["base"] for key in ordered)
    coords = []
    for key in ordered:
        atoms = residues[key]["atoms"]
        atom = next(
            (name for name in ("C4'", "P", "C1'") if name in atoms),
            None,
        )
        if atom is None:
            atom = sorted(atoms)[0]
        coords.append(atoms[atom])
    return sequence, np.asarray(coords, dtype=np.float32)


def align_coordinates(
    source_sequence: str,
    source_coords: np.ndarray,
    target_sequence: str,
) -> tuple[np.ndarray, int, float]:
    alignment = ALIGNER.align(target_sequence, source_sequence)[0]
    mapped: list[np.ndarray | None] = [None] * len(target_sequence)
    matches = 0
    columns = 0
    for (target_start, target_end), (source_start, source_end) in zip(
        alignment.aligned[0],
        alignment.aligned[1],
    ):
        width = min(target_end - target_start, source_end - source_start)
        for offset in range(width):
            target_index = target_start + offset
            source_index = source_start + offset
            mapped[target_index] = source_coords[source_index]
            if target_sequence[target_index] == source_sequence[source_index]:
                matches += 1
        columns += width
    missing = [index for index, value in enumerate(mapped) if value is None]
    for index in missing:
        available = [
            (abs(index - other), other)
            for other, value in enumerate(mapped)
            if value is not None
        ]
        _, nearest = min(available)
        mapped[index] = mapped[nearest]
    return (
        np.stack(mapped).astype(np.float32),
        len(missing),
        matches / columns if columns else 0.0,
    )


def build_ernie() -> tuple[BertModel, dict[str, int]]:
    config = BertConfig(
        vocab_size=26,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=1026,
        pad_token_id=0,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        add_pooling_layer=False,
    )
    model = BertModel(config)
    raw = load_file(ERNIE_DIR / "model.safetensors")
    state = {}
    for key, value in raw.items():
        if not key.startswith("model."):
            continue
        new_key = key[len("model.") :].replace(".layer_norm.", ".LayerNorm.")
        state[new_key] = value
    model.load_state_dict(state, strict=False)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "position_embeddings" not in name or "weight" not in name:
                continue
            dimension = parameter.size(-1)
            positions = torch.arange(parameter.size(0), dtype=torch.float).unsqueeze(1)
            divisor = torch.exp(
                torch.arange(0, dimension, 2, dtype=torch.float)
                * (-math.log(10000.0) / dimension)
            )
            encoding = torch.zeros(parameter.size(0), dimension)
            encoding[:, 0::2] = torch.sin(positions * divisor)
            encoding[:, 1::2] = torch.cos(positions * divisor)
            parameter.copy_(encoding)
    vocab = {}
    for index, line in enumerate((ERNIE_DIR / "vocab.txt").read_text().splitlines()):
        vocab[line.strip()] = index
    return model.eval(), vocab


def main() -> int:
    payload = json.loads(SETS_PATH.read_text(encoding="utf-8"))
    sets = payload["splits"]
    folds = {}
    for line in SS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        chain_id, dotbracket = line.split("\t")[:2]
        folds[chain_id] = dotbracket
    rnafm_payload = torch.load(
        EXTERNAL / "emb_rnafm.pt",
        map_location="cpu",
        weights_only=False,
    )
    rnalmo_payload = torch.load(
        EXTERNAL / "emb_rinalmo.pt",
        map_location="cpu",
        weights_only=False,
    )
    rnafm = rnafm_payload.get("embeddings", rnafm_payload)
    rinalmo = rnalmo_payload.get("embeddings", rnalmo_payload)

    cache = (
        json.loads(CIF_CACHE.read_text(encoding="utf-8"))
        if CIF_CACHE.exists()
        else {}
    )
    ernie_model, vocab = build_ernie()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ernie_model = ernie_model.to(device)
    base_map = {
        "A": vocab["A"],
        "C": vocab["C"],
        "G": vocab["G"],
        "U": vocab["U"],
        "T": vocab["U"],
        "N": vocab.get("<unk>", 3),
    }

    records = []
    audit = {"sets": {}, "feature_definitions": {}}
    for set_name, set_records in sets.items():
        audit["sets"][set_name] = {}
        for chain_id, record in sorted(set_records.items()):
            sequence = str(record["sequence"]).upper()
            labels = torch.tensor(record["labels"], dtype=torch.float)
            dotbracket = folds.get(chain_id, "")
            if len(dotbracket) != len(sequence):
                raise ValueError(
                    f"{chain_id}: dot-bracket length {len(dotbracket)} "
                    f"!= sequence length {len(sequence)}"
                )
            pdb_id, _, chain = chain_id.partition("_")
            cif_text = fetch_cif(pdb_id, cache)
            cif_sequence, coords = parse_coordinates(cif_text, chain)
            coords, imputed, cif_identity = align_coordinates(
                cif_sequence,
                coords,
                sequence,
            )
            ss_emb = dotbracket_features(dotbracket)
            structural = topk_structure_edges(coords)
            sequence_edge = sequence_edges(len(sequence))
            pair_edge = undirected(pairs_from_dotbracket(dotbracket))
            with torch.no_grad():
                token_ids = torch.tensor(
                    [[base_map.get(char, vocab.get("<unk>", 3)) for char in sequence]],
                    dtype=torch.long,
                    device=device,
                )
                hidden = ernie_model(
                    input_ids=token_ids,
                    attention_mask=torch.ones_like(token_ids),
                ).last_hidden_state
                ernie = hidden.squeeze(0)[: len(sequence)].half().cpu()
            records.append(
                {
                    "set": set_name,
                    "rna_id": chain_id,
                    "y": labels,
                    "ss_emb": ss_emb,
                    "rnafm": rnafm[chain_id].float(),
                    "rinalmo": rinalmo[chain_id].float(),
                    "dual": torch.cat(
                        [rnafm[chain_id].float(), rinalmo[chain_id].float()],
                        dim=1,
                    ),
                    "ernie": ernie.float(),
                    "seq_edge": sequence_edge,
                    "ss_edge": pair_edge,
                    "struct_edge": structural,
                }
            )
            audit["sets"][set_name][chain_id] = {
                "length": len(sequence),
                "positive_residues": int(labels.sum()),
                "seq_edges": int(sequence_edge.size(1)),
                "ss_edges": int(pair_edge.size(1)),
                "struct_edges": int(structural.size(1)),
                "struct_mean_out_degree": (
                    float(structural.size(1) / len(sequence))
                    if sequence
                    else 0.0
                ),
                "cif_sequence_identity": cif_identity,
                "imputed_structure_coordinates": imputed,
            }

    CIF_CACHE.write_text(
        json.dumps(cache, sort_keys=True),
        encoding="utf-8",
    )

    def relation_graph(
        record: dict[str, Any],
        feature: str,
        relations: list[str],
        with_ss_emb: bool = False,
    ) -> Data:
        parts = []
        types = []
        for relation_index, relation in enumerate(relations):
            edge = (
                record["struct_edge"]
                if relation == "struct3d"
                else record["seq_edge"]
                if relation == "seq"
                else record["ss_edge"]
            )
            parts.append(edge)
            types.append(
                torch.full(
                    (edge.size(1),),
                    relation_index,
                    dtype=torch.long,
                )
            )
        graph = Data(
            x=record[feature],
            edge_index=torch.cat(parts, dim=1),
            edge_type=torch.cat(types),
            y=record["y"],
            rna_id=record["rna_id"],
            external_set=record["set"],
            num_nodes=len(record["y"]),
            num_relations=len(relations),
        )
        if with_ss_emb:
            graph.ss_emb = record["ss_emb"]
        return graph

    variants = {
        "graph_rnafm": ("rnafm", ["seq", "ss"], False),
        "dual_rinalmo": ("dual", ["seq", "ss"], False),
        "seq3d_rnafm": ("rnafm", ["seq", "struct3d"], False),
        "seqss3d_rnafm": ("rnafm", ["seq", "ss", "struct3d"], False),
        "seq3d_ernie": ("ernie", ["seq", "struct3d"], False),
        "grassp_rnafm": ("rnafm", ["struct3d"], True),
        "grassp_ernie": ("ernie", ["struct3d"], True),
    }
    output: dict[str, Any] = {
        "source": display_path(SETS_PATH),
        "feature_definitions": {
            "rnafm": "RNA-FM layer-12 embeddings copied from the prior external audit",
            "rinalmo": "RiNALMo-giga embeddings copied from the prior external audit",
            "dual": "RNA-FM 640-d followed by RiNALMo-giga 1280-d",
            "ernie": f"ERNIE-RNA 12-layer 768-d embeddings from {ERNIE_DISPLAY}",
            "struct3d": "directed per-residue top-8 Euclidean neighbours using C4', then P, then C1'",
            "ss_emb": "[paired, opening, closing, unpaired] from the released dot-bracket",
        },
        "graphs": {},
    }
    for name, (feature, relations, with_ss_emb) in variants.items():
        output["graphs"][name] = [
            relation_graph(record, feature, relations, with_ss_emb)
            for record in records
        ]
    torch.save(output, OUTPUT)
    audit["output"] = display_path(OUTPUT)
    audit["n_chains"] = len(records)
    audit["feature_definitions"] = output["feature_definitions"]
    AUDIT.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "chains": len(records),
                "sets": {
                    name: len(value)
                    for name, value in audit["sets"].items()
                },
                "output": str(OUTPUT.relative_to(ROOT)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
