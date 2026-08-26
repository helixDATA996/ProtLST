#!/usr/bin/env python3
"""Create deterministic residue-level z0/z1/z2/z3 trajectories from sequences."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from prot_lst.protein_vae_contrastive import ProteinVAEContrastiveTrajectory
from esm.models.esmc import ESMC


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records = []
    name = None
    sequence = []
    for raw in path.open(encoding="utf-8"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(sequence)))
            name = line[1:].split()[0] or f"sequence_{len(records) + 1}"
            sequence = []
        else:
            sequence.append(line)
    if name is not None:
        records.append((name, "".join(sequence)))
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--sequence")
    source.add_argument("--fasta", type=Path)
    ap.add_argument("--name", default="protein")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    records = [(args.name, args.sequence)] if args.sequence else read_fasta(args.fasta)
    if not records:
        raise RuntimeError("no sequences found")
    if any(not sequence for _, sequence in records):
        raise ValueError("empty sequences are not supported")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    esm_name = checkpoint.get("esm_model", "esmc_600m")
    esm_dim = 1152 if esm_name.endswith("600m") else 960
    latent_dim = checkpoint.get("latent_dim", 256)
    esm = ESMC.from_pretrained(esm_name).to(device).eval()
    model = ProteinVAEContrastiveTrajectory(esm_dim, latent_dim).to(device).eval()
    model.load_state_dict(checkpoint["model"])

    output = []
    with torch.no_grad():
        for name, sequence in records:
            tokens = esm._tokenize([sequence])
            mask = tokens[:, 1:-1] != esm.tokenizer.pad_token_id
            embeddings = esm(tokens).embeddings[:, 1:-1].float()
            states = model(embeddings, mask, sample=False)["states"][0, :len(sequence)].cpu()
            output.append({"name": name, "sequence": sequence, "trajectory": states})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"records": output, "shape_contract": "[L,4,256]", "checkpoint": str(args.checkpoint)}, args.output)
    print(json.dumps({"completed": True, "records": len(output), "output": str(args.output),
                      "shapes": {item["name"]: list(item["trajectory"].shape) for item in output}}, indent=2))


if __name__ == "__main__":
    main()
