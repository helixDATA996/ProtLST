from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch


EXPECTED_SPLITS = {"train": 45_000, "validation": 5_000, "test": 5_000}


def run(command: list[str], env: dict[str, str]) -> None:
    print(json.dumps({"launch": command}, ensure_ascii=False), flush=True)
    subprocess.run(command, check=True, env=env)


def validate_text_cache(path: Path) -> None:
    payload = torch.load(path, map_location="cpu")
    cache = payload.get("cache", {})
    if len(cache) != sum(EXPECTED_SPLITS.values()):
        raise RuntimeError(f"text cache has {len(cache)} records, expected 55000")
    bad = [accession for accession, item in cache.items() if tuple(item["views"].shape[:1]) != (1,)]
    if bad:
        raise RuntimeError(f"Function-only text cache invariant failed for {bad[:3]}")


def validate_manifest(path: Path) -> None:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    counts = Counter(row["split"] for row in rows)
    if dict(counts) != EXPECTED_SPLITS:
        raise RuntimeError(f"manifest split counts are {dict(counts)}, expected {EXPECTED_SPLITS}")
    if len({row["accession"] for row in rows}) != len(rows):
        raise RuntimeError("manifest contains duplicate accessions")


def validate_esm_index(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("esm_model") != "esmc_300m" or payload.get("esm_dim") != 960:
        raise RuntimeError("ESM cache is not ESM-C 300M with 960-dimensional residues")
    if len(payload.get("index", {})) != sum(EXPECTED_SPLITS.values()):
        raise RuntimeError(f"ESM index has {len(payload.get('index', {}))} records, expected 55000")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and train the locked 55k stage-2 joint model.")
    parser.add_argument("--data", default="data/uniprot_reviewed.jsonl")
    parser.add_argument("--split-file", default="data/stage2_55k_splits.tsv")
    parser.add_argument("--text-model", required=True)
    parser.add_argument("--text-cache", default="runs/stage2_55k_function_text.pt")
    parser.add_argument("--manifest", default="runs/stage2_55k_manifest.jsonl")
    parser.add_argument("--esm-dir", default="runs/stage2_55k_esm")
    parser.add_argument("--vae", default="runs/trajectory_vae_50k_2ep_anticollapse.pt.epoch2")
    parser.add_argument("--out", default="runs/joint_multitask_4ep.pt")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    data = Path(args.data); split_file = Path(args.split_file); text_model = Path(args.text_model)
    text_cache = Path(args.text_cache); manifest = Path(args.manifest); esm_dir = Path(args.esm_dir)
    vae = Path(args.vae); out = Path(args.out)
    for required in (data, split_file, text_model, vae):
        if not required.exists():
            raise FileNotFoundError(required)
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    python = sys.executable

    if not text_cache.exists():
        run([python, "-m", "prot_lst.scripts.vae_trajectory.cache_text_embeddings",
             "--model", str(text_model), "--data", str(data), "--split-file", str(split_file),
             "--split", "all", "--batch-size", "8", "--max-length", "512",
             "--device", args.device, "--out", str(text_cache)], env)
    validate_text_cache(text_cache)
    print(json.dumps({"validated": "function_text_cache", "records": 55_000}), flush=True)

    if not manifest.exists():
        run([python, "-m", "prot_lst.scripts.vae_trajectory.build_attribution_manifest",
             "--data", str(data), "--split-file", str(split_file), "--text-cache", str(text_cache),
             "--max-length", "1024", "--out", str(manifest)], env)
    validate_manifest(manifest)
    print(json.dumps({"validated": "manifest", "splits": EXPECTED_SPLITS}), flush=True)

    esm_index = esm_dir / "index.json"
    if not esm_index.exists():
        run([python, "-m", "prot_lst.scripts.vae_trajectory.build_esm_shards",
             "--manifest", str(manifest), "--out-dir", str(esm_dir), "--esm-model", "esmc_300m",
             "--shard-size", "256", "--batch-size", "2", "--device", args.device], env)
    validate_esm_index(esm_index)
    print(json.dumps({"validated": "esm_cache", "records": 55_000, "esm_dim": 960}), flush=True)

    final_epoch = Path(str(out) + ".epoch4")
    if final_epoch.exists():
        print(json.dumps({"completed": True, "checkpoint": str(final_epoch), "skipped_training": True}), flush=True)
        return
    resume = next((Path(str(out) + f".epoch{epoch}") for epoch in range(3, 0, -1)
                   if Path(str(out) + f".epoch{epoch}").exists()), None)
    command = [python, "-m", "prot_lst.scripts.vae_trajectory.train_attribution_experiment",
               "--manifest", str(manifest), "--shard-index", str(esm_index), "--text-cache", str(text_cache),
               "--vae", str(vae), "--arm", "joint_multitask", "--out", str(out),
               "--epochs", "4", "--batch-size", "2", "--bucket-size", "128",
               "--layers", "2", "--heads", "8", "--lr", "5e-5", "--vae-lr", "5e-6",
               "--validation-limit", "5000", "--device", args.device, "--cosine-weight", "0.01",
               "--reconstruction-weight", "0.1", "--beta", "0.001", "--kl-free-bits", "0.01",
               "--kl-collapse-threshold", "0.1", "--kl-monitor-window", "100", "--log-every", "20"]
    if resume is not None:
        command.extend(["--resume", str(resume)])
    run(command, env)


if __name__ == "__main__":
    main()
