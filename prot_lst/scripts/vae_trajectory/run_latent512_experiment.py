from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print(json.dumps({"launch": command}), flush=True)
    subprocess.run(command, check=True)


def latest(output: Path, maximum: int) -> Path | None:
    return next((Path(str(output) + f".epoch{epoch}") for epoch in range(maximum - 1, 0, -1)
                 if Path(str(output) + f".epoch{epoch}").exists()), None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the controlled latent-512 VAE trajectory experiment.")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    python = sys.executable
    vae = Path("runs/trajectory_vae_50k_2ep_latent512.pt")
    vae_epoch2 = Path(str(vae) + ".epoch2")
    if not vae_epoch2.exists():
        command = [python, "-m", "prot_lst.scripts.vae_trajectory.train_trajectory_vae",
                   "--data", "data/uniprot_reviewed.jsonl", "--esm-model", "esmc_300m",
                   "--limit", "50000", "--epochs", "2", "--batch-size", "1",
                   "--grad-accumulation", "2", "--latent-dim", "512", "--model-dim", "512",
                   "--layers", "2", "--heads", "8", "--device", args.device,
                   "--no-progress", "--out", str(vae)]
        resume = latest(vae, 2)
        if resume:
            command.extend(["--resume", str(resume)])
        run(command)

    joint = Path("runs/joint_multitask_latent512_4ep.pt")
    joint_epoch4 = Path(str(joint) + ".epoch4")
    if not joint_epoch4.exists():
        command = [python, "-m", "prot_lst.scripts.vae_trajectory.train_attribution_experiment",
                   "--manifest", "runs/stage2_55k_manifest.jsonl",
                   "--shard-index", "runs/stage2_55k_esm/index.json",
                   "--text-cache", "runs/stage2_55k_function_text.pt", "--vae", str(vae_epoch2),
                   "--arm", "joint_multitask", "--out", str(joint), "--epochs", "4",
                   "--batch-size", "1", "--grad-accumulation", "2", "--bucket-size", "128",
                   "--layers", "2", "--heads", "8",
                   "--lr", "5e-5", "--vae-lr", "5e-6", "--validation-limit", "5000",
                   "--device", args.device, "--log-every", "20"]
        resume = latest(joint, 4)
        if resume:
            command.extend(["--resume", str(resume)])
        run(command)

    result = Path("runs/joint_multitask_latent512_4ep.test.json")
    if not result.exists():
        run([python, "-m", "prot_lst.scripts.vae_trajectory.evaluate_attribution_experiment",
             "--checkpoint", str(joint), "--manifest", "runs/stage2_55k_manifest.jsonl",
             "--shard-index", "runs/stage2_55k_esm/index.json",
             "--text-cache", "runs/stage2_55k_function_text.pt", "--split", "test",
             "--limit", "0", "--batch-size", "1", "--bootstrap", "0",
             "--device", args.device, "--out", str(result)])
    print(json.dumps({"completed": True, "vae": str(vae_epoch2), "joint": str(joint),
                      "test": str(result)}), flush=True)


if __name__ == "__main__":
    main()
