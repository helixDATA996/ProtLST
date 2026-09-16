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


def latest(output: Path) -> Path | None:
    return next((Path(str(output) + f".epoch{epoch}") for epoch in range(3, 0, -1)
                 if Path(str(output) + f".epoch{epoch}").exists()), None)


def train(args, arm: str, output: Path) -> None:
    if Path(str(output) + ".epoch4").exists():
        print(json.dumps({"skip_completed": str(output)}), flush=True)
        return
    command = [sys.executable, "-m", "prot_lst.scripts.vae_trajectory.train_attribution_experiment",
               "--manifest", "runs/stage2_55k_manifest.jsonl",
               "--shard-index", "runs/stage2_55k_esm/index.json",
               "--text-cache", "runs/stage2_55k_function_text.pt",
               "--vae", args.vae, "--arm", arm, "--out", str(output),
               "--epochs", "4", "--batch-size", "1", "--grad-accumulation", "2",
               "--bucket-size", "128", "--layers", "2", "--heads", "8",
               "--lr", "5e-5", "--vae-lr", "5e-6", "--validation-limit", "5000",
               "--device", args.device, "--log-every", "20"]
    resume = latest(output)
    if resume:
        command.extend(["--resume", str(resume)])
    run(command)


def evaluate(args, checkpoint: Path, output: Path) -> None:
    if output.exists():
        return
    run([sys.executable, "-m", "prot_lst.scripts.vae_trajectory.evaluate_attribution_experiment",
         "--checkpoint", str(checkpoint), "--manifest", "runs/stage2_55k_manifest.jsonl",
         "--shard-index", "runs/stage2_55k_esm/index.json",
         "--text-cache", "runs/stage2_55k_function_text.pt", "--split", "test",
         "--limit", "0", "--batch-size", "1", "--bootstrap", "0",
         "--device", args.device, "--out", str(output)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare positive-only Function alignment without negatives.")
    parser.add_argument("--vae", default="runs/trajectory_vae_50k_2ep_latent512.pt.epoch2")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    for required in (args.vae, "runs/stage2_55k_manifest.jsonl", "runs/stage2_55k_esm/index.json",
                     "runs/stage2_55k_function_text.pt"):
        if not Path(required).exists():
            raise FileNotFoundError(required)
    baseline = Path("runs/esm_positive_cosine_4ep.pt")
    trajectory = Path("runs/joint_multitask_latent512_positive_cosine_4ep.pt")
    train(args, "esm", baseline)
    train(args, "joint_multitask", trajectory)
    evaluate(args, baseline, Path("runs/esm_positive_cosine_4ep.test.json"))
    evaluate(args, trajectory, Path("runs/joint_multitask_latent512_positive_cosine_4ep.test.json"))
    print(json.dumps({"completed": True, "baseline": str(baseline),
                      "trajectory": str(trajectory)}), flush=True)


if __name__ == "__main__":
    main()
