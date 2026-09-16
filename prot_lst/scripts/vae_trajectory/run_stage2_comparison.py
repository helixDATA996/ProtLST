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


def latest_epoch(output: Path) -> Path | None:
    return next((Path(str(output) + f".epoch{epoch}") for epoch in range(3, 0, -1)
                 if Path(str(output) + f".epoch{epoch}").exists()), None)


def train_arm(args, arm: str, output: Path) -> None:
    if Path(str(output) + ".epoch4").exists():
        print(json.dumps({"skip_completed": str(output), "arm": arm}), flush=True)
        return
    command = [sys.executable, "-m", "prot_lst.scripts.vae_trajectory.train_attribution_experiment",
               "--manifest", args.manifest, "--shard-index", args.shard_index,
               "--text-cache", args.text_cache, "--vae", args.vae, "--arm", arm,
               "--out", str(output), "--epochs", "4", "--batch-size", "2",
               "--bucket-size", "128", "--grad-accumulation", "8",
               "--layers", "2", "--heads", "8", "--lr", "5e-5", "--vae-lr", "5e-6",
               "--validation-limit", "5000", "--device", args.device, "--log-every", "20"]
    resume = latest_epoch(output)
    if resume:
        command.extend(["--resume", str(resume)])
    run(command)


def evaluate(args, checkpoint: Path, output: Path) -> None:
    if output.exists():
        print(json.dumps({"skip_completed_test": str(output)}), flush=True)
        return
    run([sys.executable, "-m", "prot_lst.scripts.vae_trajectory.evaluate_attribution_experiment",
         "--checkpoint", str(checkpoint), "--manifest", args.manifest,
         "--shard-index", args.shard_index, "--text-cache", args.text_cache,
         "--split", "test", "--limit", "0", "--batch-size", "2", "--bootstrap", "0",
         "--device", args.device, "--out", str(output)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train matched ESM and accumulated ProtLST comparisons.")
    parser.add_argument("--manifest", default="runs/stage2_55k_manifest.jsonl")
    parser.add_argument("--shard-index", default="runs/stage2_55k_esm/index.json")
    parser.add_argument("--text-cache", default="runs/stage2_55k_function_text.pt")
    parser.add_argument("--vae", default="runs/trajectory_vae_50k_2ep_anticollapse.pt.epoch2")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for path in (args.manifest, args.shard_index, args.text_cache, args.vae):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})

    baseline = Path("runs/esm_baseline_accum8_4ep.pt")
    improved = Path("runs/joint_multitask_accum8_4ep.pt")
    train_arm(args, "esm", baseline)
    train_arm(args, "joint_multitask", improved)
    evaluate(args, baseline, Path("runs/esm_baseline_accum8_4ep.test.json"))
    evaluate(args, improved, Path("runs/joint_multitask_accum8_4ep.test.json"))
    print(json.dumps({"completed": True, "baseline": str(baseline), "improved": str(improved)}), flush=True)


if __name__ == "__main__":
    main()
