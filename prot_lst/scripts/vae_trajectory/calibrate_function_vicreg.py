from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from prot_lst.scripts.vae_trajectory.evaluate_attribution_experiment import construct, function_metrics
from prot_lst.scripts.vae_trajectory.train_attribution_experiment import (
    ShardStore, encode, make_batch, read_text_caches, text_vector,
)


def pooled_h3(model, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    score = model.pool.score(hidden).squeeze(-1).masked_fill(~mask, -1e4)
    return (hidden * score.softmax(-1).unsqueeze(-1)).sum(1)


def vicreg_terms(output: torch.Tensor, target: torch.Tensor, gamma: float) -> tuple[torch.Tensor, ...]:
    normalized = F.normalize(output.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    invariance = 1.0 - (normalized * target).sum(-1).mean()
    scaled = normalized * math.sqrt(normalized.shape[-1])
    centered = scaled - scaled.mean(0)
    std = torch.sqrt(centered.var(0, unbiased=False) + 1e-4)
    variance = F.relu(gamma - std).mean()
    if len(scaled) < 2:
        covariance = scaled.new_zeros(())
    else:
        cov = centered.T @ centered / (len(scaled) - 1)
        covariance = (cov.square().sum() - cov.diagonal().square().sum()) / cov.shape[0]
    return invariance, variance, covariance, normalized


def collapse_metrics(output: torch.Tensor, target: torch.Tensor, seed: int = 91,
                     gamma: float = 0.7) -> dict[str, float]:
    output = F.normalize(output.float(), dim=-1); target = F.normalize(target.float(), dim=-1)
    generator = torch.Generator(device=output.device).manual_seed(seed)
    permutation = torch.randperm(len(output), generator=generator, device=output.device)
    scaled = output * math.sqrt(output.shape[-1]); centered = scaled - scaled.mean(0)
    covariance = centered.T @ centered / max(1, len(output) - 1)
    trace = covariance.diagonal().sum(); participation = trace.square() / covariance.square().sum().clamp_min(1e-8)
    return {"positive_cosine": float((output * target).sum(-1).mean()),
            "permuted_cosine": float((output * target[permutation]).sum(-1).mean()),
            "output_random_pair_cosine": float((output * output[permutation]).sum(-1).mean()),
            "mean_scaled_dimension_std": float(centered.std(0, unbiased=False).mean()),
            "fraction_dimensions_below_gamma": float((centered.std(0, unbiased=False) < gamma).float().mean()),
            "covariance_participation_rank": float(participation)}


@torch.no_grad()
def build_cache(checkpoint: str, manifest: str, shard_index: str, text_cache: str,
                device: torch.device, output: Path) -> dict:
    ck = torch.load(checkpoint, map_location=device); arm, model, vae = construct(ck, device)
    if arm != "joint_multitask": raise ValueError("VICReg calibration requires joint_multitask")
    rows = [json.loads(line) for line in open(manifest, encoding="utf-8")]
    rows.sort(key=lambda row: (row["length"], row["accession"]))
    store = ShardStore(shard_index); texts = read_text_caches([text_cache])
    grouped = {split: {"pooled": [], "text": [], "hashes": []}
               for split in ("train", "validation", "test")}
    for index, row in enumerate(rows, 1):
        embedding, mask = make_batch([row], store, ck["esm_dim"], device)
        result = encode(arm, model, vae, embedding, mask, sample=False)
        pooled = pooled_h3(model, result["stages"][3], mask)
        bucket = grouped[row["split"]]
        bucket["pooled"].append(pooled[0].cpu().half())
        bucket["text"].append(text_vector(texts, row["accession"]).cpu().half())
        bucket["hashes"].append(row["function_hash"])
        if index % 1000 == 0: print(json.dumps({"cached_h3": index, "total": len(rows)}), flush=True)
    payload = {"source_checkpoint": str(Path(checkpoint).resolve())}
    for split, values in grouped.items():
        payload[split] = {"pooled": torch.stack(values["pooled"]),
                          "text": torch.stack(values["text"]), "hashes": values["hashes"]}
    output.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, output)
    return payload


@torch.no_grad()
def project_all(head, values: torch.Tensor, device: torch.device, batch_size: int = 512) -> torch.Tensor:
    return torch.cat([F.normalize(head(values[start:start + batch_size].to(device).float()).float(), dim=-1).cpu()
                      for start in range(0, len(values), batch_size)])


def retrieval_report(output: torch.Tensor, target: torch.Tensor, hashes: list[str], seed: int,
                     gamma: float) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = output.to(device); target = F.normalize(target.float(), dim=-1).to(device)
    permutation = torch.randperm(len(output), generator=torch.Generator(device=device).manual_seed(seed), device=device)
    report = function_metrics(output, target, hashes, permutation)
    report["collapse"] = collapse_metrics(output, target, seed, gamma)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/joint_multitask_latent512_positive_cosine_4ep.pt")
    parser.add_argument("--manifest", default="runs/stage2_55k_manifest.jsonl")
    parser.add_argument("--shard-index", default="runs/stage2_55k_esm/index.json")
    parser.add_argument("--text-cache", default="runs/stage2_55k_function_text.pt")
    parser.add_argument("--feature-cache", default="runs/latent512_positive_h3_pooled.pt")
    parser.add_argument("--out", default="runs/joint_multitask_latent512_positive_vicreg.pt")
    parser.add_argument("--report", default="runs/joint_multitask_latent512_positive_vicreg.test.json")
    parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4); parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--variance-weight", type=float, default=1.0); parser.add_argument("--covariance-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=117); parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache_path = Path(args.feature_cache)
    cache = torch.load(cache_path, map_location="cpu") if cache_path.exists() else build_cache(
        args.checkpoint, args.manifest, args.shard_index, args.text_cache, device, cache_path)
    checkpoint = torch.load(args.checkpoint, map_location=device); _, model, _ = construct(checkpoint, device)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    head = model.pool.proj; head.requires_grad_(True); head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    train = cache["train"]; validation = cache["validation"]; best_loss = float("inf"); best_state = None; history = []
    for epoch in range(args.epochs):
        order = torch.randperm(len(train["pooled"]), generator=torch.Generator().manual_seed(args.seed + epoch))
        totals = torch.zeros(4); batches = 0
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            pooled = train["pooled"][index].to(device).float(); target = train["text"][index].to(device).float()
            output = head(pooled); invariance, variance, covariance, _ = vicreg_terms(output, target, args.gamma)
            loss = invariance + args.variance_weight * variance + args.covariance_weight * covariance
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            totals += torch.tensor([float(loss.detach()), float(invariance.detach()), float(variance.detach()), float(covariance.detach())]); batches += 1
        head.eval(); val_output = project_all(head, validation["pooled"], device)
        inv, var, cov, _ = vicreg_terms(val_output, validation["text"], args.gamma)
        val_loss = float(inv + args.variance_weight * var + args.covariance_weight * cov)
        metrics = collapse_metrics(val_output, validation["text"], args.seed, args.gamma)
        item = {"epoch": epoch + 1, "train_loss": float(totals[0] / batches),
                "train_alignment": float(totals[1] / batches), "train_variance": float(totals[2] / batches),
                "train_covariance": float(totals[3] / batches), "validation_loss": val_loss, **metrics}
        history.append(item); print(json.dumps(item), flush=True)
        if val_loss < best_loss:
            best_loss = val_loss; best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        head.train()
    head.load_state_dict(best_state); head.eval()
    model_state = model.state_dict(); checkpoint["model"] = {key: value.detach().cpu() for key, value in model_state.items()}
    checkpoint["function_objective"] = "positive_pair_cosine_plus_vicreg_no_negatives"
    checkpoint["function_head_calibration"] = {"frozen_trunk": True, "gamma": args.gamma,
        "variance_weight": args.variance_weight, "covariance_weight": args.covariance_weight,
        "epochs": args.epochs, "batch_size": args.batch_size, "history": history}
    checkpoint.pop("optimizer", None); Path(args.out).parent.mkdir(parents=True, exist_ok=True); torch.save(checkpoint, args.out)
    validation_output = project_all(head, validation["pooled"], device)
    test_output = project_all(head, cache["test"]["pooled"], device)
    report = {"validation": retrieval_report(validation_output, validation["text"], validation["hashes"], args.seed, args.gamma),
              "test": retrieval_report(test_output, cache["test"]["text"], cache["test"]["hashes"], args.seed, args.gamma),
              "configuration": checkpoint["function_head_calibration"], "checkpoint": str(Path(args.out).resolve())}
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"completed": True, "checkpoint": args.out, "report": args.report}, indent=2), flush=True)


if __name__ == "__main__": main()
