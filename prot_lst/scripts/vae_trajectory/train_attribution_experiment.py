from __future__ import annotations
import argparse, json, math, os, random, sys
from collections import OrderedDict, deque
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(ROOT))
from prot_lst.protein_vae_contrastive import ProteinVAEContrastiveTrajectory
from prot_lst.scripts.vae_trajectory.feature_taxonomy import (
    DOMAIN_LABELS, DOMAIN_SOURCE_TYPES, RESIDUE_LABELS, RESIDUE_SOURCE_TYPES,
    map_feature_type,
)
from prot_lst.scripts.vae_trajectory.vae_objectives import free_bits_kl

ARMS = ("esm", "vae_z3", "z3_transformer", "trajectory_frozen", "joint_function", "joint_multitask")
STRUCT = set(RESIDUE_SOURCE_TYPES)
DOMAIN = set(DOMAIN_SOURCE_TYPES)
BRIDGE_PRETRAIN_PREFIXES = ("in_proj.", "residue.", "stage.", "encoder.", "norm.", "reconstruct.")


class ShardStore:
    def __init__(self, index_path: str, max_loaded: int = 2):
        self.path = Path(index_path).parent
        self.meta = json.load(open(index_path, encoding="utf-8"))
        self.loaded = OrderedDict()
        self.max_loaded = max_loaded

    def get(self, accession: str) -> torch.Tensor:
        loc = self.meta["index"][accession]
        name = loc["shard"]
        if name not in self.loaded:
            self.loaded[name] = torch.load(self.path / name, map_location="cpu")
            while len(self.loaded) > self.max_loaded:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(name)
        return self.loaded[name]["embeddings"][loc["row"]].float()


class AttentionProjection(nn.Module):
    def __init__(self, input_dim: int, text_dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 1))
        self.proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, text_dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        score = self.score(x).squeeze(-1).masked_fill(~mask, -1e4)
        pooled = (x * score.softmax(-1).unsqueeze(-1)).sum(1)
        return F.normalize(self.proj(pooled), dim=-1)


class ESMBaseline(nn.Module):
    def __init__(self, esm_dim: int, text_dim: int):
        super().__init__()
        self.function = AttentionProjection(esm_dim, text_dim)
        self.structure_head = nn.Sequential(nn.LayerNorm(esm_dim), nn.Linear(esm_dim, len(RESIDUE_LABELS)))
        self.domain_head = nn.Sequential(nn.LayerNorm(esm_dim), nn.Linear(esm_dim, len(DOMAIN_LABELS)))

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"text": self.function(embeddings, mask), "residue_hidden": embeddings}


class InterleavedBridge(nn.Module):
    """Fuse z0-z3 with stage-causal attention and decode H3 into z4."""

    def __init__(self, latent: int, model_dim: int, text_dim: int, esm_dim: int, stages: int = 4, max_len: int = 1024, layers: int = 2, heads: int = 8):
        super().__init__()
        self.stages = stages
        self.in_proj = nn.Sequential(nn.LayerNorm(latent), nn.Linear(latent, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
        self.residue = nn.Embedding(max_len, model_dim)
        self.stage = nn.Embedding(stages, model_dim)
        layer = nn.TransformerEncoderLayer(model_dim, heads, 4 * model_dim, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(model_dim)
        self.pool = AttentionProjection(model_dim, text_dim)
        self.structure_head = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, len(RESIDUE_LABELS)))
        self.domain_head = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, len(DOMAIN_LABELS)))
        self.reconstruct = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, esm_dim))

    @staticmethod
    def stage_causal_mask(length: int, stages: int, device: torch.device) -> torch.Tensor:
        """Return [L*T,L*T] mask; a query may only read its stage or earlier stages."""
        stage_ids = torch.arange(stages, device=device).repeat(length)
        return stage_ids[None, :] > stage_ids[:, None]

    def forward(self, trajectory: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        b, l, t, _ = trajectory.shape
        x = self.in_proj(trajectory)
        x = x + self.residue(torch.arange(l, device=x.device)[None, :, None]) + self.stage(torch.arange(t, device=x.device)[None, None, :])
        token_mask = mask[:, :, None].expand(b, l, t).reshape(b, l * t)
        causal_mask = self.stage_causal_mask(l, t, x.device)
        hidden = self.norm(self.encoder(x.reshape(b, l * t, -1), mask=causal_mask, src_key_padding_mask=~token_mask)).reshape(b, l, t, -1)
        hidden = hidden.masked_fill(~mask[:, :, None, None], 0.0)
        stages = [hidden[:, :, i] for i in range(t)]
        reconstruction = self.reconstruct(stages[-1]).masked_fill(~mask.unsqueeze(-1), 0.0)
        return {"hidden": hidden, "stages": stages, "text": self.pool(stages[-1], mask),
                "reconstruction": reconstruction, "z4": reconstruction}


def read_text_caches(paths: list[str]) -> dict:
    result = {}
    for path in paths:
        result.update(torch.load(path, map_location="cpu")["cache"])
    return result


def text_vector(cache: dict, accession: str) -> torch.Tensor:
    return cache[accession]["views"].float()[0]


def local_target(row: dict, length: int, device: torch.device, labels) -> torch.Tensor:
    label_to_index = {label: index for index, label in enumerate(labels)}
    level = "residue" if tuple(labels) == RESIDUE_LABELS else "domain"
    y = torch.zeros(length, len(labels), device=device)
    for feature in row.get("features", []):
        label = map_feature_type(feature.get("type", ""), level)
        if label not in label_to_index:
            continue
        start = max(1, int(feature.get("start", 1))) - 1
        end = min(length, int(feature.get("end", start + 1)))
        y[start:end, label_to_index[label]] = 1
    return y


def balanced_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                 pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    x, y = logits[mask], target[mask]
    if x.numel() == 0:
        return logits.sum() * 0.0
    if pos_weight is not None:
        return F.binary_cross_entropy_with_logits(x, y, pos_weight=pos_weight.to(x.device, x.dtype))
    positive = y.sum(0)
    active = positive > 0
    if not active.any():
        return x.new_zeros(())
    x, y, positive = x[:, active], y[:, active], positive[active]
    pos_weight = ((x.shape[0] - positive) / positive).clamp(1, 50)
    return F.binary_cross_entropy_with_logits(x, y, pos_weight=pos_weight)


def global_pos_weight(rows: list[dict], labels, level: str) -> torch.Tensor:
    """Compute fixed residue-level class weights without materializing dense targets."""
    positive = torch.zeros(len(labels), dtype=torch.float64)
    total = 0
    flag = "has_structure" if level == "residue" else "has_domain"
    label_to_index = {label: index for index, label in enumerate(labels)}
    for row in rows:
        if not row.get(flag):
            continue
        length = int(row["length"]); total += length; intervals = {label: [] for label in labels}
        for feature in row.get("features", []):
            label = map_feature_type(feature.get("type", ""), level)
            if label not in intervals:
                continue
            start = max(1, int(feature.get("start", 1))) - 1
            end = min(length, int(feature.get("end", start + 1)))
            if end > start: intervals[label].append((start, end))
        for label, spans in intervals.items():
            covered = 0; cursor_start = cursor_end = -1
            for start, end in sorted(spans):
                if start > cursor_end:
                    covered += max(0, cursor_end - cursor_start); cursor_start, cursor_end = start, end
                else:
                    cursor_end = max(cursor_end, end)
            covered += max(0, cursor_end - cursor_start)
            positive[label_to_index[label]] += covered
    if total == 0:
        raise RuntimeError(f"no supervised training positions for {level}")
    weight = ((total - positive) / positive.clamp_min(1)).clamp(1, 50)
    weight[positive == 0] = 1
    return weight.float()


def macro_auprc(target: torch.Tensor, score: torch.Tensor) -> float:
    y = target.numpy(); p = score.numpy(); values = []
    for index in range(y.shape[1]):
        if y[:, index].sum() > 0:
            values.append(average_precision_score(y[:, index], p[:, index]))
    return float(np.mean(values)) if values else 0.0


def multilabel_auprc(target: torch.Tensor, score: torch.Tensor, labels) -> dict:
    y = target.numpy(); p = score.numpy(); per_class = {}
    active_values = []
    for index, label in enumerate(labels):
        positives = int(y[:, index].sum())
        value = float(average_precision_score(y[:, index], p[:, index])) if positives else None
        per_class[label] = {"auprc": value, "positives": positives, "positions": int(y.shape[0])}
        if value is not None:
            active_values.append(value)
    micro = float(average_precision_score(y.reshape(-1), p.reshape(-1))) if y.sum() else 0.0
    return {"macro_auprc": float(np.mean(active_values)) if active_values else 0.0,
            "micro_auprc": micro, "per_class": per_class}


def retrieval_at_k(protein: torch.Tensor, text: torch.Tensor, hashes: list[str]) -> dict[str, float]:
    if not hashes:
        return {"top1": 0.0, "top5": 0.0, "top10": 0.0}
    similarity = F.normalize(protein, dim=-1) @ F.normalize(text, dim=-1).T
    ranked = similarity.topk(min(10, len(hashes)), dim=1).indices.cpu().tolist()
    return {f"top{k}": sum(any(hashes[index] == hashes[candidate] for candidate in candidates[:k])
                            for index, candidates in enumerate(ranked)) / len(hashes)
            for k in (1, 5, 10)}


def positive_cosine_alignment(protein: torch.Tensor, batch_text: torch.Tensor) -> torch.Tensor:
    """Align paired H3/text vectors without in-batch or queued negatives."""
    protein = F.normalize(protein.float(), dim=-1)
    batch_text = F.normalize(batch_text.float(), dim=-1)
    return 1.0 - (protein * batch_text).sum(dim=-1).mean()


def queued_vicreg_terms(output: torch.Tensor, queued: list[torch.Tensor], gamma: float,
                        covariance_dimensions: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate VICReg statistics with a detached history queue for tiny batches."""
    normalized = F.normalize(output.float(), dim=-1)
    current = normalized * math.sqrt(normalized.shape[-1])
    population = torch.cat([*queued, current], dim=0) if queued else current
    centered = population - population.mean(dim=0)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    variance = F.relu(gamma - std).mean()
    dimensions = population.shape[-1]
    sampled = min(covariance_dimensions, dimensions)
    if len(population) < 2 or sampled < 2:
        covariance = population.new_zeros(())
    else:
        indices = torch.linspace(0, dimensions - 1, sampled, device=population.device).round().long()
        selected = centered[:, indices]
        matrix = selected.T @ selected / (len(population) - 1)
        off_diagonal = matrix.square().sum() - matrix.diagonal().square().sum()
        covariance = off_diagonal / sampled
        if sampled < dimensions:
            covariance = covariance * (dimensions - 1) / (sampled - 1)
    return normalized, variance, covariance


def append_vicreg_queue(queue: list[torch.Tensor], values: torch.Tensor, limit: int) -> None:
    queue.append(values.detach())
    excess = sum(len(item) for item in queue) - limit
    while excess > 0 and queue:
        if excess >= len(queue[0]):
            excess -= len(queue.pop(0))
        else:
            queue[0] = queue[0][excess:]
            excess = 0


def early_stopping_update(score: float, best: float, stale_epochs: int,
                          min_delta: float) -> tuple[bool, float, int]:
    improved = score > best + min_delta
    return (True, score, 0) if improved else (False, best, stale_epochs + 1)


def bucket_batches(rows: list[dict], batch_size: int, bucket_size: int, seed: int) -> list[list[dict]]:
    # Form deterministic length buckets first, then shuffle bucket order. The
    # length-sorted ESM cache keeps each bucket in one or two adjacent shards.
    ordered = sorted(rows, key=lambda r: (r["length"], r["accession"]))
    buckets = []
    for start in range(0, len(ordered), bucket_size):
        bucket = ordered[start:start + bucket_size]
        buckets.append([bucket[i:i + batch_size] for i in range(0, len(bucket), batch_size)])
    random.Random(seed).shuffle(buckets)
    return [batch for bucket in buckets for batch in bucket]


def make_batch(rows, store, esm_dim, device):
    width = max(r["length"] for r in rows)
    embeddings = torch.zeros(len(rows), width, esm_dim, device=device)
    mask = torch.zeros(len(rows), width, dtype=torch.bool, device=device)
    for i, row in enumerate(rows):
        value = store.get(row["accession"])
        n = value.shape[0]
        embeddings[i, :n] = value.to(device)
        mask[i, :n] = True
    return embeddings, mask


def pairwise_corr(protein: torch.Tensor, text: torch.Tensor) -> float:
    if len(protein) < 3:
        return 0.0
    off = ~torch.eye(len(protein), device=protein.device, dtype=torch.bool)
    x = (protein @ protein.T)[off]; y = (text @ text.T)[off]
    x = x - x.mean(); y = y - y.mean()
    return float(((x * y).mean() / (x.std(unbiased=False) * y.std(unbiased=False)).clamp_min(1e-8)).cpu())


def encode(arm, model, vae, embeddings, mask, sample=None):
    if arm == "esm":
        return model(embeddings, mask)
    if sample is None:
        sample = vae.training and any(parameter.requires_grad for parameter in vae.parameters())
    vae_out = vae(embeddings, mask, sample=sample)
    trajectory = vae_out["states"]
    if arm == "vae_z3":
        return {"text": model(trajectory[:, :, 3], mask), "vae": vae_out, "trajectory": trajectory}
    if arm == "z3_transformer":
        trajectory = trajectory[:, :, 3:4]
    result = model(trajectory, mask)
    result.update({"vae": vae_out, "trajectory": trajectory})
    return result


@torch.no_grad()
def validate(arm, model, vae, rows, store, texts, esm_dim, device, batch_size, limit):
    model.eval()
    if vae is not None:
        vae.eval()
    subset = sorted(rows, key=lambda r: r["accession"])[:limit] if limit else sorted(rows, key=lambda r: r["accession"])
    projections, targets, function_hashes = [], [], []; structure_y=[]; structure_score=[]; domain_y=[]; domain_score=[]
    for start in range(0, len(subset), batch_size):
        batch = subset[start:start + batch_size]
        embeddings, mask = make_batch(batch, store, esm_dim, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            result=encode(arm, model, vae, embeddings, mask); projections.append(result["text"])
        targets.append(torch.stack([text_vector(texts, r["accession"]) for r in batch]).to(device))
        function_hashes.extend(row["function_hash"] for row in batch)
        if arm in {"esm", "joint_multitask"}:
            structure_hidden=result["residue_hidden"] if arm=="esm" else result["stages"][1]
            domain_hidden=result["residue_hidden"] if arm=="esm" else result["stages"][2]
            for row_index,row in enumerate(batch):
                length=row["length"]
                if row["has_structure"]:
                    structure_y.append(local_target(row,length,torch.device("cpu"),RESIDUE_LABELS)); structure_score.append(torch.sigmoid(model.structure_head(structure_hidden[row_index,:length])).float().cpu())
                if row["has_domain"]:
                    domain_y.append(local_target(row,length,torch.device("cpu"),DOMAIN_LABELS)); domain_score.append(torch.sigmoid(model.domain_head(domain_hidden[row_index,:length])).float().cpu())
    projection_tensor, target_tensor = torch.cat(projections), torch.cat(targets)
    function_metric = pairwise_corr(F.normalize(projection_tensor, dim=-1), F.normalize(target_tensor, dim=-1))
    retrieval = retrieval_at_k(projection_tensor, target_tensor, function_hashes)
    structure = multilabel_auprc(torch.cat(structure_y),torch.cat(structure_score),RESIDUE_LABELS) if structure_y else {"macro_auprc":0.0,"micro_auprc":0.0,"per_class":{}}
    domain = multilabel_auprc(torch.cat(domain_y),torch.cat(domain_score),DOMAIN_LABELS) if domain_y else {"macro_auprc":0.0,"micro_auprc":0.0,"per_class":{}}
    structure_metric = structure["macro_auprc"]
    domain_metric = domain["macro_auprc"]
    selection_score = float(np.mean([function_metric,structure_metric,domain_metric])) if arm in {"esm","joint_multitask"} else function_metric
    model.train()
    if vae is not None and any(p.requires_grad for p in vae.parameters()):
        vae.train()
    return {"function_pairwise_correlation":function_metric,"function_top1_retrieval":retrieval["top1"],
            "function_top5_retrieval":retrieval["top5"],"function_top10_retrieval":retrieval["top10"],
            "h1_macro_auprc":structure_metric,"h1_micro_auprc":structure["micro_auprc"],"h1_per_class":structure["per_class"],
            "h2_macro_auprc":domain_metric,"h2_micro_auprc":domain["micro_auprc"],"h2_per_class":domain["per_class"],
            "selection_score":selection_score,"records":len(subset)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--shard-index", required=True); ap.add_argument("--text-cache", action="append", required=True); ap.add_argument("--vae", required=True)
    ap.add_argument("--arm", choices=ARMS, required=True); ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--batch-size", type=int, default=4); ap.add_argument("--bucket-size", type=int, default=128); ap.add_argument("--grad-accumulation",type=int,default=1); ap.add_argument("--model-dim", type=int, default=None, help="default: use the pretrained bridge width, otherwise 512"); ap.add_argument("--layers", type=int, default=2); ap.add_argument("--heads", type=int, default=8); ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--vae-lr", type=float, default=5e-6); ap.add_argument("--seed", type=int, default=17); ap.add_argument("--validation-limit", type=int, default=512); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--no-amp", action="store_true", help="disable CUDA BF16 autocast"); ap.add_argument("--cosine-weight",type=float,default=0.01); ap.add_argument("--reconstruction-weight",type=float,default=0.1); ap.add_argument("--beta",type=float,default=0.001); ap.add_argument("--kl-free-bits",type=float,default=0.01); ap.add_argument("--kl-collapse-threshold",type=float,default=0.1); ap.add_argument("--kl-monitor-window",type=int,default=100); ap.add_argument("--log-every",type=int,default=20); ap.add_argument("--resume",default="",help="resume from an epoch checkpoint")
    ap.add_argument("--function-vicreg", action="store_true", help="add positive-only H3 variance/covariance regularization")
    ap.add_argument("--vicreg-gamma", type=float, default=0.5); ap.add_argument("--vicreg-variance-weight", type=float, default=0.25); ap.add_argument("--vicreg-covariance-weight", type=float, default=0.02)
    ap.add_argument("--vicreg-queue-size", type=int, default=256); ap.add_argument("--vicreg-covariance-dimensions", type=int, default=128)
    ap.add_argument("--early-stopping-patience", type=int, default=0, help="stop after this many stale validation epochs; 0 disables")
    ap.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    a = ap.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed); device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    if a.grad_accumulation < 1: raise ValueError("--grad-accumulation must be >= 1")
    if a.function_vicreg and (a.arm != "joint_multitask" or a.vicreg_queue_size < 2 or a.vicreg_covariance_dimensions < 2):
        raise ValueError("VICReg requires joint_multitask, queue size >= 2, and covariance dimensions >= 2")
    if a.early_stopping_patience < 0 or a.early_stopping_min_delta < 0:
        raise ValueError("early stopping patience and min delta must be non-negative")
    rows = [json.loads(x) for x in open(a.manifest, encoding="utf-8")]; train = sorted([r for r in rows if r["split"] == "train"], key=lambda r: r["accession"]); validation = [r for r in rows if r["split"] == "validation"]; train = train[:a.limit] if a.limit else train
    structure_pos_weight=global_pos_weight(train,RESIDUE_LABELS,"residue") if a.arm in {"esm","joint_multitask"} else None
    domain_pos_weight=global_pos_weight(train,DOMAIN_LABELS,"domain") if a.arm in {"esm","joint_multitask"} else None
    texts = read_text_caches(a.text_cache); store = ShardStore(a.shard_index); esm_dim = store.meta["esm_dim"]; vae_ck = torch.load(a.vae, map_location=device); latent = vae_ck.get("latent_dim", 256); text_dim = next(iter(texts.values()))["views"].shape[-1]; vae = None
    a.model_dim = a.model_dim or vae_ck.get("bridge_config", {}).get("model_dim", 512)
    if a.arm == "esm":
        model = ESMBaseline(esm_dim, text_dim).to(device)
    else:
        vae = ProteinVAEContrastiveTrajectory(esm_dim, latent).to(device); vae.load_state_dict(vae_ck["model"]); joint = a.arm.startswith("joint_")
        if not joint:
            for parameter in vae.parameters(): parameter.requires_grad_(False)
            vae.eval()
        model = AttentionProjection(latent, text_dim).to(device) if a.arm == "vae_z3" else InterleavedBridge(latent, a.model_dim, text_dim, esm_dim, 1 if a.arm == "z3_transformer" else 4, layers=a.layers, heads=a.heads).to(device)
        if a.arm in {"trajectory_frozen", "joint_function", "joint_multitask"} and vae_ck.get("bridge"):
            current = model.state_dict()
            compatible = {key: value for key, value in vae_ck["bridge"].items()
                          if key.startswith(BRIDGE_PRETRAIN_PREFIXES)
                          and key in current and current[key].shape == value.shape}
            model.load_state_dict(compatible, strict=False)
            print(json.dumps({"pretrained_bridge_tensors": len(compatible), "model_dim": a.model_dim}), flush=True)
    groups = [{"params": list(model.parameters()), "lr": a.lr}]
    if vae is not None and a.arm.startswith("joint_"): groups.append({"params": list(vae.parameters()), "lr": a.vae_lr})
    parameters = [parameter for group in groups for parameter in group["params"]]
    if not parameters: raise RuntimeError("optimizer has no trainable parameters")
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4); history=[]; best_metric=-float("inf"); stale_epochs=0; start_epoch=0; out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); verified_update=False; verified_heads=set(); kl_window=deque(maxlen=a.kl_monitor_window); vicreg_queue=[]
    if a.resume:
        resume = torch.load(a.resume, map_location=device)
        if resume.get("arm") != a.arm:
            raise ValueError(f"resume arm {resume.get('arm')} does not match {a.arm}")
        expected_objective = "positive_pair_cosine_plus_vicreg_no_negatives" if a.function_vicreg else "positive_pair_cosine_no_negatives"
        if resume.get("function_objective") != expected_objective:
            raise ValueError("resume checkpoint uses a different Function objective")
        model.load_state_dict(resume["model"])
        if vae is not None and resume.get("vae") is not None: vae.load_state_dict(resume["vae"])
        optimizer.load_state_dict(resume["optimizer"])
        history=list(resume.get("history",[])); best_metric=float(resume.get("best_selection_score",resume.get("validation",{}).get("selection_score",-float("inf")))); start_epoch=int(resume["epoch"])
        stale_epochs=int(resume.get("early_stopping_stale_epochs",0))
        vicreg_queue=[item.to(device) for item in resume.get("function_vicreg_queue",[])]
        if resume.get("torch_rng_state") is not None: torch.set_rng_state(resume["torch_rng_state"].cpu())
        if device.type=="cuda" and resume.get("cuda_rng_state") is not None: torch.cuda.set_rng_state_all([state.cpu() for state in resume["cuda_rng_state"]])
        print(json.dumps({"resumed_from":a.resume,"completed_epochs":start_epoch,"best_selection_score":best_metric}),flush=True)
    for epoch in range(start_epoch,a.epochs):
        batches=bucket_batches(train, a.batch_size, a.bucket_size, a.seed + epoch); optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(batches, 1):
            embeddings, mask = make_batch(batch, store, esm_dim, device); batch_text=torch.stack([text_vector(texts,r["accession"]) for r in batch]).to(device)
            amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda" and not a.no_amp)
            with amp_context:
                result=encode(a.arm,model,vae,embeddings,mask); function_loss=positive_cosine_alignment(result["text"],batch_text); variance_loss=function_loss.new_zeros(()); covariance_loss=function_loss.new_zeros(())
                if a.function_vicreg:
                    normalized_output,variance_loss,covariance_loss=queued_vicreg_terms(result["text"],vicreg_queue,a.vicreg_gamma,a.vicreg_covariance_dimensions)
                    loss=function_loss+a.vicreg_variance_weight*variance_loss+a.vicreg_covariance_weight*covariance_loss
                else: loss=function_loss
                structure_loss=loss.new_zeros(()); domain_loss=loss.new_zeros(()); reconstruction=loss.new_zeros(()); reconstruction_mse=loss.new_zeros(()); reconstruction_cosine=loss.new_zeros(()); raw_kl=loss.new_zeros(()); regularized_kl=loss.new_zeros(()); active_units=loss.new_zeros(())
            if a.arm in {"esm", "joint_multitask"}:
                structure=torch.stack([local_target(r,mask.shape[1],device,RESIDUE_LABELS) for r in batch]); domain=torch.stack([local_target(r,mask.shape[1],device,DOMAIN_LABELS) for r in batch]); structure_mask=mask & torch.tensor([r["has_structure"] for r in batch],device=device)[:,None]; domain_mask=mask & torch.tensor([r["has_domain"] for r in batch],device=device)[:,None]
                structure_hidden=result["residue_hidden"] if a.arm=="esm" else result["stages"][1]; domain_hidden=result["residue_hidden"] if a.arm=="esm" else result["stages"][2]
                with amp_context:
                    structure_loss=balanced_bce(model.structure_head(structure_hidden),structure,structure_mask,structure_pos_weight); domain_loss=balanced_bce(model.domain_head(domain_hidden),domain,domain_mask,domain_pos_weight); loss=loss+0.5*structure_loss+0.5*domain_loss
            if a.arm.startswith("joint_"):
                with amp_context:
                    reconstruction_mse=F.mse_loss(result["reconstruction"][mask],embeddings[mask]); reconstruction_cosine=F.cosine_similarity(result["reconstruction"][mask].float(),embeddings[mask].float(),dim=-1).mean(); reconstruction=reconstruction_mse+a.cosine_weight*(1-reconstruction_cosine); raw_kl,regularized_kl,active_units=free_bits_kl(result["vae"]["mean"],result["vae"]["logvar"],mask,a.kl_free_bits); loss=loss+a.reconstruction_weight*reconstruction+a.beta*regularized_kl
            if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss at epoch {epoch+1} step {step}")
            group_start=((step-1)//a.grad_accumulation)*a.grad_accumulation
            accumulation_size=min(a.grad_accumulation,len(batches)-group_start)
            (loss/accumulation_size).backward()
            if a.function_vicreg:
                append_vicreg_queue(vicreg_queue,normalized_output*math.sqrt(normalized_output.shape[-1]),a.vicreg_queue_size)
            if a.arm=="joint_multitask":
                required={"z4":model.reconstruct[-1].weight,"vae_mean":vae.to_mean.weight}
                if structure_mask.any(): required["h1"]=model.structure_head[-1].weight
                if domain_mask.any(): required["h2"]=model.domain_head[-1].weight
                missing=[name for name,parameter in required.items() if parameter.grad is None or not torch.count_nonzero(parameter.grad).item()]
                if missing: raise RuntimeError(f"backward pass has no gradient for active objectives: {missing}")
                verified_heads.update(required)
            update_now=step%a.grad_accumulation==0 or step==len(batches)
            if update_now:
                torch.nn.utils.clip_grad_norm_(parameters,1.0)
                if not verified_update:
                    changed_parameter = next((p for p in parameters if p.grad is not None and torch.count_nonzero(p.grad).item()), None)
                    if changed_parameter is None: raise RuntimeError("first accumulated backward pass produced no nonzero parameter gradients")
                    before_update = changed_parameter.detach().clone()
                optimizer.step()
                if not verified_update:
                    if torch.equal(before_update, changed_parameter.detach()): raise RuntimeError("optimizer step did not update a parameter with nonzero gradient")
                    verified_update=True
                optimizer.zero_grad(set_to_none=True)
            if a.arm.startswith("joint_"):
                kl_window.append(float(raw_kl.detach()))
                if len(kl_window)==a.kl_monitor_window and sum(kl_window)/len(kl_window)<a.kl_collapse_threshold:
                    raise RuntimeError(f"raw KL collapsed below {a.kl_collapse_threshold} over {a.kl_monitor_window} consecutive steps")
            history.append({"epoch":epoch+1,"step":step,"optimizer_step":update_now,"loss":float(loss.detach()),"function":float(function_loss.detach()),"function_positive_cosine":float(1.0-function_loss.detach()),"function_variance":float(variance_loss.detach()),"function_covariance":float(covariance_loss.detach()),"structure":float(structure_loss.detach()),"domain":float(domain_loss.detach()),"reconstruction":float(reconstruction.detach()),"reconstruction_mse":float(reconstruction_mse.detach()),"reconstruction_cosine":float(reconstruction_cosine.detach()),"kl":float(raw_kl.detach()),"kl_regularized":float(regularized_kl.detach()),"kl_active_units":float(active_units.detach())})
            if a.log_every and step % a.log_every == 0:
                print(json.dumps(history[-1]), flush=True)
        if a.arm=="joint_multitask" and not {"h1","h2","z4","vae_mean"}.issubset(verified_heads):
            raise RuntimeError(f"epoch {epoch+1} never activated gradients for: {sorted({'h1','h2','z4','vae_mean'}-verified_heads)}")
        metric=validate(a.arm,model,vae,validation,store,texts,esm_dim,device,a.batch_size,a.validation_limit); score=metric["selection_score"]; improved,best_metric,stale_epochs=early_stopping_update(score,best_metric,stale_epochs,a.early_stopping_min_delta); objective="positive_pair_cosine_plus_vicreg_no_negatives" if a.function_vicreg else "positive_pair_cosine_no_negatives"; state={"arm":a.arm,"model":model.state_dict(),"vae":vae.state_dict() if vae is not None else None,"optimizer":optimizer.state_dict(),"args":vars(a),"esm_dim":esm_dim,"latent_dim":latent,"text_dim":text_dim,"trajectory_states":4,"architecture":"vae_trajectory_stage_causal_bridge_z4","attention_mode":"stage_causal","reconstruction_state":"z4_from_h3_mse_plus_cosine","kl_strategy":"free_bits","function_objective":objective,"function_vicreg_queue":[item.detach().cpu() for item in vicreg_queue],"local_head_type":"multilabel","residue_labels":list(RESIDUE_LABELS),"domain_labels":list(DOMAIN_LABELS),"structure_pos_weight":structure_pos_weight,"domain_pos_weight":domain_pos_weight,"validation":metric,"validation_pairwise_correlation":metric["function_pairwise_correlation"],"selection_metric":"mean(function_pairwise_correlation,h1_macro_auprc,h2_macro_auprc)","best_selection_score":best_metric,"early_stopping_stale_epochs":stale_epochs,"epoch":epoch+1,"history":history,"torch_rng_state":torch.get_rng_state(),"cuda_rng_state":torch.cuda.get_rng_state_all() if device.type=="cuda" else None}; torch.save(state,str(out)+f".epoch{epoch+1}")
        if improved: torch.save(state,out)
        print(json.dumps({"epoch":epoch+1,"validation":metric,"best_selection_score":best_metric}),flush=True)
        if a.early_stopping_patience and stale_epochs >= a.early_stopping_patience:
            print(json.dumps({"early_stopped":True,"epoch":epoch+1,"stale_epochs":stale_epochs,"patience":a.early_stopping_patience,"best_selection_score":best_metric}),flush=True)
            break
    print(json.dumps({"completed":True,"arm":a.arm,"train_records":len(train),"validation_records":len(validation),"best_selection_score":best_metric,"checkpoint":str(out)},indent=2))
if __name__ == "__main__": main()
