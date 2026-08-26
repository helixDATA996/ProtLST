from __future__ import annotations
import argparse, json, os, random, sys
from collections import OrderedDict, deque
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(ROOT))
from prot_lst.protein_vae_contrastive import ProteinVAEContrastiveTrajectory

ARMS = ("esm", "vae_z3", "z3_transformer", "trajectory_frozen", "joint_function", "joint_multitask")
STRUCT = {"helix", "strand", "turn", "disulfide bond", "glycosylation site", "active site", "binding site", "modified residue", "short sequence motif"}
DOMAIN = {"domain", "region of interest", "repeat", "zinc finger region", "dna-binding region", "coiled-coil region", "transmembrane region", "topological domain", "signal peptide", "transit peptide"}


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
        self.structure_head = nn.Sequential(nn.LayerNorm(esm_dim), nn.Linear(esm_dim, 1))
        self.domain_head = nn.Sequential(nn.LayerNorm(esm_dim), nn.Linear(esm_dim, 1))

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"text": self.function(embeddings, mask), "residue_hidden": embeddings}


class InterleavedBridge(nn.Module):
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
        self.structure_head = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, 1))
        self.domain_head = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, 1))
        self.reconstruct = nn.Sequential(nn.LayerNorm(latent), nn.Linear(latent, esm_dim))

    def forward(self, trajectory: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        b, l, t, _ = trajectory.shape
        x = self.in_proj(trajectory)
        x = x + self.residue(torch.arange(l, device=x.device)[None, :, None]) + self.stage(torch.arange(t, device=x.device)[None, None, :])
        token_mask = mask[:, :, None].expand(b, l, t).reshape(b, l * t)
        hidden = self.norm(self.encoder(x.reshape(b, l * t, -1), src_key_padding_mask=~token_mask)).reshape(b, l, t, -1)
        stages = [hidden[:, :, i] for i in range(t)]
        return {"hidden": hidden, "stages": stages, "text": self.pool(stages[-1], mask), "reconstruction": self.reconstruct(trajectory[:, :, -1])}


def read_text_caches(paths: list[str]) -> dict:
    result = {}
    for path in paths:
        result.update(torch.load(path, map_location="cpu")["cache"])
    return result


def text_vector(cache: dict, accession: str) -> torch.Tensor:
    return cache[accession]["views"].float()[0]


def local_target(row: dict, length: int, device: torch.device, allowed: set[str]) -> torch.Tensor:
    y = torch.zeros(length, device=device)
    for feature in row.get("features", []):
        if feature.get("type", "").lower() not in allowed:
            continue
        start = max(1, int(feature.get("start", 1))) - 1
        end = min(length, int(feature.get("end", start + 1)))
        y[start:end] = 1
    return y


def balanced_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x, y = logits[mask], target[mask]
    positive = y.sum()
    if positive < 1:
        return x.new_zeros(())
    return F.binary_cross_entropy_with_logits(x, y, pos_weight=((y.numel() - positive) / positive).clamp(1, 50))


def contrastive(protein: torch.Tensor, batch_text: torch.Tensor, batch_hash: list[str], queue_text: torch.Tensor, queue_hash: list[str], temperature: float = .07) -> torch.Tensor:
    candidates = torch.cat([batch_text, queue_text], 0) if len(queue_text) else batch_text
    hashes = batch_hash + queue_hash
    logits = F.normalize(protein, dim=-1) @ F.normalize(candidates, dim=-1).T / temperature
    positives = torch.tensor([[h == q for q in hashes] for h in batch_hash], device=protein.device)
    log_prob = logits - logits.logsumexp(1, keepdim=True)
    return -(log_prob.masked_fill(~positives, 0).sum(1) / positives.sum(1).clamp_min(1)).mean()


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


def encode(arm, model, vae, embeddings, mask):
    if arm == "esm":
        return model(embeddings, mask)
    vae_out = vae(embeddings, mask, sample=False)
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
    projections, targets = [], []
    for start in range(0, len(subset), batch_size):
        batch = subset[start:start + batch_size]
        embeddings, mask = make_batch(batch, store, esm_dim, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            projections.append(encode(arm, model, vae, embeddings, mask)["text"])
        targets.append(torch.stack([text_vector(texts, r["accession"]) for r in batch]).to(device))
    metric = pairwise_corr(F.normalize(torch.cat(projections), dim=-1), F.normalize(torch.cat(targets), dim=-1))
    model.train()
    if vae is not None and any(p.requires_grad for p in vae.parameters()):
        vae.train()
    return metric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--shard-index", required=True); ap.add_argument("--text-cache", action="append", required=True); ap.add_argument("--vae", required=True)
    ap.add_argument("--arm", choices=ARMS, required=True); ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--batch-size", type=int, default=4); ap.add_argument("--bucket-size", type=int, default=128); ap.add_argument("--queue-size", type=int, default=1024); ap.add_argument("--model-dim", type=int, default=512); ap.add_argument("--layers", type=int, default=2); ap.add_argument("--heads", type=int, default=8); ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--vae-lr", type=float, default=5e-6); ap.add_argument("--seed", type=int, default=17); ap.add_argument("--validation-limit", type=int, default=512); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--no-amp", action="store_true", help="disable CUDA BF16 autocast")
    a = ap.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed); device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    rows = [json.loads(x) for x in open(a.manifest, encoding="utf-8")]; train = sorted([r for r in rows if r["split"] == "train"], key=lambda r: r["accession"]); validation = [r for r in rows if r["split"] == "validation"]; train = train[:a.limit] if a.limit else train
    texts = read_text_caches(a.text_cache); store = ShardStore(a.shard_index); esm_dim = store.meta["esm_dim"]; vae_ck = torch.load(a.vae, map_location=device); latent = vae_ck.get("latent_dim", 256); text_dim = next(iter(texts.values()))["views"].shape[-1]; vae = None
    if a.arm == "esm":
        model = ESMBaseline(esm_dim, text_dim).to(device)
    else:
        vae = ProteinVAEContrastiveTrajectory(esm_dim, latent).to(device); vae.load_state_dict(vae_ck["model"]); joint = a.arm.startswith("joint_")
        if not joint:
            for parameter in vae.parameters(): parameter.requires_grad_(False)
            vae.eval()
        model = AttentionProjection(latent, text_dim).to(device) if a.arm == "vae_z3" else InterleavedBridge(latent, a.model_dim, text_dim, esm_dim, 1 if a.arm == "z3_transformer" else 4, layers=a.layers, heads=a.heads).to(device)
    groups = [{"params": list(model.parameters()), "lr": a.lr}]
    if vae is not None and a.arm.startswith("joint_"): groups.append({"params": list(vae.parameters()), "lr": a.vae_lr})
    parameters = [parameter for group in groups for parameter in group["params"]]
    if not parameters: raise RuntimeError("optimizer has no trainable parameters")
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4); queue = deque(maxlen=a.queue_size); history=[]; best_metric=-float("inf"); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); verified_update=False
    for epoch in range(a.epochs):
        for step, batch in enumerate(bucket_batches(train, a.batch_size, a.bucket_size, a.seed + epoch), 1):
            embeddings, mask = make_batch(batch, store, esm_dim, device); batch_text=torch.stack([text_vector(texts,r["accession"]) for r in batch]).to(device); batch_hash=[r["function_hash"] for r in batch]; queue_text=torch.stack([x[0] for x in queue]).to(device) if queue else batch_text.new_zeros((0,text_dim)); queue_hash=[x[1] for x in queue]
            amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda" and not a.no_amp)
            with amp_context:
                result=encode(a.arm,model,vae,embeddings,mask); function_loss=contrastive(result["text"],batch_text,batch_hash,queue_text,queue_hash); loss=function_loss; structure_loss=loss.new_zeros(()); domain_loss=loss.new_zeros(()); reconstruction=loss.new_zeros(())
            if a.arm in {"esm", "joint_multitask"}:
                structure=torch.stack([local_target(r,mask.shape[1],device,STRUCT) for r in batch]); domain=torch.stack([local_target(r,mask.shape[1],device,DOMAIN) for r in batch]); structure_mask=mask & torch.tensor([r["has_structure"] for r in batch],device=device)[:,None]; domain_mask=mask & torch.tensor([r["has_domain"] for r in batch],device=device)[:,None]
                structure_hidden=result["residue_hidden"] if a.arm=="esm" else result["stages"][1]; domain_hidden=result["residue_hidden"] if a.arm=="esm" else result["stages"][2]
                with amp_context:
                    structure_loss=balanced_bce(model.structure_head(structure_hidden).squeeze(-1),structure,structure_mask); domain_loss=balanced_bce(model.domain_head(domain_hidden).squeeze(-1),domain,domain_mask); loss=loss+0.5*structure_loss+0.5*domain_loss
            if a.arm.startswith("joint_"):
                with amp_context:
                    reconstruction=F.mse_loss(result["reconstruction"][mask],embeddings[mask]); loss=loss+0.1*reconstruction+0.001*result["vae"]["kl"]
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(parameters,1.0)
            if not verified_update:
                changed_parameter = next((p for p in parameters if p.grad is not None and torch.count_nonzero(p.grad).item()), None)
                if changed_parameter is None: raise RuntimeError("first backward pass produced no nonzero parameter gradients")
                before_update = changed_parameter.detach().clone()
            optimizer.step()
            if not verified_update:
                if torch.equal(before_update, changed_parameter.detach()): raise RuntimeError("optimizer step did not update a parameter with nonzero gradient")
                verified_update=True
            for vector,function_hash in zip(batch_text.detach().cpu(),batch_hash): queue.append((vector,function_hash))
            history.append({"epoch":epoch+1,"step":step,"loss":float(loss.detach()),"function":float(function_loss.detach()),"structure":float(structure_loss.detach()),"domain":float(domain_loss.detach()),"reconstruction":float(reconstruction.detach())})
        metric=validate(a.arm,model,vae,validation,store,texts,esm_dim,device,a.batch_size,a.validation_limit); state={"arm":a.arm,"model":model.state_dict(),"vae":vae.state_dict() if vae is not None else None,"args":vars(a),"esm_dim":esm_dim,"latent_dim":latent,"text_dim":text_dim,"validation_pairwise_correlation":metric,"epoch":epoch+1,"history":history}; torch.save(state,str(out)+f".epoch{epoch+1}")
        if metric>best_metric: best_metric=metric; torch.save(state,out)
        print(json.dumps({"epoch":epoch+1,"validation_pairwise_correlation":metric,"best":best_metric}),flush=True)
    print(json.dumps({"completed":True,"arm":a.arm,"train_records":len(train),"validation_records":len(validation),"best_validation_pairwise_correlation":best_metric,"checkpoint":str(out)},indent=2))
if __name__ == "__main__": main()
