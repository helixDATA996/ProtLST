from __future__ import annotations

import argparse, json, os, random, sys
from pathlib import Path
import torch
from torch.nn import functional as F
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(ROOT))
from prot_lst.protein_vae_contrastive import ProteinVAEContrastiveTrajectory
from prot_lst.scripts.vae_trajectory.train_attribution_experiment import BRIDGE_PRETRAIN_PREFIXES, InterleavedBridge
from prot_lst.scripts.vae_trajectory.split_io import read_split_file
from prot_lst.scripts.vae_trajectory.vae_objectives import anti_collapse_schedule, free_bits_kl


def rows(path, limit=0, split_file="", split="train"):
    allowed=None
    if split_file:
        split_map=read_split_file(split_file); allowed={accession for accession,name in split_map.items() if name==split}
    out=[]
    for line in open(path, encoding="utf-8"):
        x=json.loads(line)
        if (allowed is None or x.get("accession") in allowed) and 50 <= len(x.get("sequence", "")) <= 1024:
            out.append(x)
            if limit and len(out)>=limit: break
    return out


@torch.no_grad()
def esm_encode(esm, seqs):
    tok=esm._tokenize(seqs); out=esm(tok)
    return out.embeddings[:,1:-1].float(), tok[:,1:-1] != esm.tokenizer.pad_token_id


def trajectory_loss(vae_out, bridge_out, embeddings, mask, beta,
                    cosine_weight: float = 0.01, free_bits: float = 0.01):
    reconstruction = F.mse_loss(bridge_out["z4"][mask], embeddings[mask])
    cosine = F.cosine_similarity(bridge_out["z4"][mask].float(), embeddings[mask].float(), dim=-1).mean()
    cosine_loss = 1.0 - cosine
    raw_kl, regularized_kl, active_units = free_bits_kl(
        vae_out["mean"], vae_out["logvar"], mask, free_bits,
    )
    total = reconstruction + cosine_weight * cosine_loss + beta * regularized_kl
    posterior_std = torch.exp(0.5 * vae_out["logvar"])[mask].mean()
    return total, {"embedding_reconstruction": reconstruction, "embedding_cosine": cosine,
                   "cosine_loss": cosine_loss, "kl": raw_kl,
                   "kl_regularized": regularized_kl, "kl_active_units": active_units,
                   "posterior_mean_abs": vae_out["mean"][mask].abs().mean(),
                   "posterior_std": posterior_std}


def shuffled_epoch_batches(size: int, batch_size: int, seed: int, max_batches: int = 0) -> list[list[int]]:
    """Return one shuffled, without-replacement traversal of dataset indices."""
    if size < 0 or batch_size <= 0 or max_batches < 0:
        raise ValueError("size and max_batches must be non-negative and batch_size must be positive")
    order = list(range(size))
    random.Random(seed).shuffle(order)
    batches = [order[start:start + batch_size] for start in range(0, size, batch_size)]
    return batches[:max_batches] if max_batches else batches


def main():
    # Keep ESM optional at module-import time so objective/batching utilities
    # remain testable on machines that only consume precomputed embeddings.
    from esm.models.esmc import ESMC

    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default="data/uniprot_sprot_2024_01.jsonl")
    ap.add_argument("--split-file", default=""); ap.add_argument("--split", default="train", choices=["train","validation","test"])
    ap.add_argument("--esm-model", default="esmc_600m", choices=["esmc_300m","esmc_600m"])
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=0, help="optional total-step cap for smoke/debug runs; 0 traverses every epoch")
    ap.add_argument("--no-progress", action="store_true", help="disable tqdm progress bars")
    ap.add_argument("--batch-size", type=int, default=2); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accumulation", type=int, default=1)
    ap.add_argument("--cosine-weight", type=float, default=0.01)
    ap.add_argument("--beta-max", type=float, default=0.001)
    ap.add_argument("--kl-free-bits", type=float, default=0.01, help="free nats per latent dimension")
    ap.add_argument("--deterministic-warmup-steps", type=int, default=2000)
    ap.add_argument("--noise-ramp-steps", type=int, default=8000)
    ap.add_argument("--latent-dim", type=int, default=256); ap.add_argument("--model-dim", type=int, default=256); ap.add_argument("--layers", type=int, default=2); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="runs/trajectory_vae_1k.pt"); ap.add_argument("--resume", default="")
    args=ap.parse_args(); random.seed(17); torch.manual_seed(17)
    if args.grad_accumulation < 1: raise ValueError("--grad-accumulation must be >= 1")
    device=torch.device(args.device if torch.cuda.is_available() else "cpu"); data=rows(args.data,args.limit,args.split_file,args.split)
    if len(data)<max(8,args.batch_size): raise RuntimeError(f"too few records: {len(data)}")
    esm=ESMC.from_pretrained(args.esm_model).to(device).eval()
    for p in esm.parameters(): p.requires_grad_(False)
    dim=1152 if args.esm_model.endswith("600m") else 960
    model=ProteinVAEContrastiveTrajectory(dim,args.latent_dim).to(device)
    bridge=InterleavedBridge(args.latent_dim,args.model_dim,256,dim,stages=4,layers=args.layers,heads=args.heads).to(device)
    vae_parameters=[parameter for name,parameter in model.named_parameters() if not name.startswith("summary.")]
    bridge_parameters=[parameter for name,parameter in bridge.named_parameters() if name.startswith(BRIDGE_PRETRAIN_PREFIXES)]
    parameters=[*vae_parameters,*bridge_parameters]
    opt=torch.optim.AdamW(parameters,lr=args.lr,weight_decay=1e-4); hist=[]; model.train(); bridge.train(); global_step=0; start_epoch=0
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    if args.resume:
        checkpoint=torch.load(args.resume,map_location=device)
        if checkpoint.get("latent_dim") != args.latent_dim or checkpoint.get("bridge_config",{}).get("model_dim") != args.model_dim:
            raise ValueError("resume checkpoint dimensions do not match requested latent/bridge dimensions")
        model.load_state_dict(checkpoint["model"]); bridge.load_state_dict(checkpoint["bridge"],strict=False)
        if checkpoint.get("optimizer") is None: raise ValueError("resume checkpoint does not contain optimizer state")
        opt.load_state_dict(checkpoint["optimizer"]); hist=list(checkpoint.get("history",[])); global_step=len(hist); start_epoch=int(checkpoint["completed_epochs"])
        if checkpoint.get("torch_rng_state") is not None: torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type=="cuda" and checkpoint.get("cuda_rng_state") is not None: torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])
        print(json.dumps({"resumed_from":args.resume,"completed_epochs":start_epoch,"steps":global_step}),flush=True)
    for epoch in range(start_epoch,args.epochs):
        remaining = max(0, args.steps - global_step) if args.steps else 0
        if args.steps and remaining == 0: break
        epoch_batches = shuffled_epoch_batches(len(data), args.batch_size, 17 + epoch, remaining)
        progress = tqdm(epoch_batches, total=len(epoch_batches), desc=f"Epoch {epoch + 1}/{args.epochs}",
                        unit="batch", dynamic_ncols=True,
                        disable=args.no_progress or not sys.stderr.isatty())
        opt.zero_grad(set_to_none=True)
        for batch_index, indices in enumerate(progress, 1):
            batch=[data[index] for index in indices]; global_step+=1
            seq=[x["sequence"] for x in batch]; emb,mask=esm_encode(esm,seq)
            schedule_step=(global_step-1)//args.grad_accumulation+1
            noise_scale,beta=anti_collapse_schedule(schedule_step,args.deterministic_warmup_steps,args.noise_ramp_steps,args.beta_max)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
                vae_out=model(emb,mask,sample=True,noise_scale=noise_scale); bridge_out=bridge(vae_out["states"],mask)
                total, m=trajectory_loss(vae_out,bridge_out,emb,mask,beta,args.cosine_weight,args.kl_free_bits)
            group_start=((batch_index-1)//args.grad_accumulation)*args.grad_accumulation
            accumulation_size=min(args.grad_accumulation,len(epoch_batches)-group_start)
            (total/accumulation_size).backward()
            update_now=batch_index%args.grad_accumulation==0 or batch_index==len(epoch_batches)
            if update_now:
                torch.nn.utils.clip_grad_norm_(parameters,1.0); opt.step(); opt.zero_grad(set_to_none=True)
            item={"epoch":epoch+1,"step":global_step,"optimizer_step":update_now,"loss":float(total.detach()),"beta":beta,"noise_scale":noise_scale,**{k:float(v.detach()) for k,v in m.items()}}; hist.append(item)
            progress.set_postfix(loss=f"{item['loss']:.4g}", recon=f"{item['embedding_reconstruction']:.4g}",
                                 cosine=f"{item['embedding_cosine']:.3f}", kl=f"{item['kl']:.4g}",
                                 active=f"{item['kl_active_units']:.0f}", noise=f"{noise_scale:.2f}",
                                 lr=f"{opt.param_groups[0]['lr']:.2e}")
            if global_step==1 or global_step%20==0: print(json.dumps(item),flush=True)
        bridge_state={name:value for name,value in bridge.state_dict().items() if name.startswith(BRIDGE_PRETRAIN_PREFIXES)}
        state={"model":model.state_dict(),"bridge":bridge_state,"optimizer":opt.state_dict(),"bridge_config":{"model_dim":args.model_dim,"layers":args.layers,"heads":args.heads},"esm_model":args.esm_model,"latent_dim":args.latent_dim,"trajectory_states":4,"architecture":"vae_trajectory_stage_causal_bridge_z4","reconstruction_target":"esm_embedding_mse_plus_cosine","trajectory_training_source":"annealed_posterior_sample","trajectory_evaluation_source":"posterior_mean","kl_strategy":"deterministic_warmup_noise_annealing_free_bits","sampling":"shuffle_once_per_epoch_without_replacement","completed_epochs":epoch+1,"data":os.path.abspath(args.data),"split_file":os.path.abspath(args.split_file) if args.split_file else "","split":args.split,"history":hist,"args":vars(args),"torch_rng_state":torch.get_rng_state(),"cuda_rng_state":torch.cuda.get_rng_state_all() if device.type=="cuda" else None}
        torch.save(state,args.out); torch.save(state,f"{args.out}.epoch{epoch+1}")
        print(json.dumps({"epoch_completed":epoch+1,"steps":global_step,"checkpoint":args.out}),flush=True)
        if args.steps and global_step >= args.steps: break
    print(json.dumps({"completed":True,"records":len(data),"epochs":state["completed_epochs"],"steps":global_step,"checkpoint":args.out,"final":hist[-1]},indent=2),flush=True)

if __name__=="__main__": main()
