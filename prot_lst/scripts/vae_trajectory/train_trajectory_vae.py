from __future__ import annotations

import argparse, json, os, random, sys
from pathlib import Path
import torch
from torch.nn import functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(ROOT))
from esm.models.esmc import ESMC
from prot_lst.protein_vae import ProteinVAE, encode_sequences
from prot_lst.protein_vae_contrastive import ProteinVAEContrastiveTrajectory, masked_pool, supervised_contrastive


def rows(path, limit=0, split_file="", split="train"):
    allowed=None
    if split_file:
        allowed={line.rstrip("\n").split("\t")[0] for line in open(split_file,encoding="utf-8").readlines()[1:] if line.rstrip("\n").split("\t")[2]==split}
    out=[]
    for line in open(path, encoding="utf-8"):
        x=json.loads(line)
        if (allowed is None or x.get("accession") in allowed) and 50 <= len(x.get("sequence", "")) <= 1024 and (x.get("go_terms") or x.get("function_text")):
            x["labels"] = set(x.get("go_terms", [])) | {"EC:" + e for e in x.get("ec_terms", []) if e}
            out.append(x)
            if limit and len(out)>=limit: break
    return out


@torch.no_grad()
def esm_encode(esm, seqs):
    tok=esm._tokenize(seqs); out=esm(tok)
    return out.embeddings[:,1:-1].float(), tok[:,1:-1] != esm.tokenizer.pad_token_id


def trajectory_loss(out, tokens, mask, labels, beta, contrastive_weight):
    weights=(1.0, 0.8, 0.6, 0.4)
    rec=sum(w*F.cross_entropy(out["state_logits"][:,:,i,:][mask], tokens[mask]) for i,w in enumerate(weights))/sum(weights)
    kl=out["kl"]
    pooled=masked_pool(out["states"], mask)
    con=supervised_contrastive(pooled, labels)
    total=rec + beta*kl + contrastive_weight*con
    return total, {"reconstruction":rec, "kl":kl, "contrastive":con,
                   "accuracy":(out["logits"].argmax(-1)[mask] == tokens[mask]).float().mean()}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default="data/uniprot_sprot_2024_01.jsonl")
    ap.add_argument("--split-file", default=""); ap.add_argument("--split", default="train", choices=["train","validation","test"])
    ap.add_argument("--esm-model", default="esmc_600m", choices=["esmc_300m","esmc_600m"])
    ap.add_argument("--limit", type=int, default=1000); ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=2); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--contrastive-weight", type=float, default=0.05); ap.add_argument("--temperature", type=float, default=.1)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="runs/trajectory_vae_1k.pt")
    args=ap.parse_args(); random.seed(17); torch.manual_seed(17)
    device=torch.device(args.device if torch.cuda.is_available() else "cpu"); data=rows(args.data,args.limit,args.split_file,args.split)
    if len(data)<8: raise RuntimeError(f"too few records: {len(data)}")
    esm=ESMC.from_pretrained(args.esm_model).to(device).eval()
    for p in esm.parameters(): p.requires_grad_(False)
    dim=1152 if args.esm_model.endswith("600m") else 960
    model=ProteinVAEContrastiveTrajectory(dim,256).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4); hist=[]; model.train()
    for step in range(args.steps):
        anchor=random.choice(data)
        shared=[x for x in data if x["labels"] & anchor["labels"]]
        batch=[anchor]
        if len(shared)>1:
            batch += random.sample(shared[1:], min(len(shared)-1, args.batch_size-1))
        if len(batch)<args.batch_size:
            pool=[x for x in data if x is not anchor and x not in batch]
            batch += random.sample(pool, min(len(pool), args.batch_size-len(batch)))
        seq=[x["sequence"] for x in batch]; emb,esm_mask=esm_encode(esm,seq); tokens,mask=encode_sequences(seq,device); mask &= esm_mask
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
            out=model(emb,mask,sample=True); beta=min(1e-2,1e-2*(step+1)/200)
            total, m=trajectory_loss(out,tokens,mask,[x["labels"] for x in batch],beta,args.contrastive_weight)
        opt.zero_grad(set_to_none=True); total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        item={"step":step+1,"loss":float(total.detach()),"beta":beta,**{k:float(v.detach()) for k,v in m.items()}}; hist.append(item)
        if step==0 or (step+1)%20==0: print(json.dumps(item),flush=True)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model":model.state_dict(),"esm_model":args.esm_model,"latent_dim":256,"trajectory_states":4,"data":os.path.abspath(args.data),"split_file":os.path.abspath(args.split_file) if args.split_file else "","split":args.split,"history":hist,"args":vars(args)},args.out)
    print(json.dumps({"completed":True,"records":len(data),"steps":args.steps,"checkpoint":args.out,"final":hist[-1]},indent=2),flush=True)

if __name__=="__main__": main()
