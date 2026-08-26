from __future__ import annotations
import argparse, json, os
from pathlib import Path
import torch
from tqdm import tqdm
import sys
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0,os.path.dirname(ROOT))
from esm.models.esmc import ESMC

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--out-dir",required=True); ap.add_argument("--esm-model",default="esmc_600m"); ap.add_argument("--shard-size",type=int,default=512); ap.add_argument("--batch-size",type=int,default=4); ap.add_argument("--device",default="cuda:0"); a=ap.parse_args()
    rows=sorted([json.loads(x) for x in open(a.manifest,encoding="utf-8")],key=lambda r:(r["length"],r["accession"])); dev=torch.device(a.device if torch.cuda.is_available() else "cpu"); model=ESMC.from_pretrained(a.esm_model).to(dev).eval(); [p.requires_grad_(False) for p in model.parameters()]
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); index={}; dim=1152 if a.esm_model.endswith("600m") else 960
    for start in tqdm(range(0,len(rows),a.shard_size)):
        shard_rows=rows[start:start+a.shard_size]; embeddings=[]
        for bs in range(0,len(shard_rows),a.batch_size):
            batch=shard_rows[bs:bs+a.batch_size]; tokens=model._tokenize([r["sequence"] for r in batch])
            with torch.no_grad(): out_emb=model(tokens).embeddings[:,1:-1].to(dtype=torch.bfloat16).cpu(); out_mask=(tokens[:,1:-1] != model.tokenizer.pad_token_id).cpu()
            for j,r in enumerate(batch): embeddings.append(out_emb[j,:r["length"]].contiguous())
        shard=out/f"shard_{start//a.shard_size:05d}.pt"; torch.save({"accessions":[r["accession"] for r in shard_rows],"embeddings":embeddings},shard)
        for j,r in enumerate(shard_rows): index[r["accession"]]={"shard":shard.name,"row":j}
    json.dump({"esm_model":a.esm_model,"esm_dim":dim,"manifest":os.path.abspath(a.manifest),"order":"length_accession","index":index},open(out/"index.json","w"),indent=2); print(json.dumps({"completed":True,"records":len(rows),"shards":(len(rows)+a.shard_size-1)//a.shard_size,"order":"length_accession","out_dir":str(out)},indent=2))
if __name__=="__main__": main()
