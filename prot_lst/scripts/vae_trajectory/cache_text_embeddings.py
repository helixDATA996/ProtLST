from __future__ import annotations

import argparse, json, os, sys
import torch
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0,ROOT)
from text_embedding_models import FrozenTextEncoder


def views(row: dict) -> list[str]:
    function=" ".join(row.get("function_text", "").replace("PubMed", "").split())
    go="; ".join(row.get("go_terms", [])); ec="; ".join(x for x in row.get("ec_terms", []) if x)
    return [f"Function: {function}", f"GO: {go}", f"EC: {ec}"]

def view_mask(row: dict) -> list[float]:
    return [float(bool(row.get("function_text", "").strip())), float(bool(row.get("go_terms"))), float(bool([x for x in row.get("ec_terms", []) if x]))]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--data",required=True); ap.add_argument("--split-file",default=""); ap.add_argument("--split",default="train"); ap.add_argument("--limit",type=int,default=0); ap.add_argument("--batch-size",type=int,default=8); ap.add_argument("--max-length",type=int,default=2048); ap.add_argument("--device",default="cuda"); ap.add_argument("--out",required=True); ap.add_argument("--causal",action="store_true")
    a=ap.parse_args(); dev=torch.device(a.device if torch.cuda.is_available() else "cpu"); enc=FrozenTextEncoder(a.model,dev,a.max_length,a.causal)
    allowed=None
    if a.split_file:
        allowed={line.rstrip("\n").split("\t")[0] for line in open(a.split_file,encoding="utf-8").readlines()[1:] if line.rstrip("\n").split("\t")[2]==a.split}
    cache={}; batch=[]
    for line in open(a.data,encoding="utf-8"):
        row=json.loads(line)
        if allowed is not None and row.get("accession") not in allowed: continue
        if not row.get("go_terms") and not row.get("function_text"): continue
        batch.append(row)
        if len(batch)<a.batch_size: continue
        texts=[views(x) for x in batch]
        flat=[y for x in texts for y in x]; emb=enc.encode(flat).reshape(len(batch),3,-1).cpu().half()
        for i,x in enumerate(batch): cache[x["accession"]]={"views":emb[i],"view_mask":view_mask(x),"labels":x.get("go_terms",[])}
        batch=[]; print(json.dumps({"cached":len(cache)}), flush=True)
        if a.limit and len(cache)>=a.limit: break
    if batch:
        flat=[y for x in [views(x) for x in batch] for y in x]; emb=enc.encode(flat).reshape(len(batch),3,-1).cpu().half()
        for i,x in enumerate(batch): cache[x["accession"]]={"views":emb[i],"view_mask":view_mask(x),"labels":x.get("go_terms",[])}
    if not cache: raise RuntimeError("no records were cached")
    dim=next(iter(cache.values()))["views"].shape[-1]
    os.makedirs(os.path.dirname(a.out),exist_ok=True); torch.save({"cache":cache,"model":os.path.abspath(a.model),"dim":dim},a.out)
    print(json.dumps({"completed":True,"records":len(cache),"dim":dim,"out":a.out},indent=2))

if __name__=="__main__": main()
