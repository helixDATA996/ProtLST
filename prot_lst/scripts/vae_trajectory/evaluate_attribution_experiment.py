from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F

ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0,os.path.dirname(ROOT))
from prot_lst.protein_vae_contrastive import ProteinVAEContrastiveTrajectory
from prot_lst.scripts.vae_trajectory.train_attribution_experiment import AttentionProjection, DOMAIN_LABELS, ESMBaseline, InterleavedBridge, RESIDUE_LABELS, ShardStore, read_text_caches, text_vector, local_target, make_batch, encode


def function_metrics(protein,text,hashes,permutation,source_ids=None):
    protein=F.normalize(protein,dim=-1); text=F.normalize(text,dim=-1); sim=protein@text.T; order=sim.argsort(1,descending=True); positive=torch.tensor([[a==b for b in hashes] for a in hashes],device=protein.device); group_rank=positive.gather(1,order).float().argmax(1); exact=(order==torch.arange(len(protein),device=protein.device)[:,None]).nonzero()[:,1]; off=~torch.eye(len(protein),device=protein.device,dtype=torch.bool)
    if source_ids is not None: off &= source_ids[:,None] != source_ids[None,:]
    x=(protein@protein.T)[off]; y=(text@text.T)[off]; x=x-x.mean(); y=y-y.mean(); corr=(x*y).mean()/(x.std(unbiased=False)*y.std(unbiased=False)).clamp_min(1e-8); pos=(protein*text).sum(-1); perm=(protein*text[permutation]).sum(-1)
    return {"exact_r1":float((exact<1).float().mean()),"exact_r5":float((exact<5).float().mean()),"exact_r10":float((exact<10).float().mean()),"group_r1":float((group_rank<1).float().mean()),"group_r5":float((group_rank<5).float().mean()),"group_r10":float((group_rank<10).float().mean()),"group_median_rank":float(group_rank.float().median()),"positive_cosine":float(pos.mean()),"permuted_cosine":float(perm.mean()),"cosine_gap":float((pos-perm).mean()),"pairwise_similarity_correlation":float(corr)}


def bootstrap(protein,text,hashes,repeats,seed):
    rng=np.random.default_rng(seed); n=len(protein); permutation=torch.tensor(rng.permutation(n),device=protein.device); point=function_metrics(protein,text,hashes,permutation); samples={k:[] for k in point}
    for _ in range(repeats):
        idx=torch.tensor(rng.integers(0,n,n),device=protein.device); sampled_hash=[hashes[int(i)] for i in idx.cpu()]; perm=torch.tensor(rng.permutation(n),device=protein.device); item=function_metrics(protein[idx],text[idx],sampled_hash,perm,idx)
        for key,value in item.items():samples[key].append(value)
    return {key:{"value":point[key],"ci95":[float(np.quantile(samples[key],.025)),float(np.quantile(samples[key],.975))]} for key in point}


def paired_bootstrap(projections, text, hashes, repeats, seed):
    """Return point metrics plus paired mode-vs-base bootstrap differences."""
    rng = np.random.default_rng(seed)
    names = list(projections)
    point_permutation = torch.tensor(rng.permutation(len(text)), device=text.device)
    points = {name: function_metrics(projections[name], text, hashes, point_permutation) for name in names}
    base = "base"
    if repeats <= 0:
        result = {name: {metric: {"value": value, "ci95": None}
                         for metric, value in metrics.items()}
                  for name, metrics in points.items()}
        for name in names:
            if name != base:
                result[name]["delta_vs_base"] = {
                    metric: {"value": points[name][metric] - points[base][metric], "ci95": None}
                    for metric in points[base]
                }
        return result
    samples = {name: {metric: [] for metric in points[base]} for name in names}
    deltas = {name: {metric: [] for metric in points[base]} for name in names if name != base}
    n = len(text)
    for _ in range(repeats):
        idx = torch.tensor(rng.integers(0, n, n), device=text.device)
        sampled_hash = [hashes[int(i)] for i in idx.cpu()]
        permutation = torch.tensor(rng.permutation(n), device=text.device)
        baseline = function_metrics(projections[base][idx], text[idx], sampled_hash, permutation, idx)
        for metric in baseline:
            samples[base][metric].append(baseline[metric])
        for name in names:
            if name == base:
                continue
            item = function_metrics(projections[name][idx], text[idx], sampled_hash, permutation, idx)
            for metric in baseline:
                samples[name][metric].append(item[metric])
                deltas[name][metric].append(item[metric] - baseline[metric])
    result = {name: {metric: {"value": points[name][metric],
                              "ci95": [float(np.quantile(samples[name][metric], .025)),
                                       float(np.quantile(samples[name][metric], .975))]}
                     for metric in points[name]} for name in names}
    for name in names:
        if name == base:
            continue
        result[name]["delta_vs_base"] = {
            metric: {"value": points[name][metric] - points[base][metric],
                     "ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))]}
            for metric, values in deltas[name].items()
        }
    return result


def construct(ck,device):
    arm=ck["arm"]; a=ck["args"]; esm_dim=ck["esm_dim"]; latent=ck["latent_dim"]; text_dim=ck["text_dim"]
    if arm not in {"esm", "vae_z3"} and ck.get("attention_mode") != "stage_causal":
        raise ValueError("checkpoint predates the stage-causal z4 architecture; retrain the attribution model")
    if arm != "vae_z3" and ck.get("local_head_type") != "multilabel":
        raise ValueError("checkpoint predates the H1/H2 multilabel heads; retrain the attribution model")
    if arm != "vae_z3" and (ck.get("residue_labels") != list(RESIDUE_LABELS) or ck.get("domain_labels") != list(DOMAIN_LABELS)):
        raise ValueError("checkpoint uses a different H1/H2 feature taxonomy; retrain the attribution model")
    vae=None
    if arm=="esm":model=ESMBaseline(esm_dim,text_dim).to(device)
    else:
        vae=ProteinVAEContrastiveTrajectory(esm_dim,latent).to(device).eval(); vae.load_state_dict(ck["vae"])
        model=AttentionProjection(latent,text_dim).to(device) if arm=="vae_z3" else InterleavedBridge(latent,a["model_dim"],text_dim,esm_dim,1 if arm=="z3_transformer" else 4,layers=a["layers"],heads=a["heads"]).to(device)
    model.load_state_dict(ck["model"]); model.eval(); return arm,model,vae


def shuffle_valid_residues(trajectory, mask):
    shuffled = trajectory.clone()
    for row in range(len(trajectory)):
        length = int(mask[row].sum())
        shuffled[row, :length] = trajectory[row, torch.randperm(length, device=trajectory.device)]
    return shuffled


def multilabel_metrics(target, scores, labels):
    support = target.sum(0).astype(int)
    positive_rate = target.mean(0)
    per_hidden, macro, micro = {}, {}, {}
    for hidden, score in scores.items():
        per_label = {}
        valid_values = []
        for index, label in enumerate(labels):
            if support[index] == 0:
                per_label[label] = None
                continue
            value = float(average_precision_score(target[:, index], score[:, index]))
            per_label[label] = value
            valid_values.append(value)
        per_hidden[hidden] = per_label
        macro[hidden] = float(np.mean(valid_values)) if valid_values else None
        micro[hidden] = float(average_precision_score(target.reshape(-1), score.reshape(-1)))
    return {"labels": list(labels),
            "support_by_label": {label: int(support[i]) for i, label in enumerate(labels)},
            "positive_rate_by_label": {label: float(positive_rate[i]) for i, label in enumerate(labels)},
            "macro_auprc_by_hidden": macro, "micro_auprc_by_hidden": micro,
            "auprc_by_label_by_hidden": per_hidden}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",required=True); ap.add_argument("--manifest",required=True); ap.add_argument("--shard-index",required=True); ap.add_argument("--text-cache",action="append",required=True); ap.add_argument("--split",choices=["validation","test"],default="validation"); ap.add_argument("--limit",type=int,default=512); ap.add_argument("--batch-size",type=int,default=4); ap.add_argument("--bootstrap",type=int,default=200); ap.add_argument("--seed",type=int,default=91); ap.add_argument("--device",default="cuda:0"); ap.add_argument("--out",required=True); a=ap.parse_args(); torch.manual_seed(a.seed); device=torch.device(a.device if torch.cuda.is_available() else "cpu"); ck=torch.load(a.checkpoint,map_location=device); arm,model,vae=construct(ck,device); store=ShardStore(a.shard_index); texts=read_text_caches(a.text_cache); rows=sorted([json.loads(x) for x in open(a.manifest) if json.loads(x)["split"]==a.split],key=lambda r:r["accession"]); rows=rows[:a.limit] if a.limit else rows
    modes=["base"]
    if arm in {"trajectory_frozen","joint_function","joint_multitask"}:modes += ["stage_shuffle","residue_shuffle","zero_z0","zero_z1","zero_z2","zero_z3","z3_only","zero"]
    projections={m:[] for m in modes}; targets=[]; hashes=[]; structure_y=[]; domain_y=[]; structure_score={}; domain_score={}; structure_positions=domain_positions=0
    with torch.no_grad():
      for start in range(0,len(rows),a.batch_size):
        batch=rows[start:start+a.batch_size]; emb,mask=make_batch(batch,store,ck["esm_dim"],device); base=encode(arm,model,vae,emb,mask); projections["base"].append(base["text"]); targets.append(torch.stack([text_vector(texts,r["accession"]) for r in batch]).to(device)); hashes.extend(r["function_hash"] for r in batch)
        if len(modes)>1:
          traj=base["vae"]["states"]
          variants={"stage_shuffle":traj[:,:,[2,0,3,1]],"residue_shuffle":shuffle_valid_residues(traj,mask),"zero_z0":traj.clone(),"zero_z1":traj.clone(),"zero_z2":traj.clone(),"zero_z3":traj.clone(),"z3_only":torch.zeros_like(traj),"zero":torch.zeros_like(traj)}
          for i in range(4):variants[f"zero_z{i}"][:,:,i]=0
          variants["z3_only"][:,:,3]=traj[:,:,3]
          for name,value in variants.items():projections[name].append(model(value,mask)["text"])
        if arm=="esm" or arm=="joint_multitask":
          hidden=[base["residue_hidden"]] if arm=="esm" else base["stages"]
          for row_i,row in enumerate(batch):
            n=row["length"]
            if row["has_structure"]:
              y=local_target(row,n,device,RESIDUE_LABELS); structure_y.append(y.cpu()); structure_positions+=n
              for stage,h in enumerate(hidden):structure_score.setdefault(f"h{stage}" if arm!="esm" else "esm",[]).append(torch.sigmoid(model.structure_head(h[row_i,:n])).cpu())
            if row["has_domain"]:
              y=local_target(row,n,device,DOMAIN_LABELS); domain_y.append(y.cpu()); domain_positions+=n
              for stage,h in enumerate(hidden):domain_score.setdefault(f"h{stage}" if arm!="esm" else "esm",[]).append(torch.sigmoid(model.domain_head(h[row_i,:n])).cpu())
    text=F.normalize(torch.cat(targets),dim=-1); result={"arm":arm,"split":a.split,"records":len(rows),"function":{},"local":{}}
    normalized={mode:F.normalize(torch.cat(values),dim=-1) for mode,values in projections.items()}
    result["function"]=paired_bootstrap(normalized,text,hashes,a.bootstrap,a.seed)
    if structure_y:
      y=torch.cat(structure_y).numpy(); scores={k:torch.cat(v).numpy() for k,v in structure_score.items()}; result["local"]["structure"]={"proteins":len(structure_y),"positions":structure_positions,**multilabel_metrics(y,scores,RESIDUE_LABELS)}
    if domain_y:
      y=torch.cat(domain_y).numpy(); scores={k:torch.cat(v).numpy() for k,v in domain_score.items()}; result["local"]["domain"]={"proteins":len(domain_y),"positions":domain_positions,**multilabel_metrics(y,scores,DOMAIN_LABELS)}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); json.dump(result,open(a.out,"w"),indent=2); print(json.dumps(result,indent=2))
if __name__=="__main__":main()
