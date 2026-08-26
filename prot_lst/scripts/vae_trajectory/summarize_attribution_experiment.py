from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

FUNCTION_METRICS = ("group_r10", "cosine_gap", "pairwise_similarity_correlation")
BASELINES = ("esm", "vae_z3", "z3_transformer")

def stats(items, metric):
    values = [float(item["function"]["base"][metric]["value"]) for item in items]
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, "seeds": len(values), "values": values}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", action="append", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    rows = [json.load(open(p, encoding="utf-8")) for p in a.input]; grouped = {}
    for row in rows: grouped.setdefault((row["arm"], row["split"]), []).append(row)
    summary = {"experiments": {}, "decision": {}, "inputs": a.input}
    for (arm, split), items in sorted(grouped.items()):
        entry = {metric: stats(items, metric) for metric in FUNCTION_METRICS}; entry["ablations"] = {}
        modes = set().union(*(item["function"] for item in items)) - {"base"}
        for mode in sorted(modes):
            vals, deltas = [], []
            for item in items:
                block = item["function"].get(mode)
                if not block: continue
                vals.append(block["pairwise_similarity_correlation"]["value"])
                delta = block.get("delta_vs_base", {}).get("pairwise_similarity_correlation")
                if delta: deltas.append(delta)
            entry["ablations"][mode] = {"pairwise_similarity_correlation": {"mean": float(np.mean(vals)) if vals else None, "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, "seeds": len(vals)}, "delta_vs_base": deltas}
        for task in ("structure", "domain"):
            candidates = [x.get("local", {}).get(task) for x in items if x.get("local", {}).get(task)]
            if candidates:
                keys = sorted(set().union(*(x.get("auprc_by_hidden", {}) for x in candidates)))
                entry.setdefault("local", {})[task] = {"proteins": int(np.mean([x["proteins"] for x in candidates])), "positions": int(np.mean([x["positions"] for x in candidates])), "positive_rate": float(np.mean([x["positive_rate"] for x in candidates])), "auprc_by_hidden": {k: float(np.mean([x["auprc_by_hidden"][k] for x in candidates if k in x["auprc_by_hidden"]])) for k in keys}}
        summary["experiments"][f"{arm}:{split}"] = entry
    for split in ("validation", "test"):
        multi = summary["experiments"].get(f"joint_multitask:{split}"); function = summary["experiments"].get(f"joint_function:{split}"); baselines = [summary["experiments"].get(f"{x}:{split}") for x in BASELINES]
        if not multi or not function or not all(baselines): summary["decision"][split] = {"status": "insufficient_inputs"}; continue
        frozen = summary["experiments"].get(f"trajectory_frozen:{split}"); mscore = multi["pairwise_similarity_correlation"]["mean"]
        summary["decision"][split] = {"status": "evaluated", "full_trajectory_beats_all_baselines": bool(all(mscore > b["pairwise_similarity_correlation"]["mean"] for b in baselines)), "joint_training_beats_frozen_trajectory": bool(frozen and mscore > frozen["pairwise_similarity_correlation"]["mean"]), "multitask_beats_function_only": bool(mscore > function["pairwise_similarity_correlation"]["mean"])}
        for mode in ("stage_shuffle", "zero_z1", "zero_z2"):
            records = multi.get("ablations", {}).get(mode, {}).get("delta_vs_base", []); drops = [-float(x["value"]) for x in records]; cis = [x["ci95"] for x in records]
            summary["decision"][split][f"{mode}_passes"] = bool(len(drops) == 3 and np.mean(drops) >= 0.10 * max(abs(mscore), 1e-8) and all(float(ci[1]) < 0 for ci in cis))
        structure = multi.get("local", {}).get("structure", {}).get("auprc_by_hidden", {}); domain = multi.get("local", {}).get("domain", {}).get("auprc_by_hidden", {})
        esm = summary["experiments"].get(f"esm:{split}", {}).get("local", {})
        esm_structure = esm.get("structure", {}).get("auprc_by_hidden", {}).get("esm", float("inf")); esm_domain = esm.get("domain", {}).get("auprc_by_hidden", {}).get("esm", float("inf"))
        summary["decision"][split]["h1_structure_specific"] = bool(structure.get("h1", -1) == max(structure.values(), default=-2) and structure.get("h1", -1) > esm_structure)
        summary["decision"][split]["h2_domain_specific"] = bool(domain.get("h2", -1) == max(domain.values(), default=-2) and domain.get("h2", -1) > esm_domain)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(summary, open(a.out, "w", encoding="utf-8"), indent=2); print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()
