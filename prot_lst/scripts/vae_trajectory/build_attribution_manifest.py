from __future__ import annotations
import argparse, hashlib, json, os, random
from pathlib import Path

STRUCT = {"helix", "strand", "turn", "disulfide bond", "glycosylation site", "active site", "binding site", "modified residue", "short sequence motif"}
DOMAIN = {"domain", "region of interest", "repeat", "zinc finger region", "dna-binding region", "coiled-coil region", "transmembrane region", "topological domain", "signal peptide", "transit peptide"}

def text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--split-file", required=True)
    ap.add_argument("--text-cache", action="append", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max-length", type=int, default=1024); ap.add_argument("--seed", type=int, default=20260823); ap.add_argument("--limit-per-split", type=int, default=0)
    a = ap.parse_args(); rng = random.Random(a.seed)
    split = {}
    with open(a.split_file, encoding="utf-8") as f:
        next(f)
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3: split[fields[0]] = fields[2]
    cache = {}
    for path in a.text_cache: cache.update(__import__("torch").load(path, map_location="cpu")["cache"])
    rows = []
    for line in open(a.data, encoding="utf-8"):
        row = json.loads(line); acc = row.get("accession", ""); seq = row.get("sequence", "")
        item = cache.get(acc)
        if split.get(acc) not in {"train", "validation", "test"} or not (50 <= len(seq) <= a.max_length): continue
        if item is None or item.get("view_mask", [0])[0] <= 0: continue
        struct_types = sorted({f.get("type", "").lower() for f in row.get("features", []) if f.get("type", "").lower() in STRUCT})
        domain_types = sorted({f.get("type", "").lower() for f in row.get("features", []) if f.get("type", "").lower() in DOMAIN})
        function_text = row.get("function_text", "").strip()
        relevant = [f for f in row.get("features", []) if f.get("type", "").lower() in STRUCT | DOMAIN]
        rows.append({"accession": acc, "split": split[acc], "sequence": seq, "length": len(seq),
                     "function_hash": text_hash(function_text), "function_text": function_text,
                     "has_structure": bool(struct_types), "has_domain": bool(domain_types),
                     "structure_types": struct_types, "domain_types": domain_types, "features": relevant})
    rng.shuffle(rows)
    if a.limit_per_split:
        rows = [row for split_name in ("train", "validation", "test") for row in [r for r in rows if r["split"] == split_name][:a.limit_per_split]]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(json.dumps({"completed": True, "records": len(rows), "by_split": {s: sum(r["split"] == s for r in rows) for s in ("train", "validation", "test")}, "unique_function": len({r["function_hash"] for r in rows}), "manifest": os.path.abspath(a.out)}, indent=2))

if __name__ == "__main__": main()
