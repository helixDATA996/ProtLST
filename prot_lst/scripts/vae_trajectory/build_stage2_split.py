from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from prot_lst.scripts.vae_trajectory.feature_taxonomy import (
    DOMAIN_LABELS, RESIDUE_LABELS, map_feature_type,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def eligible(row: dict, max_length: int) -> bool:
    return 50 <= len(row.get("sequence", "")) <= max_length


def labels_for(row: dict) -> set[str]:
    return {
        f"{level}:{mapped}"
        for level in ("residue", "domain")
        for feature in row.get("features", [])
        if (mapped := map_feature_type(feature.get("type", ""), level))
    }


def build_splits(data: str, pretrain_limit: int, train_count: int,
                 validation_count: int, test_count: int, max_length: int,
                 seed: int) -> tuple[list[dict], dict]:
    pretrain, unseen = [], []
    eligible_index = 0
    with open(data, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not eligible(row, max_length):
                continue
            eligible_index += 1
            if not row.get("function_text", "").strip():
                continue
            item = {
                "accession": row["accession"],
                "sequence_hash": digest(row["sequence"]),
                "function_hash": digest(row["function_text"].strip()),
                "labels": labels_for(row),
            }
            (pretrain if eligible_index <= pretrain_limit else unseen).append(item)
    pretrain.sort(key=lambda row: digest(f"{seed}:train:{row['accession']}"))
    if len(pretrain) < train_count:
        raise RuntimeError(f"only {len(pretrain)} eligible pretraining records for {train_count} training records")
    train = pretrain[:train_count]
    used_sequence = {row["sequence_hash"] for row in train}
    used_function = {row["function_hash"] for row in train}
    candidates = [row for row in unseen if row["sequence_hash"] not in used_sequence and row["function_hash"] not in used_function]
    candidates.sort(key=lambda row: digest(f"{seed}:heldout:{row['accession']}"))
    heldout = []
    for row in candidates:
        if row["sequence_hash"] in used_sequence or row["function_hash"] in used_function:
            continue
        heldout.append(row)
        used_sequence.add(row["sequence_hash"]); used_function.add(row["function_hash"])
        if len(heldout) == validation_count + test_count:
            break
    if len(heldout) != validation_count + test_count:
        raise RuntimeError("not enough duplicate-free held-out records")
    validation = heldout[:validation_count]; test = heldout[validation_count:]
    output = [
        {"accession": row["accession"], "source_pool": source, "split": split}
        for values, source, split in ((train, "pretrain_seen", "train"),
                                      (validation, "pretrain_unseen", "validation"),
                                      (test, "pretrain_unseen", "test"))
        for row in values
    ]
    label_support = {}
    for name, values in (("train", train), ("validation", validation), ("test", test)):
        counts = Counter(label for row in values for label in row["labels"])
        label_support[name] = {label: counts[label] for label in
                               [*(f"residue:{x}" for x in RESIDUE_LABELS), *(f"domain:{x}" for x in DOMAIN_LABELS)]}
    if any(value == 0 for value in label_support["train"].values()):
        raise RuntimeError("one or more H1/H2 labels have no training examples")
    report = {"counts": {"train": len(train), "validation": len(validation), "test": len(test)},
              "label_support": label_support, "pretrain_limit": pretrain_limit, "seed": seed,
              "exact_sequence_or_function_overlap": 0}
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True); parser.add_argument("--pretrain-limit", type=int, default=50000)
    parser.add_argument("--train-count", type=int, default=45000); parser.add_argument("--validation-count", type=int, default=5000)
    parser.add_argument("--test-count", type=int, default=5000); parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260910); args = parser.parse_args()
    rows, report = build_splits(args.data, args.pretrain_limit, args.train_count, args.validation_count,
                                args.test_count, args.max_length, args.seed)
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("accession", "source_pool", "split"), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report["counts"], "out": str(output), "report": args.report}, indent=2))


if __name__ == "__main__":
    main()
