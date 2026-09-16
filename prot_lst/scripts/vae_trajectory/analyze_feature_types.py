from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def analyze(path: str) -> dict:
    feature_count = collections.Counter()
    protein_count = collections.Counter()
    residue_count = collections.Counter()
    records = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            records += 1
            seen = set()
            for feature in row.get("features", []):
                name = str(feature.get("type", "")).strip().lower()
                if not name:
                    continue
                feature_count[name] += 1
                seen.add(name)
                start = max(1, int(feature.get("start", 1)))
                end = max(start, int(feature.get("end", start)))
                residue_count[name] += end - start + 1
            protein_count.update(seen)
    names = sorted(feature_count, key=lambda name: (-protein_count[name], name))
    return {
        "records": records,
        "feature_types": len(names),
        "statistics": [
            {
                "type": name,
                "features": feature_count[name],
                "proteins": protein_count[name],
                "residues": residue_count[name],
            }
            for name in names
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    report = analyze(args.input)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
