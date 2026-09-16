from __future__ import annotations

import csv


def read_split_file(path: str) -> dict[str, str]:
    """Read an accession/split TSV by header name."""
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or ())
        if not {"accession", "split"} <= fields:
            raise ValueError("split TSV must contain accession and split columns")
        result = {}
        for row in reader:
            split = row["split"].strip()
            accession = row["accession"].strip()
            if split not in {"train", "validation", "test"}:
                raise ValueError(f"invalid split {split!r} for {accession!r}")
            if accession in result:
                raise ValueError(f"duplicate accession in split TSV: {accession}")
            result[accession] = split
    return result
