import csv
import json

import torch

from prot_lst.scripts.vae_trajectory.build_stage2_split import build_splits
from prot_lst.scripts.vae_trajectory.calibrate_function_vicreg import vicreg_terms
from prot_lst.scripts.vae_trajectory.cache_text_embeddings import view_mask, views
from prot_lst.scripts.vae_trajectory.feature_taxonomy import SUPPORTED_FEATURE_TYPES
from prot_lst.scripts.vae_trajectory.split_io import read_split_file
from prot_lst.scripts.vae_trajectory.train_attribution_experiment import (
    append_vicreg_queue, balanced_bce, early_stopping_update, global_pos_weight,
    positive_cosine_alignment, queued_vicreg_terms, retrieval_at_k,
)


def test_split_parser_uses_named_columns(tmp_path):
    path = tmp_path / "split.tsv"
    path.write_text("split\tnote\taccession\ntrain\tx\tA1\nvalidation\ty\tA2\n", encoding="utf-8")

    assert read_split_file(str(path)) == {"A1": "train", "A2": "validation"}


def test_function_cache_has_only_function_view():
    row = {"function_text": "  binds   ATP ", "go_terms": ["GO:1"], "ec_terms": ["1.2.3.4"]}

    assert views(row) == ["Function: binds ATP"]
    assert view_mask(row) == [1.0]


def test_fixed_pos_weight_trains_negative_only_batch():
    logits = torch.zeros(2, 3, 2, requires_grad=True)
    target = torch.zeros_like(logits)
    mask = torch.ones(2, 3, dtype=torch.bool)

    loss = balanced_bce(logits, target, mask, pos_weight=torch.tensor([2.0, 3.0]))
    loss.backward()

    assert torch.count_nonzero(logits.grad) == logits.numel()


def test_global_pos_weight_excludes_unannotated_records():
    labels = ("helix",)
    rows = [
        {"length": 4, "has_structure": True, "features": [{"type": "helix", "start": 1, "end": 2}]},
        {"length": 4, "has_structure": False, "features": []},
    ]

    weight = global_pos_weight(rows, labels, "residue")

    assert torch.allclose(weight, torch.tensor([1.0]))


def test_fixed_bce_skips_batch_without_supervision():
    logits = torch.randn(2, 3, 2, requires_grad=True)
    target = torch.zeros_like(logits)
    mask = torch.zeros(2, 3, dtype=torch.bool)

    loss = balanced_bce(logits, target, mask, pos_weight=torch.tensor([2.0, 3.0]))
    loss.backward()

    assert loss.item() == 0.0
    assert torch.count_nonzero(logits.grad) == 0


def test_function_retrieval_reports_top1_top5_top10():
    vectors = torch.eye(6)
    hashes = [str(index) for index in range(6)]

    metrics = retrieval_at_k(vectors, vectors, hashes)

    assert metrics == {"top1": 1.0, "top5": 1.0, "top10": 1.0}


def test_positive_alignment_has_no_cross_sample_negatives():
    protein_a = torch.tensor([[1.0, 0.0], [0.2, 0.8]], requires_grad=True)
    protein_b = protein_a.detach().clone().requires_grad_(True)
    text_a = torch.tensor([[0.8, 0.2], [0.0, 1.0]])
    text_b = torch.tensor([[0.8, 0.2], [-1.0, 0.0]])

    positive_cosine_alignment(protein_a, text_a).backward()
    positive_cosine_alignment(protein_b, text_b).backward()

    assert torch.allclose(protein_a.grad[0], protein_b.grad[0])


def test_vicreg_variance_penalizes_collapsed_batch():
    target = torch.eye(4)
    collapsed = torch.ones(4, 4)
    spread = torch.eye(4)

    _, collapsed_variance, _, _ = vicreg_terms(collapsed, target, gamma=0.7)
    _, spread_variance, _, _ = vicreg_terms(spread, target, gamma=0.7)

    assert collapsed_variance > spread_variance


def test_queued_vicreg_supports_single_sample_gradient_without_semantic_negatives():
    queue = []
    append_vicreg_queue(queue, torch.eye(4), 3)
    assert sum(len(item) for item in queue) == 3
    output = torch.tensor([[1.0, 1.0, 0.0, 0.0]], requires_grad=True)
    _, variance, covariance = queued_vicreg_terms(output, queue, gamma=0.5, covariance_dimensions=4)
    (variance + covariance).backward()
    assert output.grad is not None
    assert torch.count_nonzero(output.grad)
    assert all(not item.requires_grad for item in queue)


def test_early_stopping_requires_minimum_improvement_and_resets_patience():
    improved, best, stale = early_stopping_update(0.5005, 0.5, 1, 0.001)
    assert not improved and best == 0.5 and stale == 2
    improved, best, stale = early_stopping_update(0.502, best, stale, 0.001)
    assert improved and best == 0.502 and stale == 0


def test_stage2_split_keeps_heldout_unseen_and_duplicate_free(tmp_path):
    path = tmp_path / "data.jsonl"
    rows = []
    for index in range(12):
        rows.append({"accession": f"P{index}", "sequence": "A" * 50 + str(index),
                     "function_text": f"function {index}", "features": [
                         {"type": name, "start": 1, "end": 3}
                         for name in SUPPORTED_FEATURE_TYPES
                     ]})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    split, report = build_splits(str(path), pretrain_limit=6, train_count=2,
                                 validation_count=2, test_count=2, max_length=1024, seed=7)

    assert report["counts"] == {"train": 2, "validation": 2, "test": 2}
    assert {row["source_pool"] for row in split if row["split"] == "train"} == {"pretrain_seen"}
    assert {row["source_pool"] for row in split if row["split"] != "train"} == {"pretrain_unseen"}
    assert len({row["accession"] for row in split}) == 6
