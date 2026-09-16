import torch
import numpy as np

from prot_lst.scripts.vae_trajectory.evaluate_attribution_experiment import multilabel_metrics
from prot_lst.scripts.vae_trajectory.feature_taxonomy import (
    DOMAIN_SOURCE_TYPES,
    EXCLUDED_FEATURE_TYPES,
    RESIDUE_SOURCE_TYPES,
)
from prot_lst.scripts.vae_trajectory.train_attribution_experiment import (
    DOMAIN_LABELS,
    RESIDUE_LABELS,
    ESMBaseline,
    InterleavedBridge,
    balanced_bce,
    local_target,
)


def test_local_target_preserves_overlapping_labels():
    row = {"features": [
        {"type": "active site", "start": 2, "end": 2},
        {"type": "binding site", "start": 2, "end": 3},
        {"type": "domain", "start": 1, "end": 4},
    ]}
    residue = local_target(row, 4, torch.device("cpu"), RESIDUE_LABELS)
    domain = local_target(row, 4, torch.device("cpu"), DOMAIN_LABELS)

    assert residue.shape == (4, 14)
    assert domain.shape == (4, 15)
    assert residue[1, RESIDUE_LABELS.index("active site")] == 1
    assert residue[1, RESIDUE_LABELS.index("binding site")] == 1
    assert residue[2, RESIDUE_LABELS.index("binding site")] == 1
    assert domain[:, DOMAIN_LABELS.index("domain")].sum() == 4


def test_multilabel_heads_have_14_and_15_outputs():
    baseline = ESMBaseline(esm_dim=8, text_dim=6)
    bridge = InterleavedBridge(8, 8, 6, 8, stages=4, max_len=8, layers=1, heads=2)
    hidden = torch.randn(2, 5, 8)

    assert baseline.structure_head(hidden).shape == (2, 5, 14)
    assert baseline.domain_head(hidden).shape == (2, 5, 15)
    assert bridge.structure_head(hidden).shape == (2, 5, 14)
    assert bridge.domain_head(hidden).shape == (2, 5, 15)


def test_non_core_features_are_mapped_to_level_specific_other():
    row = {"features": [
        {"type": "sequence conflict", "start": 2, "end": 2},
        {"type": "alternative sequence", "start": 2, "end": 4},
        {"type": "chain", "start": 1, "end": 4},
    ]}
    residue = local_target(row, 4, torch.device("cpu"), RESIDUE_LABELS)
    domain = local_target(row, 4, torch.device("cpu"), DOMAIN_LABELS)

    assert residue[:, RESIDUE_LABELS.index("other")].tolist() == [0, 1, 0, 0]
    assert domain[:, DOMAIN_LABELS.index("other")].tolist() == [0, 1, 1, 1]


def test_biological_non_core_features_have_independent_outputs():
    row = {"features": [
        {"type": "cross-link", "start": 2, "end": 2},
        {"type": "intramembrane", "start": 2, "end": 4},
    ]}
    residue = local_target(row, 4, torch.device("cpu"), RESIDUE_LABELS)
    domain = local_target(row, 4, torch.device("cpu"), DOMAIN_LABELS)

    assert residue[1, RESIDUE_LABELS.index("cross-link")] == 1
    assert residue[:, RESIDUE_LABELS.index("other")].sum() == 0
    assert domain[1:4, DOMAIN_LABELS.index("intramembrane")].sum() == 3
    assert domain[:, DOMAIN_LABELS.index("other")].sum() == 0


def test_taxonomy_covers_all_36_observed_feature_types_without_overlap():
    assert not (RESIDUE_SOURCE_TYPES & DOMAIN_SOURCE_TYPES)
    assert not (RESIDUE_SOURCE_TYPES & EXCLUDED_FEATURE_TYPES)
    assert not (DOMAIN_SOURCE_TYPES & EXCLUDED_FEATURE_TYPES)
    assert len(RESIDUE_SOURCE_TYPES | DOMAIN_SOURCE_TYPES | EXCLUDED_FEATURE_TYPES) == 36


def test_balanced_bce_uses_only_classes_with_batch_positives():
    logits = torch.zeros(2, 3, 4, requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, 1, 1] = 1
    target[1, 2, 3] = 1
    mask = torch.tensor([[True, True, False], [True, True, True]])

    loss = balanced_bce(logits, target, mask)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:, :, 0]) == 0
    assert torch.count_nonzero(logits.grad[:, :, 2]) == 0
    assert torch.count_nonzero(logits.grad[:, :, 1]) > 0
    assert torch.count_nonzero(logits.grad[:, :, 3]) > 0


def test_multilabel_metrics_report_per_class_macro_and_micro():
    target = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
    score = np.array([[0.9, 0.1, 0.2], [0.2, 0.8, 0.2], [0.8, 0.2, 0.2]], dtype=np.float32)
    metrics = multilabel_metrics(target, {"h1": score}, ("a", "b", "missing"))

    assert metrics["support_by_label"] == {"a": 2, "b": 1, "missing": 0}
    assert metrics["auprc_by_label_by_hidden"]["h1"]["a"] == 1.0
    assert metrics["auprc_by_label_by_hidden"]["h1"]["b"] == 1.0
    assert metrics["auprc_by_label_by_hidden"]["h1"]["missing"] is None
    assert metrics["macro_auprc_by_hidden"]["h1"] == 1.0
    assert 0.0 <= metrics["micro_auprc_by_hidden"]["h1"] <= 1.0
