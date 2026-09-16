import json
import sys

from prot_lst.scripts.vae_trajectory import summarize_attribution_experiment as summarize


def test_summary_preserves_multilabel_metrics(tmp_path, monkeypatch):
    evaluation = {
        "arm": "joint_multitask",
        "split": "validation",
        "function": {
            "base": {
                "group_r10": {"value": 0.5},
                "cosine_gap": {"value": 0.2},
                "pairwise_similarity_correlation": {"value": 0.3},
            }
        },
        "local": {
            "structure": {
                "proteins": 2,
                "positions": 8,
                "labels": ["helix", "active site"],
                "positive_rate_by_label": {"helix": 0.25, "active site": 0.125},
                "macro_auprc_by_hidden": {"h1": 0.8},
                "micro_auprc_by_hidden": {"h1": 0.75},
                "auprc_by_label_by_hidden": {
                    "h1": {"helix": 0.9, "active site": 0.7}
                },
            }
        },
    }
    source = tmp_path / "evaluation.json"
    output = tmp_path / "summary.json"
    source.write_text(json.dumps(evaluation), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "summarize_attribution_experiment.py",
        "--input", str(source),
        "--out", str(output),
    ])

    summarize.main()

    result = json.loads(output.read_text(encoding="utf-8"))
    local = result["experiments"]["joint_multitask:validation"]["local"]["structure"]
    assert local["labels"] == ["helix", "active site"]
    assert local["macro_auprc_by_hidden"]["h1"] == 0.8
    assert local["micro_auprc_by_hidden"]["h1"] == 0.75
    assert local["auprc_by_label_by_hidden"]["h1"]["helix"] == 0.9
    assert result["decision"]["validation"]["status"] == "insufficient_inputs"
