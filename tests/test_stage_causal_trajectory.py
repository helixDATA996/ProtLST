import torch

from prot_lst.protein_vae_contrastive import ProteinVAEContrastiveTrajectory
from prot_lst.scripts.vae_trajectory.train_attribution_experiment import InterleavedBridge, encode


def make_models():
    vae = ProteinVAEContrastiveTrajectory(esm_dim=8, latent_dim=8, heads=2)
    bridge = InterleavedBridge(
        latent=8,
        model_dim=8,
        text_dim=6,
        esm_dim=8,
        stages=4,
        max_len=8,
        layers=2,
        heads=2,
    )
    return vae, bridge


def test_trajectory_and_z4_shapes_and_padding():
    vae, bridge = make_models()
    vae.eval()
    bridge.eval()
    embeddings = torch.randn(2, 5, 8)
    mask = torch.tensor([[True, True, True, False, False], [True] * 5])

    with torch.no_grad():
        vae_out = vae(embeddings, mask, sample=False)
        result = bridge(vae_out["states"], mask)

    assert vae_out["states"].shape == (2, 5, 4, 8)
    assert result["hidden"].shape == (2, 5, 4, 8)
    assert result["z4"].shape == (2, 5, 8)
    assert result["reconstruction"] is result["z4"]
    assert torch.count_nonzero(vae_out["states"][0, 3:]) == 0
    assert torch.count_nonzero(result["hidden"][0, 3:]) == 0
    assert torch.count_nonzero(result["z4"][0, 3:]) == 0


def test_stage_causal_attention_blocks_future_stages():
    _, bridge = make_models()
    bridge.eval()
    mask = torch.ones(1, 5, dtype=torch.bool)
    trajectory = torch.randn(1, 5, 4, 8)
    changed_future = trajectory.clone()
    changed_future[:, :, 2:] += 20.0 * torch.randn_like(changed_future[:, :, 2:])
    changed_early = trajectory.clone()
    changed_early[:, :, 0] += 20.0 * torch.randn_like(changed_early[:, :, 0])

    with torch.no_grad():
        base = bridge(trajectory, mask)["hidden"]
        changed = bridge(changed_future, mask)["hidden"]
        changed_from_early = bridge(changed_early, mask)["hidden"]

    torch.testing.assert_close(base[:, :, :2], changed[:, :, :2], rtol=0, atol=1e-6)
    assert not torch.allclose(base[:, :, 3], changed[:, :, 3])
    assert not torch.allclose(base[:, :, 2], changed_from_early[:, :, 2])
    assert not torch.allclose(base[:, :, 3], changed_from_early[:, :, 3])


def test_padding_does_not_change_valid_bridge_outputs():
    _, bridge = make_models()
    bridge.eval()
    short = torch.randn(1, 3, 4, 8)
    padded = torch.cat((short, torch.randn(1, 2, 4, 8)), dim=1)

    with torch.no_grad():
        short_output = bridge(short, torch.ones(1, 3, dtype=torch.bool))["hidden"]
        padded_output = bridge(
            padded, torch.tensor([[True, True, True, False, False]])
        )["hidden"][:, :3]

    torch.testing.assert_close(short_output, padded_output, rtol=1e-5, atol=1e-6)


def test_sampling_path_and_z4_reconstruction_train_logvar():
    vae, bridge = make_models()
    vae.eval()
    embeddings = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)

    with torch.no_grad():
        mean_a = vae(embeddings, mask, sample=False)["states"]
        mean_b = vae(embeddings, mask, sample=False)["states"]
        sample_a = vae(embeddings, mask, sample=True)["states"]
        sample_b = vae(embeddings, mask, sample=True)["states"]

    torch.testing.assert_close(mean_a, mean_b)
    assert not torch.allclose(sample_a, sample_b)

    vae.train()
    bridge.train()
    optimizer = torch.optim.AdamW([*vae.parameters(), *bridge.parameters()], lr=1e-3)
    before_reconstruct = bridge.reconstruct[-1].weight.detach().clone()
    vae_out = vae(embeddings, mask, sample=True)
    result = bridge(vae_out["states"], mask)
    loss = torch.nn.functional.mse_loss(result["z4"][mask], embeddings[mask])

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert vae.to_logvar.weight.grad is not None
    assert torch.count_nonzero(vae.to_logvar.weight.grad) > 0
    optimizer.step()
    assert not torch.equal(before_reconstruct, bridge.reconstruct[-1].weight.detach())


def test_encode_samples_only_for_trainable_training_vae():
    vae, bridge = make_models()
    embeddings = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)

    vae.train()
    train_a = encode("joint_multitask", bridge, vae, embeddings, mask)["trajectory"]
    train_b = encode("joint_multitask", bridge, vae, embeddings, mask)["trajectory"]
    assert not torch.allclose(train_a, train_b)

    vae.eval()
    eval_a = encode("joint_multitask", bridge, vae, embeddings, mask)["trajectory"]
    eval_b = encode("joint_multitask", bridge, vae, embeddings, mask)["trajectory"]
    torch.testing.assert_close(eval_a, eval_b)

    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    vae.eval()
    frozen_a = encode("trajectory_frozen", bridge, vae, embeddings, mask)["trajectory"]
    frozen_b = encode("trajectory_frozen", bridge, vae, embeddings, mask)["trajectory"]
    torch.testing.assert_close(frozen_a, frozen_b)
