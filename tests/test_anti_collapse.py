import torch

from prot_lst.protein_vae import DiagonalGaussian
from prot_lst.scripts.vae_trajectory.vae_objectives import (
    anti_collapse_schedule,
    free_bits_kl,
)
from prot_lst.scripts.vae_trajectory.train_trajectory_vae import trajectory_loss


def test_zero_noise_sample_is_exact_posterior_mean():
    mean = torch.randn(2, 3, 4)
    posterior = DiagonalGaussian(mean, torch.zeros_like(mean))

    torch.testing.assert_close(posterior.sample(noise_scale=0.0), mean)
    assert not torch.allclose(posterior.sample(noise_scale=1.0), mean)


def test_anti_collapse_schedule_has_warmup_and_linear_ramp():
    assert anti_collapse_schedule(1, 20, 80, 0.001) == (0.0, 0.0)
    assert anti_collapse_schedule(20, 20, 80, 0.001) == (0.0, 0.0)
    assert anti_collapse_schedule(60, 20, 80, 0.001) == (0.5, 0.0005)
    assert anti_collapse_schedule(100, 20, 80, 0.001) == (1.0, 0.001)
    assert anti_collapse_schedule(200, 20, 80, 0.001) == (1.0, 0.001)


def test_free_bits_removes_pressure_below_threshold():
    mean = torch.full((1, 2, 4), 0.01, requires_grad=True)
    logvar = torch.zeros_like(mean, requires_grad=True)
    mask = torch.ones(1, 2, dtype=torch.bool)

    raw, regularized, active = free_bits_kl(mean, logvar, mask, free_bits=0.01)
    regularized.backward()

    assert raw < regularized
    assert active.item() == 0
    assert torch.count_nonzero(mean.grad) == 0
    assert torch.count_nonzero(logvar.grad) == 0


def test_combined_reconstruction_loss_trains_logvar_during_sampling():
    mean = torch.randn(2, 3, 4, requires_grad=True)
    logvar = torch.zeros_like(mean, requires_grad=True)
    mask = torch.ones(2, 3, dtype=torch.bool)
    latent = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
    projection = torch.nn.Linear(4, 4)
    prediction = projection(latent)
    target = torch.randn_like(prediction)

    loss, metrics = trajectory_loss(
        {"mean": mean, "logvar": logvar}, {"z4": prediction}, target, mask,
        beta=0.001, cosine_weight=0.01, free_bits=0.01,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert logvar.grad is not None
    assert torch.count_nonzero(logvar.grad) > 0
    assert set(("cosine_loss", "kl_regularized", "kl_active_units", "posterior_std")) <= metrics.keys()
