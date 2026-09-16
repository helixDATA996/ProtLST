from __future__ import annotations

import torch


def anti_collapse_schedule(step: int, warmup_steps: int, ramp_steps: int,
                           beta_max: float, max_noise_scale: float = 1.0) -> tuple[float, float]:
    """Deterministic warm-up followed by joint noise/KL annealing."""
    if step <= warmup_steps:
        progress = 0.0
    elif ramp_steps <= 0:
        progress = 1.0
    else:
        progress = min(1.0, (step - warmup_steps) / ramp_steps)
    return max_noise_scale * progress, beta_max * progress


def free_bits_kl(mean: torch.Tensor, logvar: torch.Tensor, mask: torch.Tensor,
                 free_bits: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return raw KL, free-bits KL, and dimensions above the free-bits threshold."""
    terms = 0.5 * (mean.square() + logvar.exp() - 1.0 - logvar)
    per_dimension = terms[mask].mean(0)
    raw = per_dimension.sum()
    regularized = per_dimension.clamp_min(free_bits).sum()
    active_units = (per_dimension > free_bits).sum()
    return raw, regularized, active_units
