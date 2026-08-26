from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .protein_vae import DiagonalGaussian, ProteinVAE


class ProteinVAEContrastiveTrajectory(ProteinVAE):
    """Residue-preserving VAE with explicit z0 -> z1 -> z2 -> z3 states.

    z0: posterior mean, local residue representation
    z1: depthwise local convolution, motif-scale context
    z2: Transformer context, long-range residue integration
    z3: functional projection state
    """

    def __init__(self, esm_dim: int, latent_dim: int = 256, heads: int = 8):
        super().__init__(esm_dim, latent_dim, layers=2, heads=heads)
        self.local = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim, kernel_size=5, padding=2, groups=latent_dim),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=1), nn.GELU(), nn.GroupNorm(1, latent_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=heads, dim_feedforward=4 * latent_dim,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.global_context = nn.TransformerEncoder(layer, num_layers=2)
        self.functional = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, latent_dim), nn.Tanh())

    def states(self, mean: Tensor, mask: Tensor) -> Tensor:
        z0 = mean
        z1 = z0 + self.local(z0.transpose(1, 2)).transpose(1, 2)
        z1 = z1.masked_fill(~mask.unsqueeze(-1), 0.0)
        z2 = self.global_context(z1, src_key_padding_mask=~mask)
        z2 = z2.masked_fill(~mask.unsqueeze(-1), 0.0)
        z3 = self.functional(z2)
        return torch.stack((z0, z1, z2, z3), dim=2)

    def forward(self, esm_residue: Tensor, mask: Tensor, sample: bool = True) -> dict[str, Tensor]:
        posterior = self.encode(esm_residue, mask)
        states = self.states(posterior.mean, mask)
        latent = posterior.sample() if sample else posterior.mean
        logits = self.decode(latent, mask)
        state_logits = torch.stack([self.decode(states[:, :, index, :], mask) for index in range(states.shape[2])], dim=2)
        return {"latent": latent, "mean": posterior.mean, "logvar": posterior.logvar,
                "states": states, "logits": logits, "state_logits": state_logits, "kl": posterior.kl(mask)}


def masked_pool(states: Tensor, mask: Tensor) -> Tensor:
    weights = mask[:, :, None, None].to(states.dtype)
    return (states * weights).sum(1) / weights.sum(1).clamp_min(1.0)


def supervised_contrastive(states: Tensor, labels: list[set[int]], temperature: float = 0.1) -> Tensor:
    """Multi-label supervised contrastive loss over [B,T,D] pooled states."""
    if states.shape[0] < 2:
        return states.new_zeros(())
    z = F.normalize(states, dim=-1)
    similarity = torch.einsum("btd,ctd->bct", z, z) / temperature
    eye = torch.eye(states.shape[0], device=states.device, dtype=torch.bool)
    losses = []
    for i in range(states.shape[0]):
        positives = torch.tensor([bool(labels[i] & labels[j]) for j in range(states.shape[0])], device=states.device)
        positives[i] = False
        if not positives.any():
            continue
        logits = similarity[i]
        logits = logits.masked_fill(eye[i, :, None], -1e4)
        log_prob = logits - torch.logsumexp(logits, dim=0, keepdim=True)
        losses.append(-log_prob[positives].mean())
    return torch.stack(losses).mean() if losses else states.new_zeros(())
