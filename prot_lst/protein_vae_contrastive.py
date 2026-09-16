from __future__ import annotations

import torch
from torch import Tensor, nn

from .protein_vae import DiagonalGaussian, ProteinVAE


class ProteinVAEContrastiveTrajectory(ProteinVAE):
    """Residue-preserving VAE with explicit z0 -> z1 -> z2 -> z3 states.

    z0: posterior sample during training, posterior mean during evaluation
    z1: depthwise local convolution, motif-scale context
    z2: Transformer context, long-range residue integration
    z3: functional projection state
    """

    def __init__(self, esm_dim: int, latent_dim: int = 256, heads: int = 8):
        super().__init__(esm_dim, latent_dim, layers=2, heads=heads, sequence_decoder=False)
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

    def states(self, latent: Tensor, mask: Tensor) -> Tensor:
        z0 = latent.masked_fill(~mask.unsqueeze(-1), 0.0)
        z1 = z0 + self.local(z0.transpose(1, 2)).transpose(1, 2)
        z1 = z1.masked_fill(~mask.unsqueeze(-1), 0.0)
        z2 = self.global_context(z1, src_key_padding_mask=~mask)
        z2 = z2.masked_fill(~mask.unsqueeze(-1), 0.0)
        z3 = self.functional(z2)
        z3 = z3.masked_fill(~mask.unsqueeze(-1), 0.0)
        return torch.stack((z0, z1, z2, z3), dim=2)

    def forward(self, esm_residue: Tensor, mask: Tensor, sample: bool = True,
                noise_scale: float = 1.0) -> dict[str, Tensor]:
        posterior = self.encode(esm_residue, mask)
        latent = posterior.sample(noise_scale) if sample else posterior.mean
        states = self.states(latent, mask)
        return {"latent": latent, "mean": posterior.mean, "logvar": posterior.logvar,
                "states": states, "kl": posterior.kl(mask)}
