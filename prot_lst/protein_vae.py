from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {x: i for i, x in enumerate(AA)}
PAD_ID = 20


def encode_sequences(sequences: list[str], device: torch.device) -> tuple[Tensor, Tensor]:
    lengths = [len(x) for x in sequences]
    width = max(lengths)
    tokens = torch.full((len(sequences), width), PAD_ID, dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), width), dtype=torch.bool, device=device)
    for row, sequence in enumerate(sequences):
        ids = [AA_TO_ID.get(char, PAD_ID) for char in sequence]
        tokens[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        mask[row, : len(ids)] = torch.tensor([value < len(AA) for value in ids], dtype=torch.bool, device=device)
    return tokens, mask


class DiagonalGaussian:
    def __init__(self, mean: Tensor, logvar: Tensor):
        self.mean = mean
        self.logvar = logvar.clamp(-12.0, 8.0)

    def sample(self) -> Tensor:
        return self.mean + torch.exp(0.5 * self.logvar) * torch.randn_like(self.mean)

    def kl(self, mask: Tensor) -> Tensor:
        value = 0.5 * (self.mean.square() + self.logvar.exp() - 1.0 - self.logvar)
        value = value.sum(-1)
        return (value * mask).sum() / mask.sum().clamp_min(1)


class ProteinVAE(nn.Module):
    """Residue-preserving VAE on frozen ESM-C representations."""

    def __init__(self, esm_dim: int, latent_dim: int = 256, layers: int = 3, heads: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_norm = nn.LayerNorm(esm_dim)
        self.input_proj = nn.Linear(esm_dim, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=heads, dim_feedforward=4 * latent_dim,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.to_mean = nn.Linear(latent_dim, latent_dim)
        self.to_logvar = nn.Linear(latent_dim, latent_dim)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=heads, dim_feedforward=4 * latent_dim,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=layers)
        self.sequence_head = nn.Linear(latent_dim, len(AA))
        self.summary = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, latent_dim), nn.Tanh())

    def encode(self, esm_residue: Tensor, mask: Tensor) -> DiagonalGaussian:
        x = self.input_proj(self.input_norm(esm_residue))
        x = self.encoder(x, src_key_padding_mask=~mask)
        return DiagonalGaussian(self.to_mean(x), self.to_logvar(x))

    def decode(self, latent: Tensor, mask: Tensor) -> Tensor:
        x = self.decoder(latent, src_key_padding_mask=~mask)
        return self.sequence_head(x)

    def forward(self, esm_residue: Tensor, mask: Tensor, sample: bool = True) -> dict[str, Tensor]:
        posterior = self.encode(esm_residue, mask)
        latent = posterior.sample() if sample else posterior.mean
        logits = self.decode(latent, mask)
        return {
            "latent": latent,
            "mean": posterior.mean,
            "logvar": posterior.logvar,
            "logits": logits,
            "kl": posterior.kl(mask),
        }

    def loss(self, output: dict[str, Tensor], tokens: Tensor, mask: Tensor, beta: float) -> dict[str, Tensor]:
        logits = output["logits"]
        reconstruction = F.cross_entropy(logits[mask], tokens[mask])
        prediction = logits.argmax(-1)
        accuracy = (prediction[mask] == tokens[mask]).float().mean()
        total = reconstruction + beta * output["kl"]
        return {"loss": total, "reconstruction": reconstruction, "kl": output["kl"], "accuracy": accuracy}

    @torch.no_grad()
    def summary_latent(self, latent: Tensor, mask: Tensor) -> Tensor:
        pooled = (latent * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.summary(pooled)
