from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def last_token_pool(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return hidden[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]


class FrozenTextEncoder(nn.Module):
    """Qwen3-Embedding or causal-Qwen hidden-state teacher with one pooling contract."""

    def __init__(self, model_path: str, device: torch.device, max_length: int = 2048, causal: bool = False):
        super().__init__()
        self.model_path = model_path
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
        cls = AutoModelForCausalLM if causal else AutoModel
        self.model = cls.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = self.model.config.text_config.hidden_size
        self.hidden_size = int(hidden_size)

    @torch.no_grad()
    def encode(self, texts: list[str]) -> Tensor:
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.model.device)
        output = self.model(**batch, output_hidden_states=True)
        hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output.hidden_states[-1]
        return F.normalize(last_token_pool(hidden.float(), batch["attention_mask"]), dim=-1)


class Projection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.net(x), dim=-1)
