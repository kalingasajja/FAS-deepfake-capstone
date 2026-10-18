"""
fusion/learned_fusion.py

Implements Stage 2: Learnable Score-Level and Feature-Level Fusion Models.

Models provided:
1. ScoreMLPFusion: A non-linear MLP that fuses the 3 probability scores + optional uncertainty features.
2. GatedMultimodalFusion: A Gated Multimodal Unit (GMU) that computes dynamic gating weights:
       z = sigmoid(W_g * [h_s, h_f, h_r])
   allowing the model to dynamically suppress rPPG under poor lighting / high motion.
3. CrossAttentionFusion: Multi-head cross-attention layer allowing spatial tokens to attend to
   physiological rPPG and frequency tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class ScoreMLPFusion(nn.Module):
    """
    Non-linear score-level fusion module.
    Takes [p_spatial, p_frequency, p_rppg] (+ optional metadata/uncertainty features).
    """
    def __init__(self, input_dim: int = 3, hidden_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns raw logit
        return self.net(x).squeeze(-1)


class GatedMultimodalFusion(nn.Module):
    """
    Gated Multimodal Unit (GMU) for intermediate feature representations.
    Inputs:
        h_spatial:   (batch_size, d_spatial)
        h_freq:      (batch_size, d_freq)
        h_rppg:      (batch_size, d_rppg)
    Dynamically routes modality weights using a learned gating function:
        z = softmax(Linear([h_s, h_f, h_r]))
        h_fused = z_s * f(h_s) + z_f * f(h_f) + z_r * f(h_r)
    """
    def __init__(
        self,
        d_spatial: int = 768,   # e.g. SigLIP / ViT embedding dimension
        d_freq: int = 128,      # e.g. FFT/DCT feature dimension
        d_rppg: int = 64,       # e.g. rPPG LSTM embedding dimension
        shared_dim: int = 128,
        num_classes: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        # Project all representations to a shared dimension
        self.proj_s = nn.Sequential(nn.Linear(d_spatial, shared_dim), nn.Tanh())
        self.proj_f = nn.Sequential(nn.Linear(d_freq, shared_dim), nn.Tanh())
        self.proj_r = nn.Sequential(nn.Linear(d_rppg, shared_dim), nn.Tanh())

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(d_spatial + d_freq + d_rppg, 3),
            nn.Softmax(dim=-1)
        )

        # Output classifier
        self.classifier = nn.Sequential(
            nn.Linear(shared_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(
        self,
        h_spatial: torch.Tensor,
        h_freq: torch.Tensor,
        h_rppg: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # 1. Project to shared space
        p_s = self.proj_s(h_spatial)
        p_f = self.proj_f(h_freq)
        p_r = self.proj_r(h_rppg)

        # 2. Compute dynamic gates: z has shape (batch_size, 3)
        concat_raw = torch.cat([h_spatial, h_freq, h_rppg], dim=-1)
        z = self.gate(concat_raw)

        # 3. Gated mixture
        z_s = z[:, 0:1]
        z_f = z[:, 1:2]
        z_r = z[:, 2:3]
        fused = z_s * p_s + z_f * p_f + z_r * p_r

        # 4. Classification logit
        logits = self.classifier(fused).squeeze(-1)

        return {
            "logits": logits,
            "gate_weights": z  # (batch_size, 3) -> can inspect which modality was relied on!
        }


class CrossAttentionFusion(nn.Module):
    """
    Multi-Head Cross-Attention Fusion.
    Visual Spatial tokens query Physiological (rPPG) and Frequency keys/values.
    """
    def __init__(
        self,
        d_spatial: int = 768,
        d_aux: int = 128,       # Aux dimension for concatenated [freq, rppg]
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.proj_q = nn.Linear(d_spatial, embed_dim)
        self.proj_kv = nn.Linear(d_aux, embed_dim)

        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, h_spatial: torch.Tensor, h_aux: torch.Tensor) -> torch.Tensor:
        # Ensure 3D shape (batch, seq_len=1, dim)
        if h_spatial.dim() == 2:
            h_spatial = h_spatial.unsqueeze(1)
        if h_aux.dim() == 2:
            h_aux = h_aux.unsqueeze(1)

        q = self.proj_q(h_spatial)
        kv = self.proj_kv(h_aux)

        attn_out, _ = self.mha(query=q, key=kv, value=kv)
        out = self.norm(q + attn_out).squeeze(1)
        return self.head(out).squeeze(-1)
