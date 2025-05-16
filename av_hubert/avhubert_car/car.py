import torch
import torch.nn as nn
from fairseq.modules import LayerNorm
from fairseq import utils

class CARModule(nn.Module):
    """Compression and Recovery (CAR) Module"""
    def __init__(self, d_model, d_compress, activation_fn, dropout):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.compress_proj = nn.Linear(d_model, d_compress)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.recover_proj = nn.Linear(d_compress, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.compress_proj(x)
        x = self.activation_fn(x)
        x = self.recover_proj(x)
        x = self.dropout(x)
        return x + residual 