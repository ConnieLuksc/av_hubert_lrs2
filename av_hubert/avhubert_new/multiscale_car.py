import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleCAR(nn.Module):
    def __init__(self, embed_dim, compress_dim=256, kernel_sizes=[3, 5, 7], dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.compress_dim = compress_dim
        self.kernel_sizes = kernel_sizes

        self.conv_branches = nn.ModuleList()
        for k in kernel_sizes:
            self.conv_branches.append(
                nn.Sequential(
                    nn.Conv1d(embed_dim, compress_dim, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(compress_dim),
                    nn.ReLU(),
                )
            )
        
        self.proj = nn.Linear(len(kernel_sizes) * compress_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (B, T, C)
        residual = x
        x = x.transpose(1, 2)  # (B, C, T)
        
        branch_outputs = []
        for branch in self.conv_branches:
            branch_outputs.append(branch(x))
        
        x = torch.cat(branch_outputs, dim=1)  # (B, len(kernels)*compress_dim, T)
        x = x.transpose(1, 2)  # (B, T, len(kernels)*compress_dim)
        
        x = self.proj(x)
        x = self.dropout(x)
        
        x = x + residual
        return x 