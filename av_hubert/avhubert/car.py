import torch
import torch.nn as nn
import torch.nn.functional as F

class CARModule(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim_per_conv: int, kernel_sizes: list = [3, 5, 7]):
        super().__init__()
        self.input_dim = input_dim  # Original feature dimension 'e' or 'D'
        self.bottleneck_dim_per_conv = bottleneck_dim_per_conv # Compressed dimension 'c' for each conv
        self.kernel_sizes = kernel_sizes

        self.convs = nn.ModuleList()
        for k_size in self.kernel_sizes:
            # Conv1d expects input: (N, C_in, L_in)
            # Features are typically (N, L, C_in) in Fairseq, so we'll permute.
            self.convs.append(
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=bottleneck_dim_per_conv,
                    kernel_size=k_size,
                    padding=(k_size - 1) // 2, # Ensures 'same' padding for odd kernels
                    bias=False # Often BN follows conv, so bias can be false
                )
            )

        concatenated_channels = bottleneck_dim_per_conv * len(self.kernel_sizes)

        # BatchNorm1d expects input: (N, C, L)
        self.batch_norm = nn.BatchNorm1d(num_features=concatenated_channels)

        # Linear layer to project back to original dimension 'input_dim'
        self.linear_recovery = nn.Linear(concatenated_channels, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (N, L, D) - Batch, Sequence Length, Feature Dimension
        
        # Permute for Conv1d: (N, L, D) -> (N, D, L)
        x_permuted = x.permute(0, 2, 1)

        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(x_permuted)) # Each output: (N, c, L)

        # Concatenate along the channel dimension (dim=1)
        concatenated_features = torch.cat(conv_outputs, dim=1) # Shape: (N, 3*c, L)

        # Batch Normalization
        bn_features = self.batch_norm(concatenated_features) # Shape: (N, 3*c, L)

        # Permute for Linear layer: (N, 3*c, L) -> (N, L, 3*c)
        bn_features_permuted = bn_features.permute(0, 2, 1)

        # Linear recovery: (N, L, 3*c) -> (N, L, D)
        recovered_features = self.linear_recovery(bn_features_permuted)
        
        # As per Equation 3: X'_j = Linear(FBN) + Xj
        # The CAR module itself should output Linear(FBN).
        # The residual addition of Xj will happen in the encoder layer.
        return recovered_features