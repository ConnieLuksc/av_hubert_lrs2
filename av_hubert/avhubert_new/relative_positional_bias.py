import torch
import torch.nn as nn

class RelativePositionalBias(nn.Module):
    """
    Learnable Relative Positional Bias, T5-style.
    The bias is added to the attention scores.
    """
    def __init__(self, num_heads: int, max_relative_positions: int, bidirectional: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.max_relative_positions = max_relative_positions
        self.bidirectional = bidirectional
        
        # Embedding table for relative positions.
        # Each head has its own set of relative position embeddings.
        self.relative_attention_bias = nn.Embedding(self.max_relative_positions * 2 + 1, self.num_heads)
        # Initialize with zeros or small values
        nn.init.zeros_(self.relative_attention_bias.weight)

    def _get_relative_positions(self, seq_len, device):
        """
        Generates a matrix of relative positions.
        Output shape: (seq_len, seq_len)
        """
        range_vec = torch.arange(seq_len, device=device)
        relative_matrix = range_vec[None, :] - range_vec[:, None]  # (query_pos, key_pos) -> relative_pos
        return relative_matrix

    def _map_to_buckets(self, relative_positions):
        """
        Maps relative positions to a limited number of buckets.
        Positive values for forward, negative for backward.
        Shifted by max_relative_positions to be non-negative indices for embedding table.
        """
        # Clip relative positions to the range [-max_relative_positions, max_relative_positions]
        clipped_relative_positions = torch.clamp(
            relative_positions,
            -self.max_relative_positions,
            self.max_relative_positions,
        )
        # Shift to make indices non-negative for embedding lookup
        return clipped_relative_positions + self.max_relative_positions

    def forward(self, seq_len_q: int, seq_len_k: int, device: torch.device) -> torch.Tensor:
        """
        Computes the relative positional bias.
        Args:
            seq_len_q (int): Query sequence length.
            seq_len_k (int): Key sequence length.
            device (torch.device): Device of the output tensor.

        Returns:
            torch.Tensor: Relative positional bias of shape (1, num_heads, seq_len_q, seq_len_k)
                          or (seq_len_q, seq_len_k) if num_heads is 1 and not splitting.
                          Fairseq MHA expects (bsz * num_heads, L, S) or (L,S) for attn_mask
                          So we need (num_heads, seq_len_q, seq_len_k) then it can be view-ed or expanded.
                          Let's return (seq_len_q, seq_len_k, num_heads) and permute later.
        """
        if seq_len_q != seq_len_k:
            # This implementation is simpler for self-attention where q_len == k_len
            # For cross-attention, relative positions are more complex.
            # Assuming self-attention for Conformer.
            # If cross-attention is needed, this module would need adjustment or a different approach.
            raise NotImplementedError("RelativePositionalBias currently assumes seq_len_q == seq_len_k for simplicity.")

        relative_positions = self._get_relative_positions(seq_len_q, device)
        bucket_indices = self._map_to_buckets(relative_positions)
        
        # Get bias: (seq_len_q, seq_len_k, num_heads)
        bias = self.relative_attention_bias(bucket_indices)
        
        # Reshape for MultiheadAttention: (seq_len_q, num_heads, seq_len_k) -> (num_heads, seq_len_q, seq_len_k)
        bias = bias.permute(2, 0, 1) # (num_heads, seq_len_q, seq_len_k)

        # Fairseq MultiheadAttention takes attn_mask of shape (tgt_len, src_len) or (bsz * num_heads, tgt_len, src_len)
        # If this bias is (num_heads, L, S), it needs to be expanded/repeated for batch if MHA expects bsz dimension.
        # However, MHA often broadcasts if the batch dim is missing.
        # Let's return (num_heads, seq_len_q, seq_len_k).
        # This can then be used by each batch element.
        return bias 