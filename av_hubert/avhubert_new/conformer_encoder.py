import math
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    MultiheadAttention,
)
from dataclasses import dataclass, field
from typing import List
# For quantization noise, if needed later
# from fairseq.modules import quant_noise as apply_quant_noise_

if __name__ == '__main__' or __name__ == 'conformer_encoder':
    from multiscale_car import MultiScaleCAR
    from relative_positional_bias import RelativePositionalBias
else:
    from .multiscale_car import MultiScaleCAR
    from .relative_positional_bias import RelativePositionalBias


class PositionwiseFeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, activation_fn_name, dropout, activation_dropout):
        super().__init__()
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.activation_dropout_module = FairseqDropout(activation_dropout, module_name=self.__class__.__name__)
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn_name)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x


class ConformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        attention_heads: int,
        dropout: float,
        attention_dropout: float,
        activation_dropout: float,
        activation_fn: str = "relu",
        # CAR module params
        car_compress_dim: int = 256,
        car_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7]),
        car_dropout: float = 0.1,
        max_relative_positions: int = 128, # New parameter
        # LayerNorm options
        layer_norm_eps: float = 1e-5,
        # Quantization noise (optional)
        # quant_noise: float = 0.0,
        # quant_noise_block_size: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)

        # Layer norms
        self.norm_ffn1 = LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm_attn = LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm_car = LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm_ffn2 = LayerNorm(embed_dim, eps=layer_norm_eps)

        # FFN1
        self.ffn1 = PositionwiseFeedForward(
            embed_dim, ffn_embed_dim, activation_fn, dropout, activation_dropout
        )

        # Self-Attention
        self.self_attn = MultiheadAttention(
            embed_dim,
            attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            # quant_noise=quant_noise,
            # quant_noise_block_size=quant_noise_block_size,
        )
        
        # Relative Positional Bias for Attention
        self.relative_positional_bias = RelativePositionalBias(
            num_heads=attention_heads,
            max_relative_positions=max_relative_positions,
        )

        # Multi-Scale CAR
        if isinstance(car_kernel_sizes, str):
            car_kernel_sizes = utils.eval_str_list(car_kernel_sizes, type=int)

        self.multi_scale_car = MultiScaleCAR(
            embed_dim,
            compress_dim=car_compress_dim,
            kernel_sizes=car_kernel_sizes,
            dropout=car_dropout,
        )

        # FFN2
        self.ffn2 = PositionwiseFeedForward(
            embed_dim, ffn_embed_dim, activation_fn, dropout, activation_dropout
        )

    def forward(self, x: torch.Tensor, encoder_padding_mask: Optional[torch.Tensor], attn_mask: Optional[torch.Tensor] = None):
        # x: (B, T, C)
        # encoder_padding_mask: (B, T) if provided

        # FFN1 sublayer (with half-step residual)
        residual = x
        x_norm = self.norm_ffn1(x)
        x_ffn1 = self.ffn1(x_norm)
        x = residual + 0.5 * self.dropout_module(x_ffn1)

        # MHSA sublayer
        residual = x
        x_norm = self.norm_attn(x)
        # MultiheadAttention expects (T, B, C)
        x_for_attn = x_norm.transpose(0, 1)
        T_seq, B_size, _ = x_for_attn.shape # T_seq is target/source length, B_size is batch size
        
        # Compute relative positional bias
        rel_pos_bias = self.relative_positional_bias(T_seq, T_seq, x.device) # Shape (num_heads, T_seq, T_seq)

        # PyTorch's F.multi_head_attention_forward expects a 3D attn_mask to be (batch_size * num_heads, T, T)
        # Our rel_pos_bias is (num_heads, T, T)
        current_num_heads = self.self_attn.num_heads 

        if B_size > 1 and rel_pos_bias.dim() == 3 and rel_pos_bias.size(0) == current_num_heads:
            # Expand (num_heads, T, T) to (B_size, num_heads, T, T) then reshape to (B_size * num_heads, T, T)
            attn_mask_for_mha = rel_pos_bias.unsqueeze(0).expand(B_size, current_num_heads, T_seq, T_seq)
            attn_mask_for_mha = attn_mask_for_mha.reshape(B_size * current_num_heads, T_seq, T_seq)
        else:
            # If B_size is 1, (num_heads, T, T) is effectively (1 * num_heads, T, T) and should be fine.
            # Or if rel_pos_bias isn't the expected (num_heads, T,T) shape, pass as is.
            attn_mask_for_mha = rel_pos_bias

        attn_output, _ = self.self_attn(
            query=x_for_attn,
            key=x_for_attn,
            value=x_for_attn,
            key_padding_mask=encoder_padding_mask, # (B, T)
            attn_mask=attn_mask_for_mha, 
        )
        attn_output = attn_output.transpose(0, 1)  # Back to (B, T, C)
        x = residual + self.dropout_module(attn_output)

        # Multi-Scale CAR sublayer
        residual = x
        x_norm = self.norm_car(x)
        x_car = self.multi_scale_car(x_norm) # Expects (B, T, C)
        x = residual + self.dropout_module(x_car)

        # FFN2 sublayer (with half-step residual)
        residual = x
        x_norm = self.norm_ffn2(x)
        x_ffn2 = self.ffn2(x_norm)
        x = residual + 0.5 * self.dropout_module(x_ffn2)

        return x


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        attention_heads: int,
        layers: int,
        dropout: float,
        attention_dropout: float,
        activation_dropout: float,
        activation_fn: str = "relu",
        layerdrop: float = 0.0,
        # CAR module params to pass to layers
        car_compress_dim: int = 256,
        car_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7]),
        car_dropout: float = 0.1,
        max_relative_positions: int = 128, # New parameter
        # LayerNorm options
        layer_norm_first: bool = False, # This implies a final LN after layers if True
        layer_norm_eps: float = 1e-5,
        # Positional encoding (RPE is handled inside layer, this is for other types if needed)
        # max_source_positions: int = 1024, # Example, might not be needed if RPE is fully self-contained
    ):
        super().__init__()
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.encoder_layerdrop = layerdrop
        self.embed_dim = embed_dim

        self.layer_norm_first = layer_norm_first # Pre-LN is done in ConformerEncoderLayer

        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(
                ConformerEncoderLayer(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=ffn_embed_dim,
                    attention_heads=attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    car_compress_dim=car_compress_dim,
                    car_kernel_sizes=car_kernel_sizes,
                    car_dropout=car_dropout,
                    max_relative_positions=max_relative_positions, # Pass to layer
                    layer_norm_eps=layer_norm_eps,
                )
            )
        
        # Final LayerNorm if layer_norm_first is true (meaning layers are Pre-LN, so one final LN)
        # However, Conformer layers are Pre-LN internally. So this might be redundant or for specific model designs.
        # For wav2vec2 TransformerEncoder, if layer_norm_first, a final LN is applied. Let's keep it.
        self.final_layer_norm = None
        if self.layer_norm_first: # Consistent with Fairseq TransformerEncoder
            self.final_layer_norm = LayerNorm(self.embed_dim, eps=layer_norm_eps)
        
        # TODO: Initialize relative positional encoding mechanism here if it's global
        # or ensure each layer handles it if it's layer-specific.
        # For now, assuming RPE bias is generated outside and passed or handled by a modified MHA.

    def forward(
        self,
        x: torch.Tensor, # Expected shape (B, T, C)
        padding_mask: Optional[torch.Tensor] = None, # (B, T), True for pad
        layer: Optional[int] = None, # 1-based index to extract specific layer output
        # rel_pos_bias: Optional[torch.Tensor] = None # To be passed to each layer
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        
        x = self.dropout_module(x)

        # padding_mask is (B, T), True where padded.
        # MultiheadAttention key_padding_mask expects (B, T), True where padded. This is correct.

        # TODO: Implement RPE bias generation if needed at this level.
        # For now, assuming attn_mask in layer.forward() will handle RPE.
        # This is now implemented by passing rel_pos_bias to layer_module.self_attn

        layer_outputs = []
        for i, layer_module in enumerate(self.layers):
            if self.encoder_layerdrop > 0.0 and self.training and (torch.rand(1) < self.encoder_layerdrop):
                if len(layer_outputs) > 0 : # Ensure we have a previous output to pass
                    layer_outputs.append(layer_outputs[-1])
                else: # If first layer is dropped, pass original x
                    layer_outputs.append(x)
                continue

            # RPE bias would be passed as attn_mask to layer_module.forward
            # For now, it is None. -> This is now handled internally by ConformerEncoderLayer
            x = layer_module(x, encoder_padding_mask=padding_mask, attn_mask=None) 
            if layer is not None and (i + 1) == layer:
                layer_outputs.append(x) # Store only the requested layer
                # If only one layer output is needed, we can potentially break early
                # but fairseq's TransformerEncoder collects all up to 'layer'
                # For simplicity, if 'layer' is specified, we return only that layer's output as 'x'
                # and layer_outputs will contain just that.
                # However, standard behavior is to run all layers and pick one if needed.
                # Let's adjust to store all outputs if needed, or just the final one.
            
            # For features_only=True in AVHubertModel, it might need intermediate layer outputs.
            # For now, let's always store them.
            layer_outputs.append(x)


        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)
        
        # If a specific layer was requested, x should be that layer's output.
        # The AVHubertModel's extract_features and forward(features_only=True) might use this.
        if layer is not None:
            if 0 < layer <= len(layer_outputs):
                x = layer_outputs[layer -1] # Adjust to 0-indexed
            else: # Should not happen if layer is valid
                pass 
        
        # The original TransformerEncoder returns (x, layer_results)
        # x is the output of the last layer (or specified layer if 'layer' is not None, after final LN if any)
        # layer_results contains hidden states of all layers.
        return x, layer_outputs 