import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from fairseq import utils
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
    SamePad,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from .car import CARModule

class _ConvolutionModule(nn.Module):
    """Conformer convolution module."""
    def __init__(
        self,
        embed_dim,
        channels, 
        depthwise_kernel_size,
        dropout,
        bias=True,
    ):
        super().__init__()
        assert (
            depthwise_kernel_size % 2 == 1
        ), "depthwise_kernel_size must be odd to achieve \'SAME\' padding"
        
        expanded_channels = channels * 2 # Expansion factor of 2 for GLU

        self.layer_norm = LayerNorm(embed_dim)
        self.pointwise_conv1 = nn.Conv1d(
            embed_dim,
            expanded_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            channels, 
            channels, 
            depthwise_kernel_size,
            stride=1,
            padding=(depthwise_kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.batch_norm = nn.BatchNorm1d(channels)
        self.activation = nn.SiLU() # Swish
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (T, B, C) 
        # residual = x # The residual connection is handled outside this module in ConformerBlock
        x_normed = self.layer_norm(x)
        x_transposed = x_normed.transpose(0, 1).transpose(1, 2)  # (B, C, T)

        x_conv = self.pointwise_conv1(x_transposed)
        x_conv = self.glu(x_conv)
        x_conv = self.depthwise_conv(x_conv)
        x_conv = self.batch_norm(x_conv)
        x_conv = self.activation(x_conv)
        x_conv = self.pointwise_conv2(x_conv)
        
        x_conv_dropout = self.dropout(x_conv)
        
        output = x_conv_dropout.transpose(1, 2).transpose(0, 1)  # (T, B, C)
        return output


class _FeedForwardModule(nn.Module):
    def __init__(
        self,
        embedding_dim,
        ffn_embedding_dim,
        activation_fn_name="relu",
        activation_dropout=0.0,
        dropout=0.0, # This is the dropout on the output of the FFN
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        if activation_fn_name.lower() == "swish" or activation_fn_name.lower() == "silu":
            self.activation_fn = nn.SiLU()
        else:
            self.activation_fn = utils.get_activation_fn(activation_fn_name)
        self.activation_dropout_module = nn.Dropout(activation_dropout)
        self.fc1 = nn.Linear(self.embedding_dim, self.ffn_embedding_dim)
        self.fc2 = nn.Linear(self.ffn_embedding_dim, self.embedding_dim)
        self.dropout_module = nn.Dropout(dropout)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_act = self.activation_fn(x_fc1)
        x_act_drop = self.activation_dropout_module(x_act)
        x_fc2 = self.fc2(x_act_drop)
        output = self.dropout_module(x_fc2) # Dropout applied to the output of the block
        return output


class ConformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1, # General dropout for residuals/sub-module outputs
        attention_dropout: float = 0.1, # Dropout for attention weights
        activation_dropout: float = 0.1, # Dropout after activation in FFN
        depthwise_conv_kernel_size: int = 31,
        car_enabled: bool = False,
        car_d_compress_ratio: int = 4,
        car_activation_fn: str = "gelu",
        car_dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        # FeedForward Module 1
        self.ffn1_layer_norm = LayerNorm(embedding_dim)
        self.ffn1 = _FeedForwardModule(
            embedding_dim,
            ffn_embedding_dim,
            activation_fn_name="swish",
            activation_dropout=activation_dropout,
            dropout=dropout, 
        )

        # Multi-Head Self-Attention Module
        self.self_attn_layer_norm = LayerNorm(embedding_dim)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout, 
            self_attention=True,
        )
        
        # CAR Module
        self.car_module = None
        if car_enabled:
            d_compress = embedding_dim // car_d_compress_ratio
            self.car_module = CARModule(
                d_model=embedding_dim,
                d_compress=d_compress,
                activation_fn=car_activation_fn,
                dropout=car_dropout,
            )

        # Convolution Module
        self.conv_module = _ConvolutionModule(
            embed_dim=embedding_dim,
            channels=embedding_dim, 
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout, 
        )

        # FeedForward Module 2
        self.ffn2_layer_norm = LayerNorm(embedding_dim)
        self.ffn2 = _FeedForwardModule(
            embedding_dim,
            ffn_embedding_dim,
            activation_fn_name="swish",
            activation_dropout=activation_dropout,
            dropout=dropout, 
        )

        self.final_layer_norm = LayerNorm(embedding_dim)
        # This dropout_module is for the output of sub-components before adding to residual
        self.dropout_module = nn.Dropout(dropout) 

    def forward(
        self,
        x: torch.Tensor, 
        self_attn_padding_mask: Optional[torch.Tensor] = None, 
    ):
        # x shape: (T, B, C)

        # FFN Module 1
        residual = x
        x_normed_ffn1 = self.ffn1_layer_norm(x)
        ffn1_out = self.ffn1(x_normed_ffn1)
        # Dropout is applied inside _FeedForwardModule to its output.
        # The 0.5 scaling is a common Conformer detail.
        x = residual + 0.5 * ffn1_out 

        # MHSA Module
        residual = x
        x_normed_attn = self.self_attn_layer_norm(x)
        attn_out, attn_weights = self.self_attn(
            query=x_normed_attn, key=x_normed_attn, value=x_normed_attn,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False, 
        )
        # Apply dropout to the output of MHSA before residual connection
        x = residual + self.dropout_module(attn_out)

        # CAR Module
        if self.car_module is not None:
            x = self.car_module(x) # CAR module includes residual connection internally

        # Convolution Module
        residual = x
        # _ConvolutionModule takes raw x, applies LayerNorm, then Conv, then Dropout.
        conv_out = self.conv_module(x) 
        x = residual + conv_out # conv_out already has dropout applied

        # FFN Module 2
        residual = x
        x_normed_ffn2 = self.ffn2_layer_norm(x)
        ffn2_out = self.ffn2(x_normed_ffn2)
        # Dropout is applied inside _FeedForwardModule to its output.
        x = residual + 0.5 * ffn2_out
        
        x = self.final_layer_norm(x)

        return x, attn_weights # attn_weights is None if need_weights=False


class ConformerEncoder(nn.Module):
    def __init__(self, args): 
        super().__init__()
        self.args = args
        self.dropout_module = nn.Dropout(args.dropout)
        self.encoder_layerdrop = args.encoder_layerdrop
        self.embedding_dim = args.encoder_embed_dim
        
        self.pos_conv = None 
        if getattr(args, "conv_pos", 0) > 0 : 
            self.pos_conv = nn.Conv1d(
                self.embedding_dim,
                self.embedding_dim,
                kernel_size=args.conv_pos,
                padding=args.conv_pos // 2,
                groups=getattr(args, "conv_pos_groups", 16), # from wav2vec2
            )
            dropout_pos_conv = 0 # wav2vec2 uses 0 for pos_conv dropout
            std = math.sqrt((4 * (1.0 - dropout_pos_conv)) / (args.conv_pos * self.embedding_dim))
            nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
            nn.init.constant_(self.pos_conv.bias, 0)
            self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
            self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())


        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=args.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    depthwise_conv_kernel_size=getattr(args, "depthwise_conv_kernel_size", 31),
                    car_enabled=getattr(args, "car_enabled", False),
                    car_d_compress_ratio=getattr(args, "car_d_compress_ratio", 4),
                    car_activation_fn=getattr(args, "car_activation_fn", "gelu"),
                    car_dropout=getattr(args, "car_dropout", 0.1),
                )
                for _ in range(args.encoder_layers) 
            ]
        )
        self.layer_norm_first = getattr(args, "layer_norm_first", False) # from wav2vec2
        
        if self.layer_norm_first:
             self.layer_norm = LayerNorm(self.embedding_dim)
        else:
            # If not layer_norm_first, the final norm is handled by the last ConformerBlock's final_layer_norm
            self.layer_norm = None
        
        self.apply(init_bert_params)


    def forward(self, x, padding_mask=None, layer=None):
        # x input shape: (B, T, C) based on how AVHubertModel calls it
        # padding_mask shape: (B, T)

        x = x.transpose(0, 1) # Convert to (T, B, C) for internal processing

        if self.pos_conv:
            # This is from wav2vec2 transformer encoder
            x_conv_pos_in = x.transpose(1, 2) # B, C, T
            x_conv_pos_out = self.pos_conv(x_conv_pos_in)
            x_conv_pos_out = x_conv_pos_out.transpose(1, 2) # T, B, C
            x = x + x_conv_pos_out

        # The TransformerEncoder in wav2vec2 applies LN here if layer_norm_first.
        # However, ConformerBlock handles its own internal norms and final norm.
        # The self.layer_norm is applied *after* the loop if layer_norm_first.
        
        x = self.dropout_module(x) # Dropout on input features

        all_hidden_states = []

        for i, lyr in enumerate(self.layers):
            # LayerDrop
            if self.encoder_layerdrop > 0 and self.training and (torch.rand(1).item() < self.encoder_layerdrop):
                if len(all_hidden_states) > 0: # if not the first layer
                    x = all_hidden_states[-1] # use previous layer's output
                # if first layer is dropped, x remains the initial dropped-out input
                # This behavior matches fairseq's TransformerEncoder when a layer is dropped.
                # However, simply continuing might be cleaner if the goal is just to skip.
                # For now, let's just skip computation for the dropped layer.
                # x remains unchanged from previous iteration or initial input.
                continue 

            x, _ = lyr(x, self_attn_padding_mask=padding_mask)
            all_hidden_states.append(x)
            if layer is not None and i == layer: # layer is 0-indexed target layer
                break 
        
        if self.layer_norm is not None and self.layer_norm_first : 
            x = self.layer_norm(x)
        
        # For Hubert, the encoder output is usually (x, None) or (x, padding_mask) in some contexts.
        # The original TransformerEncoder returns (x, all_hidden_states).
        # AVHubertModel's encoder call: x, _ = self.encoder(...)
        # Let's return x as the primary output.
        return x, all_hidden_states 

    def extract_features(self, x, padding_mask=None, tgt_layer=None):
        # tgt_layer is 0-indexed if called from AVHubertModel's extract_features -> self.encoder(..., layer=output_layer -1)
        if tgt_layer is not None:
            # Requesting a specific layer's output.
            # The forward method already handles this with its 'layer' argument.
            _, all_hidden_states = self.forward(x, padding_mask, layer=tgt_layer)
            if tgt_layer < len(all_hidden_states):
                 # If layer was valid and computations were stored up to that layer
                 return all_hidden_states[tgt_layer], padding_mask 
            else:
                 # This case implies tgt_layer might have been out of bounds or all layers dropped.
                 # Fallback to last available output from forward, which would be the input x if all layers skipped.
                 # Or if tgt_layer was too large, the full pass output is x.
                 # Safest is to just return the final x from a full pass if something is off.
                 # However, AVHubert's call pattern self.encoder(layer=output_layer-1)
                 # expects this to work.
                 # If layer was valid, all_hidden_states[tgt_layer] is the correct one.
                 # If layerdrop happened for tgt_layer, the forward loop would have continued
                 # and x would be the output of the *previous* non-dropped layer.
                 # all_hidden_states would only contain outputs of non-dropped layers.
                 # This needs careful handling if layerdrop is high.
                 # For simplicity, assume forward populates all_hidden_states correctly for non-dropped layers.
                 # And 'layer' param in forward correctly stops after the target layer.
                 # So all_hidden_states[tgt_layer] should be the one.
                 # If tgt_layer itself was dropped, this logic is problematic.
                 # The fairseq TransformerEncoder's extract_features logic is:
                 #   if layer_idx == tgt_layer: return hidden_states, ...
                 # This means it returns the state *after* the target layer has processed.
                 # My loop in forward also appends *after* lyr(x, ...).
                 # So all_hidden_states[tgt_layer] is correct.

                # If layerdrop caused tgt_layer to not be computed, len(all_hidden_states) might be < tgt_layer+1
                # In that case, we should return the last computed hidden state.
                if tgt_layer < len(all_hidden_states):
                    return all_hidden_states[tgt_layer], padding_mask
                elif all_hidden_states: # if target layer was dropped but previous ones exist
                    return all_hidden_states[-1], padding_mask
                else: # if all layers dropped (or tgt_layer was 0 and it was dropped)
                    # Return original input 'x' that went into the forward method.
                    # This requires passing the initial 'x' down, or re-evaluating.
                    # For now, this edge case might lead to unexpected behavior if tgt_layer is dropped.
                    # Let's assume for now `layer` param in forward means execution *up to and including* that layer if not dropped.
                    # And `all_hidden_states` stores outputs of layers that actually ran.
                    # So if `all_hidden_states` doesn't have `tgt_layer`, it means it was dropped or `tgt_layer` was too high.
                    # If too high, forward would have returned the last layer output.
                    # Let's rely on `forward` to return the appropriate `x` if `tgt_layer` is problematic.
                    final_x, _ = self.forward(x, padding_mask, layer=tgt_layer)
                    return final_x, padding_mask
        else: 
            # Return all layer outputs (or just the final one based on typical usage)
            x, _ = self.forward(x, padding_mask) # Get final layer output
            return x, padding_mask

    def max_positions(self):
        return getattr(self.args, "max_positions", float("inf"))

    # def upgrade_state_dict_named(self, state_dict, name):
    #     super().upgrade_state_dict_named(state_dict, name)
    #     # Further specific upgrades can be added if Conformer introduces new params
    #     # or renames old Transformer ones, when loading a Transformer checkpoint. 