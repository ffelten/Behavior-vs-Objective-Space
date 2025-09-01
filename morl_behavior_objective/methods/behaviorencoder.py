import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import os
import time
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from torch.distributions import Beta, Dirichlet
from typing import Optional
from torch.utils.data import Dataset

# -------------------------------
# Transformer Approach
# -------------------------------

class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset for state-action trajectory data, including ground-truth labels.
    """
    def __init__(self, states: th.Tensor, actions: th.Tensor, masks: th.Tensor, labels: th.Tensor):
        self.states = states
        self.actions = actions
        self.masks = masks
        self.labels = labels
        assert len(states) == len(actions) == len(masks) == len(labels), "All tensors must have the same length."

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        """Returns a tuple of (states, actions, mask, label,index) for a single trajectory."""
        return self.states[idx], self.actions[idx], self.masks[idx], self.labels[idx], idx

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=True):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, batch_first)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

    def forward(self, src: th.Tensor,
                src_mask: th.Tensor = None,
                src_key_padding_mask: th.Tensor = None,
                attn_bias: th.Tensor = None):
        # attn_bias: [H, T, T] or None
        if attn_bias is not None:
            H, T, _ = attn_bias.shape
            B, L, _ = src.shape
            # expand to [H, B, T, T] then reshape [B*H, T, T]
            bias = attn_bias.unsqueeze(1).expand(H, B, T, T).reshape(H*B, T, T)
            attn_mask = bias
        else:
            attn_mask = src_mask

        attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=attn_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src, attn_weights
    
class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer_params, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(**encoder_layer_params)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(encoder_layer_params['d_model'])

    def forward(self, src: th.Tensor,
                src_mask: th.Tensor = None,
                src_key_padding_mask: th.Tensor = None,
                attn_bias: th.Tensor = None):
        output = src
        attn_weights_list = []
        for layer in self.layers:
            output, attn = layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                attn_bias=attn_bias
            )
            attn_weights_list.append(attn)
        output = self.norm(output)
        return output, attn_weights_list

class FourierFeatureEmbed(nn.Module):
    """
    Maps coords to high-dim Fourier features.
    """
    def __init__(self, in_dims=2, num_bands=64, max_freq=10.0): 
        super().__init__()
        self.num_bands = num_bands
        # Create fixed bands [1, 2, 4, ..., 2^(num_bands−1)] scaled to max_freq
        bands = 2.0 ** th.linspace(0, math.log2(max_freq), num_bands)
        self.register_buffer('bands', bands)  # [num_bands]

    def forward(self, x):
        x_proj = 2 * math.pi * x.unsqueeze(-1) * self.bands  # broadcast
        x_sin = th.sin(x_proj)               # [B*L, 2, num_bands]
        x_cos = th.cos(x_proj)
        # concat along last dim → [B*L, 2 * num_bands]
        return th.cat([x_sin, x_cos], dim=-1).view(x.shape[0], -1)

class ScaledFourierFeatureEmbed(nn.Module):
    """
    Deterministic per-dimension Fourier features with learnable per-dim scale.
    Good when you want higher sensitivity without cranking max_freq too high.
    """
    def __init__(self, in_dims=2, num_bands=32, max_freq=32.0, init_log_scale=0.0):
        super().__init__()
        self.in_dims = in_dims
        self.num_bands = num_bands
        bands = 2.0 ** th.linspace(0, math.log2(max_freq), num_bands)
        self.register_buffer('bands', bands)                 # [num_bands]
        self.log_scale = nn.Parameter(th.full((in_dims,), init_log_scale))  # learnable

    def forward(self, x: th.Tensor):
        # x: [N, in_dims], assume normalized per-dim
        x_scaled = x * self.log_scale.exp()                 # [N, in_dims]
        x_proj = 2 * math.pi * x_scaled.unsqueeze(-1) * self.bands  # [N, in_dims, num_bands]
        return th.cat([th.sin(x_proj), th.cos(x_proj)], dim=-1).reshape(x.shape[0], -1)

class GaussianFourierFeatureEmbed(nn.Module):
    """
    Random Fourier Features (RFF) with a shared projection across dims:
    z(x) = [sin(2π xW), cos(2π xW)], W ~ N(0, sigma^2).
    Output size is 2*m independent of input dimension; mixes dimensions.
    """
    def __init__(self, in_dims: int, m: int = 512, sigma: float = 10.0, learnable: bool = False,
                 sigmamode:str="std", normalize_out:bool = True):
        super().__init__()
        self.in_dims = in_dims
        self.m = m
        self.normalize_out = normalize_out
        if sigmamode == "std":
            W = th.randn(in_dims,m) * sigma
        elif sigmamode == "lengthscale":
            W = th.randn(in_dims, m) / (sigma+1e-8)                   # bandwidth via sigma
        self.W = nn.Parameter(W) if learnable else nn.Parameter(W, requires_grad=False)

    def forward(self, x: th.Tensor):
        # x: [N, in_dims], assume normalized
        proj = 2 * math.pi * (x @ self.W)                 # [N, m]
        z = th.cat([th.sin(proj), th.cos(proj)], dim=-1)
        if self.normalize_out:
            z = z / math.sqrt(self.m)  # optional variance stabilization
        return z  # [N, 2m]

class CoordMLPEncoder(nn.Module):
    """
    Embeds 2-D coords into d_model via Fourier features + MLP + LayerNorm,
    with dropout for augmentation.
    Input coords: [B*L, in_dims]
    Output token embeddings: [B*L, d_model]
    """
    def __init__(self,
                 d_model:    int,
                 num_bands:  int     = 64,
                 max_freq:   float   = 10.0,
                 in_dims:    int     = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.in_dims = in_dims
        # Fourier feature projection (deterministic)
        self.ff = FourierFeatureEmbed(in_dims=self.in_dims,
                                      num_bands=num_bands,
                                      max_freq=max_freq)
        feat_dim = self.in_dims * 2 * num_bands
        
        # MLP + LayerNorm + Dropout
        self.net = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.LayerNorm(d_model)
        )

    def forward(self, coords: th.Tensor) -> th.Tensor:
        """
        coords: [B*L, in_dims]
        returns: [B*L, d_model]
        """
        if coords.shape[-1] != self.in_dims:
            raise ValueError(f"Expected coords with {self.in_dims} dimensions, got {coords.shape[-1]}")
        x = self.ff(coords)    # [B*L, feat_dim]
        x = self.net(x)        # [B*L, d_model]
        return x

class CoordMLPEncoderScaled(nn.Module):
    """
    Deterministic bands + learnable per-dimension scale (higher sensitivity without huge max_freq).
    """
    def __init__(self,
                 d_model: int,
                 in_dims: int = 2,
                 num_bands: int = 32,
                 max_freq: float = 32.0,
                 init_log_scale: float = 0.0,
                 dropout: float = 0.1):
        super().__init__()
        self.in_dims = in_dims
        self.ff = ScaledFourierFeatureEmbed(in_dims=self.in_dims, num_bands=num_bands,
                                            max_freq=max_freq, init_log_scale=init_log_scale)
        feat_dim = self.in_dims * 2 * num_bands
        self.net = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.LayerNorm(d_model)
        )

    def forward(self, coords: th.Tensor) -> th.Tensor:
        if coords.shape[-1] != self.in_dims:
            raise ValueError(f"Expected coords with {self.in_dims} dimensions, got {coords.shape[-1]}")
        x = self.ff(coords)
        return self.net(x)

class CoordMLPEncoderGaussian(nn.Module):
    """
    Random Fourier Features (RFF) using a shared Gaussian projection; mixes dimensions.
    Output feature size is 2*m (independent of in_dims).
    """
    def __init__(self,
                 d_model: int,
                 in_dims: int,
                 m: int = 512,
                 sigma: float = 10.0,
                 learnable_proj: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        self.in_dims = in_dims
        self.rff = GaussianFourierFeatureEmbed(in_dims=self.in_dims, m=m,
                                               sigma=sigma, learnable=learnable_proj)
        feat_dim = 2 * m
        self.net = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.LayerNorm(d_model)
        )

    def forward(self, coords: th.Tensor) -> th.Tensor:
        if coords.shape[-1] != self.in_dims:
            raise ValueError(f"Expected coords with {self.in_dims} dimensions, got {coords.shape[-1]}")
        x = self.rff(coords)
        return self.net(x)

class TemporalConvEncoder(nn.Module):
    """
    Takes per-step embeddings [B, L, D] and learns temporal patterns
    via 1D convolutions over the sequence dimension, with optional dilation.
    Returns refined embeddings [B, L, D].
    """
    def __init__(self,
                 emb_dim: int,
                 hidden_dim: int = 128,
                 kernel_size: int = 3,
                 num_layers: int = 2,
                 use_dilation: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        in_ch = emb_dim
        for i in range(num_layers):
            out_ch = hidden_dim if i < num_layers - 1 else emb_dim
            # compute dilation and padding
            dilation = 2 ** i if use_dilation else 1
            padding = ((kernel_size - 1) // 2) * dilation
            layers.append(
                nn.Conv1d(in_ch, out_ch,
                          kernel_size,
                          padding=padding,
                          dilation=dilation)
            )
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x: [B, L, D] → [B, D, L]
        x = x.transpose(1, 2)
        x = self.net(x)             # [B, D, L]
        x = x.transpose(1, 2)       # [B, L, D]
        return x
    
class GridEncoderDropout3D(nn.Module):
    """
    3D CNN encoder over a small temporal window.
    Input: x of shape [B, T, C, H, W]
    Output: per frame embeddings of shape [B, T, out_dim]
    """
    def __init__(self,
                 in_channels: int = 7,
                 out_dim:     int = 64,
                 hidden_channels: int = 32, # This is the base number of channels for the first conv
                 kernel_size:  tuple = (3, 3, 3), # Default, but you instantiate with (3,3,3)
                 pool_kernel:  tuple = (1, 2, 2),
                 dropout:      float = 0.1):
        super().__init__()
        # conv3d: (in_channels, hidden_channels), temporal+spatial
        # Effective padding for kernel (3,3,3) and dilation (2,1,1) should be (2,1,1)
        # If kernel_size is passed as (3,3,3) during instantiation, these padding/dilation values are fine.
        self.conv1 = nn.Conv3d(in_channels, hidden_channels, kernel_size, padding=(2,1,1),dilation=(2,1,1))
        self.bn1   = nn.BatchNorm3d(hidden_channels)
        # self.conv2 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding=(2,1,1),dilation=(2,1,1))
        # self.bn2   = nn.BatchNorm3d(hidden_channels)
        # self.conv3 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding=(2,1,1),dilation=(2,1,1))
        # self.bn3   = nn.BatchNorm3d(hidden_channels)

        # down‐sample spatially only
        self.pool  = nn.MaxPool3d(pool_kernel)
        self.dropout = nn.Dropout3d(dropout)

        # project to per‐frame embedding
        self.adaptpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        # Corrected: Input to fc is hidden_channels*4
        self.fc = nn.Linear(hidden_channels, out_dim)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x: [B, T, C, H, W] → [B, C, T, H, W]
        x = x.transpose(1, 2)
        
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))      # Out channels: hidden_channels
        x = self.pool(x)
        x = self.dropout(x)

        # # # Block 2
        # x = F.relu(self.bn2(self.conv2(x)))       # Out channels: hidden_channels
        # x = self.pool(x)
        # x = self.dropout(x)

        # Block 3
        # x = F.relu(self.bn3(self.conv3(x)))       # Out channels: hidden_channels
        # x = self.pool(x)
        # x = self.dropout(x)

        # collapse spatial dims but keep T
        x = self.adaptpool(x)                     # In channels: hidden_channels*4. Out: [B, hidden_channels*4, T, 1, 1]
        x = x.squeeze(-1).squeeze(-1)             # [B, hidden_channels*4, T]

        # transpose to [B, T, hidden_channels*4]
        x = x.transpose(1, 2)

        # project each frame embedding
        B, T_dim, C_dim = x.shape # C_dim is hidden_channels*4
        x_flat = x.reshape(B * T_dim, C_dim)
        x_fc_out = self.fc(x_flat) # fc input is hidden_channels*4
        return x_fc_out.view(B, T_dim, -1) # [B, T, out_dim]


class BehavioralAutoencoderSA_AR(nn.Module):
    """
    An autoregressive autoencoder for state-action trajectories.
    It wraps the encoder and the autoregressive SA decoder, handling the
    teacher-forcing logic for training.
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.with_cls = True
        # The SOS token is the first input to the decoder
        self.sos_token = nn.Parameter(th.randn(1, 1, self.encoder.d_model))
        self.model_type = 'BD_SA_DT'

    @th.no_grad()
    def generate_with_cls(
        self,
        cls: th.Tensor,          # shape [B, D], a chosen cluster embedding
        max_len: int
    ) -> th.Tensor:
        """
        Autoregressively generates interleaved [s0,a0,s1,a1,…] of length max_len,
        conditioned on a fixed CLS embedding.
        Returns a tensor of shape [B, max_len, state_dim] and [B, max_len,] for actions.
        """
        B, D = cls.shape
        device = cls.device

        # 1) Build the decoder memory: repeat cls, then zeros for history placeholder
        #    If you want the decoder to see *only* the CLS, skip history_mem entirely
        history_len = max_len - 1   # or 0 if you want purely from CLS
        cls_token = cls.unsqueeze(1)                                # [B,1,D]
        hist_tokens = th.zeros(B, history_len, D, device=device) # dummy placeholders
        decoder_memory = th.cat([cls_token, hist_tokens], dim=1) # [B, max_len, D]

        # 2) Start with SOS
        generated = self.sos_token.expand(B, 1, D)   # [B,1,D]

        for t in range(max_len):
            tgt_mask = self.decoder.generate_square_subsequent_mask(
                generated.size(1), device=device
            )
            # run one decoder step
            out_states, out_actions = self.decoder(
                tgt_emb=generated,
                memory=decoder_memory[:, :generated.size(1), :],
                tgt_mask=tgt_mask,
                memory_key_padding_mask=None,
                tgt_key_padding_mask=None
            )
            # take last predicted state & action
            next_state = out_states[:, -1:, :]          # [B,1,state_dim]
            next_action_logits = out_actions[:, -1:, :] # [B,1,num_actions]
            if self.decoder.predict_discrete:
                next_action = next_action_logits.argmax(-1, keepdim=True)   # [B,1]
            else:
                # continuous action: just take the real-valued output
                next_action = next_action_logits 

            # embed back into D-dimensional space
            #  (use the same pipeline as _prepare_target_embeddings)
            flat_s = next_state.reshape(B, -1)  # [B, state_dim]
            state_emb = self.encoder.coord_encoder(flat_s).view(B,1,-1)
            state_emb = self.encoder.temporal_encoder(state_emb) if hasattr(self.encoder, 'temporal_encoder') else state_emb

            # action path
            if self.decoder.predict_discrete:
                # discrete
                action_emb = self.encoder.action_encoder_discr(
                    next_action.squeeze(-1)
                ).unsqueeze(1)
            else:
                # continuous
                flat_a = next_action.view(B, -1)
                if flat_a.shape[1] == 0:
                    raise ValueError(f"Trying to encode an empty action tensor: shape {flat_a.shape}")
                action_emb = self.encoder.action_encoder_cont(flat_a).view(B,1,-1)
                action_emb = self.encoder.action_temporal_encoder_cont(action_emb)

            # add timestep + modality
            t_emb = self.encoder.embed_timestep(
                th.full((B,1), t, dtype=th.long, device=device)
            )
            state_emb  = state_emb  + t_emb + self.encoder.state_type_embedding
            action_emb = action_emb + t_emb + self.encoder.action_type_embedding

            # interleave and append
            next_emb = th.stack([state_emb, action_emb], dim=2).view(B, -1, D)
            generated = th.cat([generated, next_emb], dim=1)

        # final split back to states & actions
        gen = generated[:, 1:].view(B, -1, 2, D).permute(0,2,1,3)
        gen_states = self.decoder.state_pred_head(gen[:,0])    # [B,max_len,state_dim]
        if self.decoder.predict_discrete:
            gen_actions = self.decoder.action_pred_head_discrete(gen[:,1])  # [B,max_len,num_actions]
        else:
            gen_actions = self.decoder.action_pred_head_cont(gen[:,1])      # [B,max_len,num_actions]
        return gen_states, gen_actions

    def _prepare_target_embeddings(self, states: th.Tensor, actions: th.Tensor):
        """
        Recreates the interleaved token embeddings from the raw states and actions.
        This logic must mirror the tokenization strategy in the encoder.
        NOTE: This is tightly coupled to the `BehaviorEncoderCLSattnSATyped` implementation.
        """
        B, T, *_ = states.shape
        
        # Embed states (handling grid vs. coord)
        if states.dim() == 5:
            state_emb = self.encoder.input_proj(self.encoder.cnn_encoder(states.view(B * T, *states.shape[2:])).view(B, T, -1))
        else:
            state_emb = self.encoder.coord_encoder(states.view(B * T, -1)).view(B, T, -1)
            state_emb = self.encoder.temporal_encoder(state_emb)
        
        # Embed actions
        if self.decoder.predict_discrete:
            action_emb = self.encoder.action_encoder_discr(actions)
        else:
            flat_a = actions.view(B*T, -1)
            action_emb = self.encoder.action_encoder_cont(flat_a).view(B, T, -1)
            action_emb = self.encoder.action_temporal_encoder_cont(action_emb)

        # Add timestep and modality embeddings, just like in the encoder
        timesteps = th.arange(T, device=states.device).unsqueeze(0).expand(B, T)
        timestep_embeddings = self.encoder.embed_timestep(timesteps)
        state_emb += timestep_embeddings + self.encoder.state_type_embedding
        action_emb += timestep_embeddings + self.encoder.action_type_embedding

        # Interleave to create the target sequence: [s0, a0, s1, a1, ...]
        interleaved_emb = th.stack([state_emb, action_emb], dim=2).view(B, 2 * T, self.encoder.d_model)
        return interleaved_emb

    def forward(
        self,
        states: th.Tensor,                  # [B, T, state_dim]
        actions: th.Tensor,                 # [B, T]  (discrete)
        src_key_padding_mask: Optional[th.Tensor] = None
    ):
        B, T, _ = states.shape
        D = self.encoder.d_model

        # === 1) ENCODER PASS ===
        # This will apply: coord_encoder → temporal_encoder → interleaving → CLS + transformer
        full_mem, _, _, _, cls_emb, _ = self.encoder(states, actions, src_key_padding_mask=src_key_padding_mask)
        # full_mem is [B, 1 + 2T, D] with slots [CLS, s0,a0,s1,a1,…]

        # === 2) PREPARE DECODER MEMORY ===
        # We want the decoder to *cross*-attend to [CLS, s0,a0,…] exactly as-is:
        decoder_memory = full_mem
        # Masking: never mask CLS (pos 0); mask padded history
        if src_key_padding_mask is not None:
            cls_mask = th.zeros(B,1, dtype=th.bool, device=states.device)
            hist_mask = src_key_padding_mask.unsqueeze(-1).expand(-1,-1,2).reshape(B,2*T)
            memory_key_padding_mask = th.cat([cls_mask, hist_mask], dim=1)  # [B,1+2T]
        else:
            memory_key_padding_mask = None

        # === 3) PREPARE DECODER INPUTS (TEACHER FORCING) ===
        # Recreate the *interleaved* target embeddings (s0,a0,s1,a1,…)
        # (this *does not* include CLS)
        tgt_emb = self._prepare_target_embeddings(states, actions)  # [B, 2T, D]

        # Prepend SOS instead of CLS
        sos = self.sos_token.expand(B, -1, -1)                      # [B,1,D]
        decoder_input_emb = th.cat([sos, tgt_emb], dim=1)       # [B,1+2T,D]

        # Build causal mask and padding mask for decoder self-attn
        seq_len = 1 + 2*T
        tgt_mask = self.decoder.generate_square_subsequent_mask(seq_len, device=states.device)
        tgt_key_padding_mask = memory_key_padding_mask  # identical: first pos unmasked, then pad where needed

        # === 4) DECODER PASS ===
        states_rec, actions_rec_logits = self.decoder(
            tgt_emb=decoder_input_emb,               # [B,1+2T,D]
            memory=decoder_memory,                   # [B,1+2T,D]
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        return cls_emb,decoder_memory, states_rec, actions_rec_logits

class BehaviorTransformerDecoderSA_AR(nn.Module):
    """
    An autoregressive decoder for state-action (SA) trajectories.
    It takes an interleaved sequence of state-action embeddings and reconstructs them step-by-step.
    """
    def __init__(self, emb_dim: int, nhead: int, d_hid: int, nlayers: int,
                 state_dim: int, num_actions: int, dropout: float = 0.1, max_len: int = 100,discrete_actions:bool = False):
        super().__init__()
        self.d_model = emb_dim
        
        # Positional encoding for the interleaved target sequence (s0, a0, s1, a1, ...)
        self.pos_encoder = PositionalEncoding(emb_dim, dropout, max_len=max_len * 2 + 1)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=nhead, dim_feedforward=d_hid,
            dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=nlayers)
        self.predict_discrete = discrete_actions
        # Separate heads for state (continuous) and action (discrete) prediction
        self.state_pred_head = nn.Linear(self.d_model, state_dim)
        self.action_pred_head_discrete = nn.Linear(emb_dim, num_actions)
        self.action_pred_head_cont    = nn.Sequential(
            nn.Linear(emb_dim, num_actions),
            nn.Tanh(),
        )
        self.model_type = 'BD_SA_DT'
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt_emb: th.Tensor, memory: th.Tensor, tgt_mask: Optional[th.Tensor] = None,
                tgt_key_padding_mask: Optional[th.Tensor] = None,
                memory_key_padding_mask: Optional[th.Tensor] = None):
        """
        Args:
            tgt_emb (Tensor): The embedded target sequence. Shape: [B, T_tgt, D]
            memory (Tensor): The output from the encoder. Shape: [B, T_src, D]
        """
        # Apply positional encoding
        tgt_with_pe = self.pos_encoder(tgt_emb)
        
        # Pass through transformer decoder
        output = self.transformer_decoder(
            tgt=tgt_with_pe,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        if self.training:
            # During training, we remove the last token to match the target sequence length
            # (because we predict one token at a time)
            # This is not needed during inference, where we generate one token at a time.
            output = output[:, :-1, :]
            # The output is an interleaved sequence of embeddings.
            # Separate them to apply the correct prediction head.
            state_out_emb = output[:, 0::2, :]  # State embeddings are at even indices
            action_out_emb = output[:, 1::2, :] # Action embeddings are at odd indices
        else:
            state_out_emb = output
            action_out_emb = output

        # Project to get reconstructions
        states_rec = self.state_pred_head(state_out_emb)
        if self.predict_discrete:
            actions_rec_logits = self.action_pred_head_discrete(action_out_emb)
        else:
            actions_rec_logits = self.action_pred_head_cont(action_out_emb)
        
        return states_rec, actions_rec_logits

    def generate_square_subsequent_mask(self, sz: int, device: th.device) -> th.Tensor:
        """Generates a causal mask for autoregressive decoding."""
        return nn.Transformer.generate_square_subsequent_mask(sz, device=device)

class BehaviorEncoderCLSattnSATyped(nn.Module):
    """
    A Behavior Encoder using Decision Transformer-style tokenization and positional embeddings.
    It processes sequences of states and actions by interleaving them and assigning a learned 
    embedding to each timestep and modality (state/action).
    """
    def __init__(self, input_channels: int, cnn_output_dim: int, steps: int, nhead: int, d_hid: int, emb_dim: int,
                 num_actions: int,
                 nlayers: int = 6, dropout: float = 0.1, max_len: int = 100, input_coord_dims: int = 2, max_freq: float = 2.0,
                 # NEW: choose coord encoders for states and continuous actions
                 coord_state_kind: str = 'gaussian',           # {'det','scaled','gaussian'}
                 coord_action_kind: str = 'gaussian',          # {'det','scaled','gaussian'}
                 # NEW: per-kind hyperparams (kept simple; num_bands uses emb_dim by default)
                 scaled_init_log_scale_state: float = 0.0,
                 scaled_init_log_scale_action: float = 0.0,
                 gaussian_m_state: int = 1024,
                 gaussian_sigma_state: float = 10.0,
                 gaussian_m_action: int = 512,
                 gaussian_sigma_action: float = 10.0):
        super().__init__()
        self.d_model = emb_dim
        self.cls_token = nn.Parameter(th.randn(1, 1, emb_dim))
        self.model_type = 'BE_SA_DT'
        self.input_coord_dims = input_coord_dims
        
        # Timestep embedding, as in Decision Transformer. Replaces standard PEs.
        self.embed_timestep = nn.Embedding(max_len, emb_dim)

        # Encoders for states (handles both grid and coordinate-based envs)
        self.cnn_encoder = GridEncoderDropout3D(in_channels=input_channels, out_dim=cnn_output_dim, hidden_channels=32, kernel_size=(3, 3, 3), pool_kernel=(1, 2, 2), dropout=dropout)

        # State coord encoder (selectable)
        self.coord_encoder = self.make_coord_mlp_encoder(
            kind=coord_state_kind,
            d_model=emb_dim,
            in_dims=self.input_coord_dims,
            dropout=dropout,
            num_bands=emb_dim,                 # keep your default width
            max_freq=max_freq,                 # reuse provided max_freq
            init_log_scale=scaled_init_log_scale_state,
            m=gaussian_m_state,
            sigma=gaussian_sigma_state,
            learnable_proj=True
        )
        self.temporal_encoder  = TemporalConvEncoder(emb_dim=emb_dim,hidden_dim=d_hid,kernel_size=7,num_layers=1,dropout=dropout,use_dilation=False)

        # Encoder for actions
        self.action_encoder_discr = nn.Embedding(num_actions, emb_dim)

        # Continuous action encoder (selectable)
        self.action_encoder_cont = self.make_coord_mlp_encoder(
            kind=coord_action_kind,
            d_model=emb_dim,
            in_dims=num_actions,
            dropout=dropout,
            num_bands=emb_dim,
            max_freq=max_freq,
            init_log_scale=scaled_init_log_scale_action,
            m=gaussian_m_action,
            sigma=gaussian_sigma_action,
            learnable_proj=True
        )
        self.action_temporal_encoder_cont  = TemporalConvEncoder(emb_dim=emb_dim,hidden_dim=d_hid,kernel_size=7,num_layers=1,dropout=dropout,use_dilation=False)

        # Modality embeddings to differentiate states and actions
        self.state_type_embedding = nn.Parameter(th.randn(1, 1, emb_dim))
        self.action_type_embedding = nn.Parameter(th.randn(1, 1, emb_dim))

        self.input_proj = nn.Linear(cnn_output_dim, emb_dim) if cnn_output_dim != emb_dim else nn.Identity()

        encoder_layer_params = {
            'd_model': self.d_model,
            'nhead': nhead,
            'dim_feedforward': d_hid,
            'dropout': dropout,
            'activation': 'relu',
            'batch_first': True
        }
        self.transformer_encoder = CustomTransformerEncoder(encoder_layer_params, nlayers)
        self._dropout_p = dropout
        self._register_dropout_modules()
        # self.pooling = MHAPooling(emb_dim, num_heads=nhead)
        self.init_weights()
        print(f"Input treated with {coord_state_kind} fourier feature encoder for states and {coord_action_kind} fourier feature encoder for actions.")

    @staticmethod
    def make_coord_mlp_encoder(kind: str,
                            d_model: int,
                            in_dims: int,
                            dropout: float = 0.1,
                            # deterministic/scaled
                            num_bands: int = 64,
                            max_freq: float = 10.0,
                            init_log_scale: float = 0.0,
                            # gaussian
                            m: int = 512,
                            sigma: float = 10.0,
                            learnable_proj: bool = False) -> nn.Module:
        """
        kind ∈ {'det','scaled','gaussian'}
        """
        kind = kind.lower()
        if kind == 'det':
            return CoordMLPEncoder(d_model, in_dims=in_dims, num_bands=num_bands, max_freq=max_freq, dropout=dropout)
        if kind == 'scaled':
            return CoordMLPEncoderScaled(d_model, in_dims=in_dims, num_bands=num_bands, max_freq=max_freq,
                                        init_log_scale=init_log_scale, dropout=dropout)
        if kind == 'gaussian':
            return CoordMLPEncoderGaussian(d_model, in_dims=in_dims, m=m, sigma=sigma,
                                        learnable_proj=learnable_proj, dropout=dropout)
        raise ValueError(f"Unknown coord encoder kind: {kind}")
    
    def _register_dropout_modules(self):
        self._dropouts = []
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                self._dropouts.append(m)

    def set_dropout(self, p: float):
        for dr in self._dropouts:
            dr.p = p
        self._dropout_p = p

    def init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, states: th.Tensor, actions: th.Tensor, src_key_padding_mask: Optional[th.Tensor] = None) -> tuple:
        B, T, *_ = states.shape
    
        # 1. Encode states based on their dimension (grid vs. coord)
        if states.dim() == 5:
            state_emb = self.cnn_encoder(states.view(B * T, *states.shape[2:])).view(B, T, -1)
        elif states.dim() == 3:
            state_emb = self.coord_encoder(states.view(B * T, -1)).view(B, T, -1)
            state_emb = self.temporal_encoder(state_emb)
        else:
            raise ValueError(f"Unsupported state dimension: {states.dim()}")
        state_emb = self.input_proj(state_emb)

        # 2. Encode actions
        if actions[0][0].dtype == th.int64:
            # Discrete actions
            action_emb = self.action_encoder_discr(actions)
        else:
            # Continuous actions
            flat_act = actions.view(B*T, -1)
            action_emb = self.action_encoder_cont(flat_act).view(B, T, self.d_model)
            action_emb = self.action_temporal_encoder_cont(action_emb)

        # 3. Add timestep and modality embeddings
        timesteps = th.arange(T, device=states.device).expand(B, T)
        timestep_embeddings = self.embed_timestep(timesteps)

        state_emb = state_emb + timestep_embeddings + self.state_type_embedding
        action_emb = action_emb + timestep_embeddings + self.action_type_embedding

        # 4. Interleave state and action embeddings to form the input sequence
        # -> [s0, a0, s1, a1, ...]
        interleaved_emb = th.stack([state_emb, action_emb], dim=2).view(B, 2 * T, self.d_model)

        # 5. Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        emb = th.cat([cls, interleaved_emb], dim=1)

        # 6. Adjust padding mask for the new interleaved sequence
        final_padding_mask = None
        if src_key_padding_mask is not None:
            cls_mask = th.zeros(B, 1, dtype=th.bool, device=src_key_padding_mask.device)
            interleaved_mask = src_key_padding_mask.unsqueeze(-1).expand(-1, -1, 2).reshape(B, 2 * T)
            final_padding_mask = th.cat([cls_mask, interleaved_mask], dim=1)

        # 7. Transformer pass

        
        transformer_out, attn_list = self.transformer_encoder(
            emb,
            src_mask=None,
            src_key_padding_mask=final_padding_mask
        )


        
        # 8. Normalize outputs and get trajectory summary
        norm = transformer_out.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        normalized = transformer_out / norm
        
        # pooled, pool_weights = self.pooling(transformer_out, final_padding_mask)
        cls_emb = normalized[:, 0, :]

        stacked_attns = th.stack(attn_list)
        cls_attn = stacked_attns[:, :, :, 0, :].mean(dim=(0, 2))
        attn_list_agg = stacked_attns.sum(dim=0).sum(dim=1)

        return normalized, attn_list_agg, _, _, cls_emb, cls_attn

class Discriminator(nn.Module):
    """A simple MLP to distinguish between positive and negative pairs."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, 1)
        )
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)

class DeepInfoMaxLoss(nn.Module):
    """
    Deep InfoMax loss using a discriminator.
    Maximizes MI between a global summary vector and its local feature vectors.
    This version uses the Jensen-Shannon Divergence (JSD) estimator.
    """
    def __init__(self, encoder_dim: int):
        super().__init__()
        # The discriminator takes a concatenated global and local vector
        self.discriminator = Discriminator(input_dim=encoder_dim * 2)
        self.softplus = nn.Softplus()

    def forward(self, global_emb: th.Tensor, local_embs: th.Tensor, key_padding_mask: th.Tensor):
        """
        Args:
            global_emb (Tensor): The [CLS] embedding. Shape: [B, D_emb]
            local_embs (Tensor): The full sequence of token embeddings. Shape: [B, T, D_emb]
            key_padding_mask (Tensor): Mask for padded tokens. Shape: [B, T], True if padded.
        """
        B, T, D = local_embs.shape
        # Ensure the key_padding_mask matches the sequence length of local_embs
        if key_padding_mask.size(1) != local_embs.size(1):
            raise ValueError("Mismatch between mask and interleaved sequence length.")

        # --- Positive Pairs (Global summary with its own local features) ---
        global_emb_expanded = global_emb.unsqueeze(1).expand(-1, T, -1)
        positive_pairs = th.cat((global_emb_expanded, local_embs), dim=-1)  # [B, T, 2*D]
        positive_scores = self.discriminator(positive_pairs).squeeze(-1)  # [B, T]

        # --- Negative Pairs (Global summary with shuffled local features) ---
        # Shuffle local embeddings across the batch dimension to create negative samples
        shuffled_local_embs = th.cat((local_embs[1:], local_embs[0:1]), dim=0)
        negative_pairs = th.cat((global_emb_expanded, shuffled_local_embs), dim=-1)
        negative_scores = self.discriminator(negative_pairs).squeeze(-1) # [B, T]

        # --- Calculate JSD-based Loss ---
        # Loss for positive pairs: E[softplus(-D(positive_pairs))]
        # We want the discriminator score to be high, so -score should be low, and softplus(-score) should be low.
        loss_pos = self.softplus(-positive_scores)

        # Loss for negative pairs: E[softplus(D(negative_pairs))]
        # We want the discriminator score to be low, so softplus(score) should be low.
        loss_neg = self.softplus(negative_scores)

        # Combine and mask the loss
        # We only compute the loss over non-padded tokens.
        loss_unmasked = loss_pos + loss_neg
        
        # Create a mask for valid (non-padded) tokens
        valid_token_mask = ~key_padding_mask

        # Apply the mask and compute the mean loss over valid tokens
        masked_loss = loss_unmasked[valid_token_mask].mean()

        return masked_loss

class InstanceLoss(nn.Module):
    def __init__(self, temperature, device):
        super(InstanceLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.loss_type = "IL"

        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, current_batch_size):
        N = 2 * current_batch_size
        mask = th.ones((N, N), dtype=th.bool, device=self.device)
        mask = mask.fill_diagonal_(False)  # Remove self-comparisons
        for i in range(current_batch_size):
            # Remove positive pairs (z_i[k] vs z_j[k] and z_j[k] vs z_i[k])
            mask[i, current_batch_size + i] = False
            mask[current_batch_size + i, i] = False
        return mask

    def forward(self, z_i, z_j):
        current_batch_size = z_i.shape[0]

        if current_batch_size == 0:
            # Or handle as appropriate, e.g., if z_j.shape[0] is also 0
            return th.tensor(0.0, device=self.device, requires_grad=True)
        
        N = 2 * current_batch_size
        z = th.cat((z_i, z_j), dim=0)

        sim = th.matmul(z, z.T) / self.temperature
        sim_i_j = th.diag(sim, current_batch_size)
        sim_j_i = th.diag(sim, -current_batch_size)

        positive_samples = th.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_mask = self.mask_correlated_samples(current_batch_size)
        negative_samples = sim[negative_mask].reshape(N, -1)

        labels = th.zeros(N).to(positive_samples.device).long()
        logits = th.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


def create_model_BECwASATyped(input_channels, cnn_output_dim, steps, nhead, d_hid, emb_dim, num_actions, nlayers, input_coord_dims, dropout=0.1, gaussian_m_state=0, gaussian_m_action=0, gaussian_sigma_state=0, gaussian_sigma_action=0, pe_type='rope'):
    """
    Factory function for the BehaviorEncoderCLSattnSATyped model.
    """
    model = BehaviorEncoderCLSattnSATyped(
        input_channels=input_channels,
        cnn_output_dim=cnn_output_dim,
        max_len=steps,
        steps=steps,
        nhead=nhead,
        d_hid=d_hid,
        emb_dim=emb_dim,
        num_actions=num_actions,
        nlayers=nlayers,
        dropout=dropout,
        # pe_type=pe_type,
        input_coord_dims=input_coord_dims,
        gaussian_m_state=gaussian_m_state,
        gaussian_m_action=gaussian_m_action,
        gaussian_sigma_state=gaussian_sigma_state,
        gaussian_sigma_action=gaussian_sigma_action
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

