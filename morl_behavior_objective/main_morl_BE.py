import numpy as np
import math
import random
import time
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D  # ensure this import is present
import argparse
import heapq
import seaborn as sns
import copy
import pickle
import sys
from imitation.data.types import TrajectoryWithRew # type: ignore[import]
import os
from tqdm import tqdm
import umap #type: ignore[import]
import pandas as pd
import torch as th
import torch.nn.functional as F # type: ignore[import]
import torch.utils.data as data     # type: ignore[import]
import torch.nn as nn # type: ignore[import]
from collections import Counter
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

from sklearn.metrics import (silhouette_score, davies_bouldin_score,
    calinski_harabasz_score, adjusted_rand_score, precision_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    recall_score,f1_score,
    roc_auc_score, average_precision_score,
    roc_curve, auc, precision_recall_curve
)
from sklearn.metrics import silhouette_samples
from sklearn.model_selection import KFold # type: ignore[import]
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler,
    QuantileTransformer, PowerTransformer, Normalizer
)
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
import sklearn.calibration as skcal
from sklearn.cluster import KMeans,SpectralClustering,HDBSCAN, AgglomerativeClustering, MeanShift,AffinityPropagation,Birch,estimate_bandwidth # type: ignore[import]
import hdbscan # type: ignore[import]
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
from sklearn.decomposition import PCA
from itertools import combinations
import warnings
from imitation.data.types import TrajectoryWithRew,Trajectory # To structure trajectory data #type: ignore[import]
from imitation.data import serialize #type: ignore[import]
from stable_baselines3 import PPO, DQN
from imitation.algorithms.adversarial.airl import AIRL #type: ignore[import]
from imitation.algorithms.adversarial.gail import GAIL #type: ignore[import]
from imitation.algorithms import sqil #type: ignore[import]
from imitation.rewards.reward_nets import BasicShapedRewardNet #type: ignore[import]
from imitation.util.networks import RunningNorm #type: ignore[import]
from imitation.policies.serialize import load_policy,policy_registry #type: ignore[import]
from stable_baselines3.common.type_aliases import Schedule #type: ignore[import]
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.sac import SAC
from stable_baselines3.common.env_util import make_vec_env
from typing import Optional
from torch.utils.data import Dataset
warnings.filterwarnings("ignore")

# from methods.mlirl import *
# from methods.behaviorencoder import *
# from methods.behaviorencoder_utils import *
# from my_envs.gridworlds import *
# from my_envs.gridworlds_utils import *
# from f_e_sensitivity import *
# from my_envs.gym_gridworlds import *
# from methods.my_highway_env import *
# from asselinecheck import *

# from essinfogail.envs import *

## CLASSES

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

## UTILS

def prepare_sa_trajectories(
    env_id: str,
    trajectories: list[Trajectory],
    true_labels: np.ndarray,
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """
    Prepares state-action trajectories for a typed transformer model.
    This function handles padding for variable-length trajectories and separates
    states, actions, and labels into distinct tensors. It also masks out any steps
    that occur after the first time a known goal state is reached. For environments
    like Traj2d with variable lengths, it pads shorter trajectories by repeating the
    last valid state and action.

    Args:
        env_id (str): The ID of the environment, used to identify the goal state.
        trajectories (list): A list of imitation.Trajectory objects.
        true_labels (np.ndarray): An array of ground-truth integer labels for each trajectory.
        max_len (Optional[int]): The maximum length to pad/truncate to. If None,
                                 it's determined by the longest trajectory.

    Returns:
        A tuple containing:
        - all_states (th.Tensor): Padded state sequences. Shape: [N, T, ...state_dims].
        - all_actions (th.Tensor): Padded action sequences. Shape: [N, T].
        - all_masks (th.Tensor): Boolean mask, True for padded/post-goal elements. Shape: [N, T].
        - all_labels (th.Tensor): Ground-truth labels. Shape: [N].
    """
    if not trajectories:
        return th.empty(0), th.empty(0), th.empty(0), th.empty(0)

    max_len = max(len(traj.obs) for traj in trajectories)
    min_len = min(len(traj.obs) for traj in trajectories)

    # Define ending_state based on env_id
    ending_state = None
    if env_id == "PuddleWorld-v0":
        ending_state = np.array([2, 4])
        # ending_state = np.array([3, 6])
    elif env_id == "TwoLakesFishing-v0":
        ending_state = np.array([6, 3, 1])
    elif env_id == "ConditionalAssemblyLine-v0":
        ending_state = np.array([3, 14, 0])

    # Infer shapes and dtypes from the first trajectory
    first_traj = trajectories[0]
    state_shape = np.array(first_traj.obs).shape[1:]
    state_dtype = th.float32
    action_dtype = th.int64

    num_trajs = len(trajectories)
    all_states = th.zeros((num_trajs, max_len, *state_shape), dtype=state_dtype)
    # all_actions = th.zeros((num_trajs, max_len), dtype=action_dtype)
    first_act = np.array(trajectories[0].acts)
    if first_act.ndim == 0 or first_act.ndim == 1:
        # discrete: shape (L,) ints
        discrete = True
        action_dim = 1
        all_actions = th.zeros((num_trajs, max_len), dtype=th.long)
    else:
        # continuous: shape (L, A) floats
        discrete = False
        action_dim = first_act.shape[1]
        all_actions = th.zeros((num_trajs, max_len, action_dim),
                                  dtype=th.float32)
    all_masks = th.ones((num_trajs, max_len), dtype=th.bool)  # True means padded/masked

    for i, traj in enumerate(tqdm(trajectories, desc="Preparing State-Action Trajectories")):
        obs_np = np.array(traj.obs)
        acts_np = np.array(traj.acts)
        original_seq_len = len(obs_np)

        # Determine the valid length of the trajectory (before padding)
        valid_len = original_seq_len
        if ending_state is not None:
            matches = np.where(np.all(obs_np == ending_state, axis=1))[0]
            if len(matches) > 0:
                goal_index = matches[0]
                valid_len = goal_index + 1
        
        # The effective length is the part of the trajectory we will use, capped by max_len
        effective_len = min(valid_len, max_len)
        if effective_len == 0:
            continue

        # --- Fill valid part of the tensors ---
        
        # States
        all_states[i, :effective_len] = th.tensor(obs_np[:effective_len], dtype=state_dtype)

        # Actions: We need an action for each state.
        # The trajectory provides len(obs)-1 actions. We repeat the last action for the last state.
        # if len(acts_np) > 0:
        #     num_acts_to_copy = min(len(acts_np), effective_len - 1)
        #     if num_acts_to_copy > 0:
        #         actions_to_copy = th.tensor(acts_np[:num_acts_to_copy], dtype=action_dtype)
        #         all_actions[i, :num_acts_to_copy] = actions_to_copy
        #         last_action = actions_to_copy[-1]
        #         all_actions[i, num_acts_to_copy:effective_len] = last_action
        #     else: # effective_len is 1, so we take 0 actions from trajectory
        #         all_actions[i, :effective_len] = th.tensor(acts_np[0], dtype=action_dtype)
        # else:
        #     # No actions in trajectory, use a default (e.g., 0)
        #     all_actions[i, :effective_len] = 0
        if discrete:
            # previous logic for ints
            if len(acts_np) > 0:
                # same as before, but wrap in a 1-D view
                num_copy = min(len(acts_np), effective_len-1)
                if num_copy > 0:
                    a = th.tensor(acts_np[:num_copy], dtype=th.long)
                    all_actions[i, :num_copy] = a
                    last = a[-1]
                    all_actions[i, num_copy:effective_len] = last
                else:
                    # only one step
                    all_actions[i, :effective_len] = int(acts_np[0])
            else:
                all_actions[i, :effective_len] = 0
        else:
            # continuous
            if acts_np.ndim == 1:
                # flatten scalar per step → make it [L,1]
                acts_np = acts_np.reshape(-1, 1)
            # now acts_np is [L, action_dim]
            num_copy = min(len(acts_np), effective_len-1)
            if num_copy > 0:
                a = th.tensor(acts_np[:num_copy], dtype=th.float32)  # [num_copy, A]
                all_actions[i, :num_copy, :] = a
                last = a[-1]  # [A]
                all_actions[i, num_copy:effective_len, :] = last
            else:
                # only one step
                all_actions[i, :effective_len, :] = th.tensor(acts_np[0], dtype=th.float32)


        # Mask: False for valid steps, True for padded/post-goal steps
        all_masks[i, :effective_len] = False

        # --- Fill padded part of the tensors ---
        if effective_len < max_len:
            last_state_val = all_states[i, effective_len - 1]
            last_action_val = all_actions[i, effective_len - 1]
            
            all_states[i, effective_len:] = last_state_val
            all_actions[i, effective_len:] = last_action_val

    all_labels = th.tensor(true_labels, dtype=th.long)

    return all_states, all_actions, all_masks, all_labels, max_len

def datasets_preparation_sa(
    states: th.Tensor,
    actions: th.Tensor,
    masks: th.Tensor,
    labels: th.Tensor,
    train_size: int,
    val_size: int,
    test_size: int,
    loader_batch: int,
    val_bptt: int,
    test_bptt: int,
    seed: int = 42
) -> tuple:
    """
    Creates train, validation, and test dataloaders from prepared state-action tensors.
    This is the new equivalent of your `datasets_preparation` function.

    Args:
        states, actions, masks, labels: Tensors from prepare_sa_data_from_trajectories.
        train_size, val_size, test_size: Number of samples for each split.
        loader_batch, val_bptt, test_bptt: Batch sizes for the dataloaders.
        seed: Random seed for splitting.

    Returns:
        A tuple matching the original function's output structure for easy integration:
        (full_dataset, train_dataset, val_dataset, test_dataset,
         total_dataloader, train_dataloader, val_dataloader, test_dataloader)
    """
    # Create the full dataset using the new TrajectoryDataset class
    full_dataset = TrajectoryDataset(states, actions, masks, labels)

    # Split the dataset
    generator = th.Generator().manual_seed(seed)
    print((len(full_dataset), train_size, val_size, test_size))
    train_dataset, val_dataset, test_dataset = th.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Create DataLoaders
    total_dataloader = data.DataLoader(full_dataset, batch_size=1, shuffle=True)
    train_dataloader = data.DataLoader(train_dataset, batch_size=loader_batch, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=val_bptt, shuffle=False)
    test_dataloader = data.DataLoader(test_dataset, batch_size=test_bptt, shuffle=False)

    return full_dataset, train_dataset, val_dataset, test_dataset, total_dataloader, train_dataloader, val_dataloader, test_dataloader

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

def expand_interleaved_mask(original_mask: th.Tensor) -> th.Tensor:
    """
    Expands a mask for interleaved states and actions.
    
    Args:
        original_mask (Tensor): Original mask of shape [B, T] (e.g., [B, 126]).
    
    Returns:
        Tensor: Expanded mask of shape [B, 2*T] (e.g., [B, 252]).
    """
    B, T = original_mask.shape
    # Repeat each mask element twice (for state and action)
    interleaved_mask = original_mask.unsqueeze(-1).repeat(1, 1, 2).view(B, 2 * T)
    return interleaved_mask

def segment_contrastive_loss(
    full_cls_emb: th.Tensor,
    encoder: nn.Module,
    states: th.Tensor,
    actions: th.Tensor,
    masks: th.Tensor,
    L: int,
    temperature: float = 0.5,
    num_segments: int = 2,
    pairwise_segments:bool = False,
    contrastive_loss_fn : nn.Module = InstanceLoss(0.5,device=th.device("cpu"))
):
    """
    Global-vs-Segment contrastive loss.
    Contrasts the CLS embedding of the full trajectory against the CLS embeddings
    of num_segments random segments of length L from that same trajectory.

    Args:
      full_cls_emb: Tensor [B, D_emb] of pre-computed full trajectory embeddings.
      encoder: The behavior encoder model.
      states, actions, masks: Batch of trajectory data.
      L: segment length.
      temperature: τ in the NT-Xent formula.

    Returns:
      loss: A scalar contrastive loss.
      z_segs: Embeddings of the segments for potential MI loss calculation.
    """
    B, T, _ = states.shape
    device = states.device

    # 1) Calculate valid lengths and sample start indices for two segments
    valid_lengths = (~masks).sum(dim=1)
    max_starts = (valid_lengths - L).clamp(min=0)
    
    # Check if any trajectory in the batch can actually produce a segment
    if max_starts.sum() == 0:
        return th.tensor(0.0, device=device), None, None

    rand = th.rand(B, num_segments, device=device)
    starts = (rand * max_starts.unsqueeze(1).float()).long().clamp(max=max_starts.unsqueeze(1))

    seg_batches = []
    for k in range(num_segments):
        idx_k = starts[:, k]
        seg_states = th.stack([states[b, idx_k[b]:idx_k[b] + L] for b in range(B)], dim=0)
        seg_actions = th.stack([actions[b, idx_k[b]:idx_k[b] + L] for b in range(B)], dim=0)
        seg_masks = th.zeros((B, L), dtype=th.bool, device=device)
        seg_batches.append((seg_states, seg_actions, seg_masks))

    # 3) Encode segments → CLS
    z_full = F.normalize(full_cls_emb, dim=1)
    z_segs = []
    for k in range(num_segments):
        cls_k = encoder(*seg_batches[k])[4]  # [B, D]
        z_segs.append(F.normalize(cls_k, dim=1))

    # 4) Loss: full vs each segment
    loss_full_vs_segs = 0.0
    for z in z_segs:
        loss_full_vs_segs = loss_full_vs_segs + contrastive_loss_fn(z_full, z)
    loss_full_vs_segs = loss_full_vs_segs / float(num_segments)

    # 5) Optional pairwise loss among segments
    loss_pairwise = th.tensor(0.0, device=device)
    if pairwise_segments and num_segments > 1:
        pairs = 0
        for i in range(num_segments):
            for j in range(i + 1, num_segments):
                loss_pairwise = loss_pairwise + contrastive_loss_fn(z_segs[i], z_segs[j])
                pairs += 1
        loss_pairwise = loss_pairwise / float(pairs)

    loss = loss_full_vs_segs + loss_pairwise
    return loss, z_segs

def run_encoder_only_training_sa(
    env_id: str,
    encoder: nn.Module,
    dataloader: data.DataLoader,
    K: int,
    device: th.device,
    epochs_pre: int = 150,
    epochs_formal: int = 0,
    lr: float = 1e-3,
    beta: float = 0.3,    # contrastive
    gamma: float = 1.0,   # InfoMax
    delta: float = 0.0,   # clustering
    seed: int = 42,
):
    """
    Encoder-only training (no decoder, no reconstruction loss).
    Uses: SimCSE-style contrastive on CLS, DeepInfoMax(global-local), optional segment contrastive,
    K-Means init + CDEC-style clustering loss in formal stage.
    Returns: (encoder, None, None, cluster_centroids, None)
    """
    if getattr(encoder, "model_type", "") == "BE_SA_VDT":
        raise ValueError("Encoder-only loop not supported for BE_SA_VDT (variational) encoders. Use alpha=0 in the AE loop instead.")

    # infomax_loss_fn = DeepInfoMaxSymLoss(encoder.d_model).to(device)
    infomax_loss_fn = DeepInfoMaxLoss(encoder.d_model).to(device)
    # infomax_loss_seg_fn = DeepInfoMaxLoss(encoder.d_model).to(device)
    optimizer = th.optim.Adam(
        list(encoder.parameters()) + list(infomax_loss_fn.discriminator.parameters()),
        lr=lr
    )
    # optimizer = th.optim.Adam(
    #     list(encoder.parameters()) + list(infomax_loss_fn.discriminator.parameters()) + list(infomax_loss_seg_fn.discriminator.parameters()),
    #     lr=lr
    # )

    sigma = 0.5 if env_id == "Traj2d" else 0.5 if env_id == "Reacher-v4" else 0.5 if env_id == "Pusherv4" else 0.5 if env_id == "Walker2dv4" else 0.5 if env_id == "Humanoidv4" else 0.5
    tau = 0.1 if env_id == "Traj2d" else 0.3 if env_id == "Reacher-v4" else 0.3 if env_id == "Pusherv4" else 0.0005 if env_id == "Walker2dv4" else 0.0005 if env_id == "Humanoidv4" else 0.3
    contrastive_loss_fn = InstanceLoss(temperature=tau, device=device)

    # ---------------- Stage 1: Pre-Training (no recon) ----------------
    epoch_pbar = tqdm(range(epochs_pre), desc="Pre-training Encoder (no recon)")
    for epoch in epoch_pbar:
        encoder.train(); infomax_loss_fn.discriminator.train()
        total_loss = 0.0

        for batch in dataloader:
            states, actions, masks, _, _ = [b.to(device) for b in batch]
            L_min, L_max = (
                (8, 12) if env_id in ("TwoLakesFishing-v0", "Traj2d")
                else (8, 16) if env_id in ("Reacher-v4","Pusherv4")
                else (8, 16) if env_id in ("Walker2dv4", "Humanoidv4")
                else (8, 16)
            )
            T = states.shape[1]
            current_L_max = min(L_max, T - 1)
            L = th.randint(L_min, current_L_max + 1, (1,)).item() if current_L_max >= L_min else L_min

            optimizer.zero_grad()

            # Two stochastic passes (dropout)
            norm1, _, _, _, cls1, _ = encoder(states, actions, src_key_padding_mask=masks)
            norm2, _, _, _, cls2, _ = encoder(states, actions, src_key_padding_mask=masks)

            # Deep InfoMax on local tokens (exclude CLS @ pos 0)
            local_mask1 = expand_interleaved_mask(masks)
            local_mask2 = expand_interleaved_mask(masks)
            loss_infomax = (
                infomax_loss_fn(cls1, norm1[:, 1:, :], local_mask1) +
                infomax_loss_fn(cls2, norm2[:, 1:, :], local_mask2)
            ) / 2.0

            # Instance contrastive on CLS
            loss_contrastive = contrastive_loss_fn(cls1, cls2)

            # Segment contrastive (optional)
            loss_seg_1 = segment_contrastive_loss(cls1, encoder, states, actions, masks, L=L,
                                                  temperature=tau, contrastive_loss_fn=contrastive_loss_fn, num_segments=4,
                                                  pairwise_segments=True)[0]
            loss_seg_2 = segment_contrastive_loss(cls2, encoder, states, actions, masks, L=L,
                                                  temperature=tau, contrastive_loss_fn=contrastive_loss_fn, num_segments=4,
                                                  pairwise_segments=True)[0]
            # loss_seg_1 = segment_infomax_and_contrastive(cls1,encoder,states,actions,masks,L,infomax_loss_seg_fn,contrastive_loss_fn,num_segments=4,pairwise_segments=True)[0]
            # loss_seg_2 = segment_infomax_and_contrastive(cls2,encoder,states,actions,masks,L,infomax_loss_seg_fn,contrastive_loss_fn,num_segments=4,pairwise_segments=True)[0]
            loss_seg = (loss_seg_1 + loss_seg_2) / 2.0

            loss = beta * loss_contrastive + gamma * loss_infomax + sigma * loss_seg
            loss.backward()
            th.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        epoch_pbar.set_description(
            f"PT {epoch+1}/{epochs_pre} Avg: {total_loss/len(dataloader):.4f} "
            f"Ct:{beta*loss_contrastive.item():.4f} DIM:{gamma*loss_infomax.item():.4f} SG:{sigma*loss_seg.item():.4f}"
        )

    # --------------- Init Cluster Centroids (KMeans on CLS) ----------------
    encoder.eval()
    all_embeddings = []
    with th.no_grad():
        for batch in dataloader:
            states, actions, masks, _, _ = [b.to(device) for b in batch]
            _, _, _, _, cls_emb, _ = encoder(states, actions, src_key_padding_mask=masks)
            cls_emb = F.normalize(cls_emb, dim=1)
            all_embeddings.append(cls_emb.cpu())
    all_embeddings = th.cat(all_embeddings, dim=0).numpy()

    kmeans = KMeans(n_clusters=K, n_init=2000, max_iter=2000, random_state=seed, algorithm='elkan')
    kmeans.fit(all_embeddings)
    cluster_centers = kmeans.cluster_centers_

    cluster_centroids = th.tensor(cluster_centers, dtype=th.float, device=device, requires_grad=True)
    with th.no_grad():
        cluster_centroids[:] = F.normalize(cluster_centroids, dim=1)
    cluster_centroids = nn.Parameter(cluster_centroids)
    optimizer_centroids = th.optim.Adam([cluster_centroids], lr=1e-7)

    return encoder, None, None, cluster_centroids, None

def inference_on_dataloader_cdec_sa(
    encoder, cluster_centroids, dataloader, trajectory_manager, device, index_offset=0
):
    """
    Runs inference for the CDEC-SA model, assigning clusters based on nearest centroids.
    """
    encoder.eval()
    
    all_concatenations = []
    all_indices = []

    with th.no_grad():
        for batch in tqdm(dataloader, desc="Inference (CDEC-SA)"):
            states, actions, masks, _, indices_batch = [b.to(device) for b in batch]
            
            
            if encoder.model_type != "BE_SA_VDT":
                _, _, _, _, cls_emb, _ = encoder(states, actions, src_key_padding_mask=masks)
            else:
                _,_,_,_,cls_emb, _,_,_  = encoder(states, actions, src_key_padding_mask=masks)
            cls_emb = F.normalize(cls_emb, p=2, dim=1)
            # dists = th.cdist(cls_emb, cluster_centroids.detach())
            # predicted_labels = dists.argmin(dim=1)
            sims = F.cosine_similarity(cls_emb.unsqueeze(1), cluster_centroids.unsqueeze(0), dim=-1)
            predicted_labels = sims.argmax(dim=1)

            
            all_concatenations.append(cls_emb.cpu())
            all_indices.append(indices_batch.cpu() + index_offset)

            for i in range(len(indices_batch)):
                traj_idx = indices_batch[i].item() + index_offset  # Adjust index for online
                trajectory_manager[traj_idx]['cls_emb'] = cls_emb[i].cpu().numpy()
                trajectory_manager[traj_idx]['predicted_cluster_label'] = predicted_labels[i].item()
                trajectory_manager[traj_idx]['concatenation'] = cls_emb[i].cpu().numpy()

    final_concatenations = th.cat(all_concatenations, dim=0)
    final_indices = th.cat(all_indices, dim=0).numpy()

    return trajectory_manager, final_concatenations, final_indices

def _fit_hdbscan_seen(X_seen: np.ndarray, granularity: float, seed: int):
    nX = len(X_seen)
    min_cluster_size = max(5, int(granularity * nX))
    min_samples = max(1, int(math.sqrt(min_cluster_size)))
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='cosine'
    ).fit(X_seen)
    labels = model.labels_
    unique_core = np.unique(labels[labels != -1])
    # cluster centers as medians (robust), renormalize
    centers = []
    for c in unique_core:
        pts = X_seen[labels == c]
        centers.append(np.median(pts, axis=0))
    if len(centers) > 0:
        centers = np.vstack(centers)
        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
    else:
        centers = np.zeros((0, X_seen.shape[1]))
    return model, labels, unique_core, centers

def _build_registry(X_seen: np.ndarray, labels: np.ndarray, core_ids: np.ndarray, centers: np.ndarray) -> dict:
    # For each cluster: store center, cosine radius (95th pct), Mahalanobis stats, and ECDF for cosine distances
    reg = {}
    for i, cid in enumerate(core_ids):
        pts = X_seen[labels == cid]
        if len(pts) == 0:
            continue
        c = centers[i]  # unit-norm center
        # L2-normalize points for cosine geometry
        pts_n = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8)
        # cosine distance on unit sphere: d = 1 - cos
        cos = np.clip(pts_n @ c, -1.0, 1.0)
        d_cos = 1.0 - cos
        r95 = float(np.quantile(d_cos, 0.95))  # acceptance radius
        dcos_sorted = np.sort(d_cos.astype(np.float32))  # for ECDF lookup

        # covariance for Mahalanobis on normalized space (shrinkage)
        try:
            lw = LedoitWolf().fit(pts_n)
            mu = lw.location_
            prec = lw.precision_
        except Exception:
            print("Registry exception")
            mu = pts_n.mean(axis=0)
            cov = np.cov(pts_n.T) + 1e-6 * np.eye(pts_n.shape[1])
            prec = np.linalg.pinv(cov)
        reg[int(cid)] = {
            "center": c,   # unit
            "r95": r95,
            "mu": mu,      # normalized space
            "prec": prec,  # normalized space
            "count": len(pts),
            "dcos_sorted": dcos_sorted,
        }
    return reg

def _assign_or_flag_online(X_online: np.ndarray, registry: dict, maha_p: float = 0.99):
    # Assign by nearest center (cosine over L2-normalized vectors), then gate; score by calibrated ECDFs
    if len(registry) == 0 or len(X_online) == 0:
        empty = np.full((len(X_online),), -1, dtype=int)
        return empty, np.ones((len(X_online),), dtype=bool), np.zeros((len(X_online),), dtype=float)
    cids = sorted(registry.keys())
    C = np.stack([registry[c]["center"] for c in cids], axis=0)  # [C, D], unit-norm
    Xn = X_online / (np.linalg.norm(X_online, axis=1, keepdims=True) + 1e-8)

    # nearest center by cosine similarity
    sims = Xn @ C.T  # [N, C]
    argmax = sims.argmax(axis=1)
    best_cids = np.array([cids[j] for j in argmax], dtype=int)
    best_centers = C[argmax]

    # cosine gate
    d_cos = 1.0 - np.clip(np.sum(Xn * best_centers, axis=1), -1.0, 1.0)
    r95 = np.array([registry[int(cid)]["r95"] for cid in best_cids])
    pass_cos = d_cos <= r95

    # Mahalanobis gate
    df = Xn.shape[1]
    chi_thr = float(chi2.ppf(maha_p, df))
    maha_vals = []
    for x, cid in zip(Xn, best_cids):
        mu = registry[int(cid)]["mu"]
        prec = registry[int(cid)]["prec"]
        diff = x - mu
        m2 = float(diff @ prec @ diff)
        maha_vals.append(m2)
    maha_vals = np.array(maha_vals)
    pass_maha = maha_vals <= chi_thr

    # final labels
    accepted = pass_cos & pass_maha
    labels_online = np.where(accepted, best_cids, -1)
    novel_mask = labels_online == -1

    # probability-calibrated novelty score in [0,1], higher => more novel
    # F_cos: per-cluster ECDF(d_cos); F_maha: chi2 CDF(maha)
    F_cos = np.zeros_like(d_cos, dtype=np.float32)
    for i, (dc, cid) in enumerate(zip(d_cos, best_cids)):
        entry = registry[int(cid)]
        dsorted = entry.get("dcos_sorted", None)
        if dsorted is None or len(dsorted) == 0:
            # Fallback when no ECDF available for a cluster (e.g., newly spawned)
            # Use normalized distance vs. its radius as a proxy in [0,1]
            r_local = float(entry.get("r95", 1.0))
            F_cos[i] = float(np.clip(dc / max(r_local, 1e-6), 0.0, 1.0))
        else:
            r = np.searchsorted(dsorted, dc, side='right')
            F_cos[i] = r / max(1, len(dsorted))
    F_maha = chi2.cdf(maha_vals, df=df).astype(np.float32)

    novelty_scores = np.maximum(F_cos, F_maha)  # unified, comparable across clusters
    return labels_online, novel_mask, novelty_scores

def _spawn_new_clusters_from_buffer(novel_points: np.ndarray, min_cluster_size: int = 5) -> list[dict]:
    """
    From a buffer of novel points, discover micro-clusters and build complete
    registry entries for each (center, r95, mu/prec, dcos_sorted, count).
    """
    if novel_points is None or len(novel_points) < min_cluster_size:
        return []

    # Discover compact groups among novel points using cosine distance
    hdb_micro = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=max(1, int(math.sqrt(min_cluster_size))),
        metric='cosine'
    ).fit(novel_points)

    lbls = hdb_micro.labels_
    core_cids = np.unique(lbls[lbls != -1])
    new_entries: list[dict] = []

    for cid in core_cids:
        pts = novel_points[lbls == cid]
        if len(pts) < min_cluster_size:
            continue

        # Center as robust median, then L2-normalize
        ctr = np.median(pts, axis=0)
        ctr = ctr / (np.linalg.norm(ctr) + 1e-8)

        # L2-normalize points for cosine geometry
        pts_n = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8)
        
        # Cosine distance stats for radius and ECDF
        cos = np.clip(pts_n @ ctr, -1.0, 1.0)
        d_cos = 1.0 - cos
        r95 = float(np.quantile(d_cos, 0.95))
        dcos_sorted = np.sort(d_cos.astype(np.float32))

        # Mahalanobis stats in normalized space (shrinkage)
        try:
            lw = LedoitWolf().fit(pts_n)
            mu = lw.location_
            prec = lw.precision_
        except Exception:
            mu = pts_n.mean(axis=0)
            cov = np.cov(pts_n.T) + 1e-6 * np.eye(pts_n.shape[1])
            prec = np.linalg.pinv(cov)

        new_entries.append({
            "center": ctr,
            "r95": r95,
            "mu": mu,
            "prec": prec,
            "count": int(len(pts)),
            "dcos_sorted": dcos_sorted,
        })

    return new_entries

def _bootstrap_ci(metric_fn, y_true, scores, n_boot=500, alpha=0.95, seed=0):
    rng = np.random.RandomState(seed)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            vals.append(float(metric_fn(y_true[idx], scores[idx])))
        except Exception:
            vals.append(np.nan)
    vals = np.array(vals)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, (np.nan, np.nan)
    lo = np.quantile(vals, (1.0 - alpha) / 2.0)
    hi = np.quantile(vals, 1.0 - (1.0 - alpha) / 2.0)
    return float(vals.mean()), (float(lo), float(hi))

def _plot_novelty_metrics(y_true, continuous_scores, hard_mask, out_prefix, title="", show=True):
    """
    Plots ROC and PR curves (with baselines), confusion matrix at threshold 0.5,
    best-F1 threshold summary, and reliability (calibration) plot. Saves PDF to ./images/<out_prefix>_novelty.pdf.
    y_true: binary (1=novel / positive), continuous_scores: higher -> more novel,
    hard_mask: boolean array of hard decisions (True=novel).
    """
    os.makedirs("./images", exist_ok=True)
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # ROC / PR
    try:
        fpr, tpr, _ = roc_curve(y_true, continuous_scores)
        roc_auc = auc(fpr, tpr)
    except Exception:
        fpr, tpr, roc_auc = np.array([]), np.array([]), np.nan

    try:
        prec, rec, pr_thresh = precision_recall_curve(y_true, continuous_scores)
        ap = average_precision_score(y_true, continuous_scores)
    except Exception:
        prec, rec, pr_thresh, ap = np.array([]), np.array([]), np.array([]), np.nan

    # Bootstrap CIs
    mean_roc, (roc_lo, roc_hi) = _bootstrap_ci(roc_auc_score, np.asarray(y_true), np.asarray(continuous_scores), n_boot=200)
    mean_ap, (ap_lo, ap_hi) = _bootstrap_ci(average_precision_score, np.asarray(y_true), np.asarray(continuous_scores), n_boot=200)

    # Best F1 threshold (from PR curve)
    best_f1 = None
    best_thresh = None
    try:
        # prec, rec from precision_recall_curve contain one extra point; thresholds align with prec[1:], rec[1:]
        f1_scores = (2 * prec * rec) / (prec + rec + 1e-12)
        # Find best index excluding the last dummy point
        idx = np.nanargmax(f1_scores)
        best_f1 = float(f1_scores[idx])
        # threshold for best F1: pr_thresh has length len(prec)-1
        if len(pr_thresh) > 0 and idx > 0:
            best_thresh = float(pr_thresh[idx - 1])
        else:
            # Fallback: pick 0.5
            best_thresh = 0.5
    except Exception:
        best_f1 = np.nan
        best_thresh = 0.5

    # Predictions at standard 0.5 and at best-F1 threshold
    try:
        y_pred_05 = (np.asarray(continuous_scores) >= 0.5).astype(int)
        y_pred_best = (np.asarray(continuous_scores) >= best_thresh).astype(int)
    except Exception:
        y_pred_05 = np.zeros_like(y_true, dtype=int)
        y_pred_best = np.zeros_like(y_true, dtype=int)

    # Compute confusion matrices
    cm_05 = confusion_matrix(y_true, y_pred_05, labels=[1,0])  # row: true (pos,neg), col: pred (pos,neg)
    cm_best = confusion_matrix(y_true, y_pred_best, labels=[1,0])

    # Figure layout: ROC, PR, ConfMatrix, Reliability
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_roc = axes[0, 0]
    ax_pr = axes[0, 1]
    ax_conf = axes[1, 0]
    ax_cal = axes[1, 1]

    # ROC plot
    if fpr.size:
        ax_roc.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], 'k--', label="random (0.5)")
    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR (Recall)")
    ax_roc.set_title("ROC curve")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True)

    # PR plot (baseline = prevalence)
    if rec.size:
        ax_pr.plot(rec, prec, label=f"AP={ap:.3f}")
    prevalence = float(np.mean(y_true))
    ax_pr.hlines(prevalence, 0, 1, colors='k', linestyles='--', label=f"baseline={prevalence:.3f}")
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall curve")
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1)
    ax_pr.legend(loc="lower left")
    ax_pr.grid(True)

    # Confusion matrix at threshold=0.5 (annotated) and summary text for best-F1
    try:
        # Show the 0.5 threshold confusion matrix (rows: true novel=1, true seen=0)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_05, display_labels=["novel (1)", "seen (0)"])
        disp.plot(ax=ax_conf, cmap="Blues", colorbar=False, values_format='d')
        ax_conf.set_title("Confusion matrix (threshold=0.5)")
        # Add text with best-F1 threshold info
        text = f"best-F1 thresh={best_thresh:.3f}\nbest-F1={best_f1:.3f}\nTP={cm_best[0,0]} FP={cm_best[1,0]}\nFN={cm_best[0,1]} TN={cm_best[1,1]}"
        ax_conf.text(0.98, 0.02, text, transform=ax_conf.transAxes, ha='right', va='bottom', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    except Exception:
        ax_conf.text(0.1, 0.5, "Confusion matrix failed", fontsize=10)

    # Calibration / reliability
    try:
        prob_true, prob_pred = skcal.calibration_curve(y_true, continuous_scores, n_bins=8, strategy='quantile')
        ax_cal.plot(prob_pred, prob_true, marker='o', label='reliability')
        ax_cal.plot([0,1],[0,1],'k--', label='perfect')
        ax_cal.set_xlabel("Predicted score (binned)"); ax_cal.set_ylabel("Observed fraction positive")
        ax_cal.set_title("Reliability plot")
        ax_cal.legend()
    except Exception:
        ax_cal.text(0.1, 0.5, "Calibration plot failed", fontsize=10)

    fig.suptitle(title or f"Novelty evaluation: {out_prefix}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fn = f"./images/{out_prefix}_novelty.pdf"
    plt.savefig(fn, dpi=200)
    if show:
        plt.show()
    plt.close(fig)

    # print summary numbers
    print(f"[Novelty eval] ROC AUC = {roc_auc:.4f} (boot mean {mean_roc:.4f}, CI [{roc_lo:.3f},{roc_hi:.3f}])")
    print(f"[Novelty eval] AP      = {ap:.4f} (boot mean {mean_ap:.4f}, CI [{ap_lo:.3f},{ap_hi:.3f}])")
    print(f"[Novelty eval] Best-F1 thresh={best_thresh:.3f} Best-F1={best_f1:.4f}")

    return {
        "roc_auc": roc_auc, "roc_auc_boot_mean": mean_roc, "roc_auc_ci": (roc_lo, roc_hi),
        "ap": ap, "ap_boot_mean": mean_ap, "ap_ci": (ap_lo, ap_hi),
        "best_f1_thresh": best_thresh, "best_f1": best_f1
    }

def visualize_controller_output(
    Z_seen: np.ndarray,
    Z_online: np.ndarray,
    labels_seen: np.ndarray,
    labels_online: np.ndarray,
    novelty_scores: np.ndarray,
    registry: dict,
    reducer,
    title: str,
    is_3d: bool = False
):
    """Visualizes the output of the controller logic."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d' if is_3d else None)

    # 1. Plot Seen Data (training set for the controller)
    for cid in np.unique(labels_seen):
        mask = labels_seen == cid
        if cid == -1:
            color, marker, label = 'gray', 'x', "Seen (Noise)"
        else:
            color, marker, label = plt.cm.tab10(int(cid) % 10), 'o', f"Seen (Cluster {int(cid)})"
        
        if is_3d:
            ax.scatter(Z_seen[mask, 0], Z_seen[mask, 1], Z_seen[mask, 2], c=[color], s=20, alpha=0.6, marker=marker, label=label)
        else:
            ax.scatter(Z_seen[mask, 0], Z_seen[mask, 1], c=[color], s=20, alpha=0.6, marker=marker, label=label)

    # 2. Plot Online Data
    if len(Z_online) > 0:
        # Assigned points
        assigned_mask = labels_online != -1
        for cid in np.unique(labels_online[assigned_mask]):
            mask = labels_online == cid
            color, marker, label = plt.cm.tab10(int(cid) % 10), '^', f"Online (Assigned to {int(cid)})"
            if is_3d:
                ax.scatter(Z_online[mask, 0], Z_online[mask, 1], Z_online[mask, 2], c=[color], s=60, alpha=0.9, marker=marker, label=label)
            else:
                ax.scatter(Z_online[mask, 0], Z_online[mask, 1], c=[color], s=60, alpha=0.9, marker=marker, label=label)

        # Novel points (sized by novelty score)
        novel_mask = labels_online == -1
        if np.any(novel_mask):
            # Normalize scores for better size visibility
            scores_norm = (novelty_scores[novel_mask] - novelty_scores[novel_mask].min()) / (novelty_scores[novel_mask].max() - novelty_scores[novel_mask].min() + 1e-6)
            sizes = 50 + 200 * scores_norm
            
            if is_3d:
                p = ax.scatter(Z_online[novel_mask, 0], Z_online[novel_mask, 1], Z_online[novel_mask, 2], c='red', s=sizes, marker='*', label='Online (Novel)')
            else:
                p = ax.scatter(Z_online[novel_mask, 0], Z_online[novel_mask, 1], c='red', s=sizes, marker='*', label='Online (Novel)')
            
            # Add a colorbar legend for the novelty scores
            cbar = plt.colorbar(p, ax=ax, shrink=0.6)
            cbar.set_label('Novelty Score')
            # Set ticks to show the original score range
            ticks = np.linspace(novelty_scores[novel_mask].min(), novelty_scores[novel_mask].max(), 5)
            cbar.set_ticks(np.linspace(0, 1, 5))
            cbar.set_ticklabels([f"{t:.2f}" for t in ticks])


    # 3. Plot Cluster Centers from Registry
    if registry:
        centers_emb = np.stack([v['center'] for v in registry.values()])
        Z_centers = reducer.transform(centers_emb)
        if is_3d:
            ax.scatter(Z_centers[:, 0], Z_centers[:, 1], Z_centers[:, 2], c='black', s=250, marker='P', edgecolor='white', label='Cluster Centers')
        else:
            ax.scatter(Z_centers[:, 0], Z_centers[:, 1], c='black', s=250, marker='P', edgecolor='white', label='Cluster Centers')

    ax.set_title(title)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    if is_3d: ax.set_zlabel("UMAP-3")
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.show()

def evaluate_policy_reward(policy, env, num_episodes=50, mode_idx=None, env_name=""):
    """
    Evaluates a policy by running it in the environment for a number of episodes
    and returns the average total reward.
    """
    total_rewards = []
    for _ in range(num_episodes):
        # Reset the environment with the correct mode for the cluster being evaluated
        if env_name == "Traj2d":
            obs,info = env.unwrapped.reset(options={"mode_idx": mode_idx})
        elif hasattr(env.unwrapped, "reset"): # Handle VecEnv
             obs = env.unwrapped.reset(mode_idx=mode_idx)
             if isinstance(obs, tuple):
                obs = obs[0]
        else:
            obs, _ = env.reset(mode_idx=mode_idx)

        terminated = False
        truncated = False
        episode_reward = 0
        max_steps = env.unwrapped._max_episode_steps if hasattr(env, 'unwrapped') else env._max_episode_steps
        
        for _ in range(max_steps):
            action, _ = policy.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray) and env_name=="TwoLakesFishing":
                action = int(action.item())
            elif env_name == "Traj2d" or env_name == "Reacher-v4":
                action=action

            # Use unwrapped env for step if it exists, to ensure single env logic
            if hasattr(env.unwrapped, "step"):
                obs, reward, terminated, truncated, info = env.unwrapped.step(action)
            else:
                obs, reward, terminated, truncated, info = env.step(action)

            if env_name == "Traj2d":
                reward = info['reward_eval']
            elif env_name == "Reacher-v4":
                reward = info['reward_train']

            episode_reward += reward
            if terminated or truncated:
                break
        if env_name == "Traj2d":
            # Average the normalized step rewards over the episode length
            max_steps = env.unwrapped._max_episode_steps if hasattr(env, 'unwrapped') else env._max_episode_steps
            total_rewards.append(episode_reward / max_steps)
        else:
            total_rewards.append(episode_reward)
    if env_name == "Traj2d":
        mean_reward = np.mean(np.exp(total_rewards)) if total_rewards else 0.0
        std_reward = np.std(np.exp(total_rewards)) if total_rewards else 0.0
    else:
        mean_reward = np.mean(total_rewards) if total_rewards else 0.0
        std_reward = np.std(total_rewards) if total_rewards else 0.0
    return mean_reward, std_reward

def calculate_expert_reward(trajectories, env, mode_idx=None, env_name=""):
    """
    Calculates the average reward of a set of expert trajectories by replaying them.
    This version resets the environment to the trajectory's actual start state.
    """
    total_rewards = []
    for traj in trajectories:
        # Reset the environment to the trajectory's specific start state.
        initial_obs = traj.obs[0]
        if env_name == "Traj2d":
            env.unwrapped.reset(options={"mode_idx": mode_idx})
        elif env_name == "Reacher-v4":
            env.unwrapped.reset(mode_idx=mode_idx)
        else:
            env.unwrapped.reset(mode_idx=mode_idx, start_state=initial_obs)

        episode_reward = 0
        # Replay the actions from the trajectory
        for action in traj.acts:
            if env_name == "Traj2d" or env_name == "Reacher-v4":
                action = action
            else:
                if isinstance(action, np.ndarray):
                    action = int(action.item())
            
            # Use unwrapped env for step if it exists
            if hasattr(env.unwrapped, "step"):
                obs, reward, terminated, truncated, info = env.unwrapped.step(action)
            else:
                obs, reward, terminated, truncated, info = env.step(action)
            
            if env_name == "Traj2d":
                reward = info['reward_eval']
            elif env_name == "Reacher-v4":
                reward = info['reward_train']
            
            episode_reward += reward
            if terminated or truncated:
                break
        
        if env_name == "Traj2d":
            # Average the normalized step rewards over the episode length
            max_steps = env.unwrapped._max_episode_steps if hasattr(env, 'unwrapped') else env._max_episode_steps
            total_rewards.append(episode_reward / max_steps)
        else:
            total_rewards.append(episode_reward)
    if env_name == "Traj2d":
        mean_reward = np.mean(np.exp(total_rewards)) if total_rewards else 0.0
        std_reward = np.std(np.exp(total_rewards)) if total_rewards else 0.0
    else:
        mean_reward = np.mean(total_rewards) if total_rewards else 0.0
        std_reward = np.std(total_rewards) if total_rewards else 0.0
    return mean_reward, std_reward

def calculate_original_expert_reward_stats(trajectories_with_rew: list[TrajectoryWithRew]):
    """
    Calculates the mean and standard deviation of rewards directly from a list
    of trajectories that have rewards stored in them. This avoids the
    stochasticity of replaying actions in the environment.
    """
    total_rewards = [np.sum(traj.rews) for traj in trajectories_with_rew if hasattr(traj, 'rews') and traj.rews is not None]
    if not total_rewards:
        return 0.0, 0.0
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    return mean_reward, std_reward

def visualize_classic_scalers_on_flat_states(
    seen_states, seen_labels, online_states, online_labels,
    seed=0, n_neighbors=20, min_dist=0.5,
    fit_mode: str = "both",          # NEW: 'seen' or 'both'
    umap_metric: str = "cosine"   # NEW: 'euclidean' or 'cosine'
):
    scalers = {
        "Standard": StandardScaler(),
        "MinMax(0,1)": MinMaxScaler(feature_range=(-1,1)),
        "MaxAbs": MaxAbsScaler(),
        "Robust": RobustScaler(quantile_range=(25, 75)),
        "Quantile-Uniform": QuantileTransformer(n_quantiles=min(1000, len(seen_states)), output_distribution='uniform', random_state=seed),
        "Quantile-Normal": QuantileTransformer(n_quantiles=min(1000, len(seen_states)), output_distribution='normal', random_state=seed),
        "Power(Yeo-Johnson)": PowerTransformer(method='yeo-johnson', standardize=True),
        "Normalizer(L2)": Normalizer(norm='l2'),
    }
    print(f"[Scaler Viz] Seen={seen_states.shape}, Online={online_states.shape}, fit_mode={fit_mode}, umap_metric={umap_metric}")

    # UMAP reducer (fit on seen-only or both, see below)
    reducer = umap.UMAP(
        random_state=seed, n_neighbors=n_neighbors, min_dist=min_dist,
        n_components=2, metric=umap_metric
    )

    cols = 3
    rows = int(np.ceil(len(scalers) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten()
    for ax in axes[len(scalers):]:
        ax.axis('off')

    for idx, (name, scaler) in enumerate(scalers.items()):
        try:
            # Fit scaler on seen or both (visualization only)
            fit_X = seen_states if fit_mode == "seen" else np.vstack([seen_states, online_states])
            scaler.fit(fit_X)
            X_seen = scaler.transform(seen_states)
            X_online = scaler.transform(online_states)
        except Exception as e:
            print(f"[Scaler Viz] {name}: failed to fit/transform ({e}). Skipping.")
            axes[idx].axis('off')
            continue

        # Stats on seen (unchanged)
        dists = np.linalg.norm(X_seen[:, None] - X_seen[None, :], axis=-1)
        y = np.array(seen_labels)
        intra = dists[y[:, None] == y[None, :]]
        inter = dists[y[:, None] != y[None, :]]
        intra_mean = float(intra.mean()) if intra.size else float('nan')
        inter_mean = float(inter.mean()) if inter.size else float('nan')
        print(f"[Scaler Viz] {name}: mean intra={intra_mean:.4f}, mean inter={inter_mean:.4f}")

        # Fit UMAP on seen or both, then split/transform accordingly
        if fit_mode == "both":
            X_all = np.vstack([X_seen, X_online])
            Z_all = reducer.fit_transform(X_all)
            Z_seen = Z_all[:len(X_seen)]
            Z_online = Z_all[len(X_seen):]
        else:
            Z_seen = reducer.fit_transform(X_seen)
            Z_online = reducer.transform(X_online)

        ax = axes[idx]
        # Seen (true labels)
        for lbl in np.unique(seen_labels):
            mask = (np.array(seen_labels) == lbl)
            ax.scatter(Z_seen[mask, 0], Z_seen[mask, 1], s=12, alpha=0.7, label=f"Seen {int(lbl)}")
        # Online (true labels)
        for lbl in np.unique(online_labels):
            mask = (np.array(online_labels) == lbl)
            ax.scatter(Z_online[mask, 0], Z_online[mask, 1], s=28, alpha=0.9, marker='^', label=f"Online {int(lbl)}")
        ax.set_title(f"{name}\n(intra={intra_mean:.3f}, inter={inter_mean:.3f})")
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        ax.legend(fontsize=8, loc='best')

    plt.tight_layout()
    plt.show()

def _build_scaler(name: str, seed: int = 0, feature_range=(-1.0, 1.0), n_quantiles: int = 1000):
    name = name.lower()
    if name == 'standard': return StandardScaler()
    if name == 'minmax': return MinMaxScaler(feature_range=tuple(feature_range))
    if name == 'maxabs': return MaxAbsScaler()
    if name == 'robust': return RobustScaler(quantile_range=(25, 75))
    if name == 'quantile_uniform': return QuantileTransformer(n_quantiles=n_quantiles, output_distribution='uniform', random_state=seed)
    if name == 'quantile_normal': return QuantileTransformer(n_quantiles=n_quantiles, output_distribution='normal', random_state=seed)
    if name == 'power_yeo': return PowerTransformer(method='yeo-johnson', standardize=True)
    if name == 'l2norm': return Normalizer(norm='l2')
    if name == 'none': return None
    raise ValueError(f"Unknown scaler: {name}")

def _is_coord_states(x) -> bool:
    x = np.asarray(x)
    return x.ndim == 2  # [T, D]

def _is_cont_actions(a_list) -> bool:
    a0 = np.asarray(a_list[0])
    return a0.ndim == 2 and np.issubdtype(a0.dtype, np.floating)

def _fit_on_flat(list_of_td, scaler, fit_mode='seen', list_of_td_online=None):
    # Flatten: stack all [T,D] -> [sum_T, D]
    X_seen = np.vstack([np.asarray(s) for s in list_of_td])
    if fit_mode == 'both' and list_of_td_online is not None:
        X_online = np.vstack([np.asarray(s) for s in list_of_td_online])
        X_fit = np.vstack([X_seen, X_online])
    else:
        X_fit = X_seen
    scaler.fit(X_fit)
    return scaler

def _apply_per_traj(list_of_td, scaler):
    if scaler is None: return list_of_td
    out = []
    for s in list_of_td:
        s_np = np.asarray(s)
        if s_np.ndim != 2:
            out.append(s)  # leave grids or discrete as-is
            continue
        T, D = s_np.shape
        out.append(scaler.transform(s_np.reshape(-1, D)).reshape(T, D))
    return out

def get_label_display_properties(HIGH_CONTRAST_PREDICTED_COLORS,label_value, args_env, current_color_label_mapping, label_type="true"):
    color = "gray"  # Default color
    name = f"Unknown Label {label_value}" # Default name

    if label_type == "true":
        # Find the color and name from the environment-specific mapping
        # The current_color_label_mapping is {color_string: behavior_name_string}
        # We need to find which color_string corresponds to label_value
        if args_env.Highway:
            if label_value == 5: color = "tab:blue"
            elif label_value == 6: color = "tab:orange"
            elif label_value == 7: color = "tab:green"
            elif label_value == 8: color = "tab:red"
            elif label_value == 9: color = "tab:purple"
            elif label_value == 10: color = "tab:brown"
            else: color = "b" # Default for '???'
            name = current_color_label_mapping.get(color, f"Behavior {label_value}")
        elif args_env.PuddleWorld:
            if label_value == 10: color = "b"
            elif label_value == 11: color = "r"
            else: color = "g"
            name = current_color_label_mapping.get(color, f"Behavior {label_value}")
        elif args_env.GridWorld:
            if label_value == 10: color = "g"
            elif label_value == 11: color = "y"
            elif label_value == 12: color = "r"
            elif label_value == 13: color = "tab:pink"
            else: color = "b"
            name = current_color_label_mapping.get(color, f"Behavior {label_value}")
        elif args_env.TwoLakesFishing or args_env.CAL:
            if label_value == 10: color = "b"
            elif label_value == 11: color = "r"
            else: color = "g"
            name = current_color_label_mapping.get(color, f"Behavior {label_value}")
        elif args_env.Traj2d:
            if label_value == 10: color = "tab:blue"
            elif label_value == 11: color = "tab:red"
            elif label_value == 12: color = "tab:green"
            elif label_value == 13: color = "tab:purple"
            elif label_value == 14: color = "tab:brown"
            elif label_value == 15: color = "tab:orange"
            else: color = "g"
            name = current_color_label_mapping.get(color, f"Behavior {label_value}")
        elif args_env.Reacherv4 or args_env.Pusherv4:
            if label_value == 10: color = "tab:blue"
            elif label_value == 11: color = "tab:red"
            elif label_value == 12: color = "tab:green"
            elif label_value == 13: color = "tab:purple"
            elif label_value == 14: color = "tab:brown"
            elif label_value == 15: color = "tab:orange"
            else: color = "g"
            name = current_color_label_mapping.get(color, f"Behavior {label_value}")
        elif args_env.Humanoidv4 or args_env.Walker2dv4:
            if label_value == 10: color = "tab:blue"
            elif label_value == 11: color = "tab:red"
            elif label_value == 12: color = "tab:green"
            else: color = "g"
            name = current_color_label_mapping.get(color, f"Behavior {label_value}")
    elif label_type == "predicted_ac": # Agglomerative Clustering
        color = HIGH_CONTRAST_PREDICTED_COLORS[label_value % len(HIGH_CONTRAST_PREDICTED_COLORS)]
        name = f"AC Cluster {label_value}"
    elif label_type == "predicted_cp": # Model's own Cluster Projector
        color = HIGH_CONTRAST_PREDICTED_COLORS[label_value % len(HIGH_CONTRAST_PREDICTED_COLORS)]
        name = f"Model Pred. Cluster {label_value}"
    return {"color": color, "name": name}

def compute_mode_counts(num_trajs, num_modes, ratio):
    # ratio: float >= 1.0
    weights = [ratio**(num_modes-1-i) for i in range(num_modes)]
    total = sum(weights)
    counts = [int(round(num_trajs * w / total)) for w in weights]
    # Adjust last count to ensure sum == num_trajs
    counts[-1] += num_trajs - sum(counts)
    return counts

## PARSER

def setup_parser():
    parser = argparse.ArgumentParser(description='Model-Free Multi Intention Maximum Likelihood IRL: Experiments Runner')
    
    arg_env = parser.add_argument_group('Environment Selection')
    # arg_env.add_argument("-G","--GridWorld", help="Apply the selected algorithm to the 4 corners GridWorld",action="store_true")
    # arg_env.add_argument("-P","--PuddleWorld", help="Apply the selected algorithm to the PuddleWorld",action="store_true")
    # arg_env.add_argument("-HW","--Highway", help="Apply the selected algorithm to the Highway environment",action="store_true")
    # arg_env.add_argument("-F","--TwoLakesFishing", help="Apply the selected algorithm to the TwoLakesFishing environment",action="store_true")
    # arg_env.add_argument("-C","--CAL", help="Apply the selected algorithm to the CAL environment",action="store_true")
    arg_env.add_argument("-T2D","--Traj2d", help="Apply the selected algorithm to the Traj2d environment",action="store_true")
    arg_env.add_argument("-Rv4","--Reacherv4", help="Apply the selected algorithm to the Reacher-v4 environment",action="store_true")
    arg_env.add_argument("-Pv4","--Pusherv4", help="Apply the selected algorithm to the Pusher-v4 environment",action="store_true")
    arg_env.add_argument("-Hv4","--Humanoidv4", help="Apply the selected algorithm to the Humanoid-v4 environment",action="store_true")
    arg_env.add_argument("-W2D","--Walker2dv4", help="Apply the selected algorithm to the Walker2d-v4 environment",action="store_true")

    arg_alg = parser.add_argument_group('Unseen Split Selection')
    arg_alg.add_argument("-split","--use_seen_unseen_split", action="store_true", 
                    help="Use seen-unseen split for training. If not set, train on the entire dataset.")
    arg_alg.add_argument("-nUnModes","--num_unseen_modes", type=int, default=1, help="Number of unseen modes in the unseen split for the dataset.")

    arg_irl = parser.add_argument_group('IRL Selection')
    arg_irl.add_argument("-gail","--GAIL", help="Run GAIL IRL",action="store_true")
    arg_irl.add_argument("-sqil","--SQIL", help="Run SQIL IRL",action="store_true")
    arg_irl.add_argument("-airl","--AIRL", help="Run AIRL IRL",action="store_true")

    arg_hyp = parser.add_argument_group('Hyperparameters')
    arg_hyp.add_argument('-em','--max_em_iters', type=int,default=20,help='int: Maximum number of EM iterations')
    arg_hyp.add_argument('-irl_lr','--irl_learning_rate', type=float,default=0.03,help='float: Learning rate for IRL')
    arg_hyp.add_argument('-irl_tol','--irl_max_likelihood_change', type=float,default=0.0002,help='float: Tolerance for IRL convergence')
    arg_hyp.add_argument('-irl_steps','--irl_max_steps', type=int,default=100,help='int: Maximum number of IRL steps')
    arg_hyp.add_argument('-beta','--boltzmann_beta', type=float,default=0.5,help='float: Boltzmann beta for action selection')
    arg_hyp.add_argument('-nT','--num_trajs', type=int,default=100,help='int: Number of expert trajectories to generate')
    arg_hyp.add_argument('-seed','--seed', type=int,default=42,help='int: Random seed for reproducibility') 
    arg_hyp.add_argument('--ratio', type=int, default=1,help="Ratio for splitting trajectories between modes. 1 uniform 3 first gets most last gets least, etc.")
    arg_hyp.add_argument('--embedding_strategy', type=str, default='cls_only', choices=['cls_only', 'hybrid','goal_oriented'], help='Strategy for creating the final trajectory embedding for clustering.')

    arg_vis = parser.add_argument_group('Visualization')
    arg_vis.add_argument('-vW','--visualize_world', help='Visualize the GridWorld or PuddleWorld',action="store_true")
    arg_vis.add_argument('-vA','--visualize_attention', help='Visualize the interaction and aggregated attention',action="store_true")
    arg_vis.add_argument('-vP','--visualize_posterior', help='Visualize the posterior probabilities',action="store_true")
    arg_vis.add_argument('-vC','--visualize_clusters', help='Visualize the clusters',action="store_true")
    arg_vis.add_argument('-vT','--visualize_transformer', help='Visualize the transformer attention layer',action="store_true")
    arg_vis.add_argument('-vG','--visualize_graphs', help='Visualize the cluster quality graphs',action="store_true")
    arg_vis.add_argument('-3d','--threeD', help='Visualize UMAP in 3D',action="store_true",default=False)
    arg_vis.add_argument('-r','--render', help='Render the environment',action="store_true",default=False)
    return parser

## MAIN

def main():
    parser = setup_parser()
    args = parser.parse_args()
    if args.Traj2d:
        unseen_modes = args.num_unseen_modes
        K = 6-unseen_modes if args.use_seen_unseen_split else 6
        K_known = 6
    # elif args.PuddleWorld:
    #     K = 2
    # elif args.Highway:
    #     K = 3 #before 6 but now simpler with 3 behaviors
    #     timesteps = 100000
    # elif args.TwoLakesFishing or args.CAL:
    #     K = 2
    elif args.Reacherv4 or args.Pusherv4:
        unseen_modes = args.num_unseen_modes
        K = 6-unseen_modes if args.use_seen_unseen_split else 6
        K_known = 6
    elif args.Humanoidv4 or args.Walker2dv4:
        K = 2 if args.use_seen_unseen_split else 3
        unseen_modes = 1
        K_known = 3
    else:
        raise ValueError("No available environment selected. Please select either --Two Lakes Fishing (-F), --Traj2d(-T2D), --Reacher-v4(-Rv4) or --Pusher-v4(-Pv4).")

    unseen_modes = args.num_unseen_modes
    visualize_original = False
    visualize_scalers = False
    normalize_input = True
    saving = False
    # EM_ITERS = args.max_em_iters
    # SACC=False
    # irl_learning_rate = args.irl_learning_rate
    # irl_max_likelihood_change = args.irl_max_likelihood_change
    # irl_max_steps = args.irl_max_steps
    # boltzmann_beta = args.boltzmann_beta
    # visualize_world = args.visualize_world
    # visualize_posterior = args.visualize_posterior
    num_trajs = args.num_trajs
    # model_free = args.ModelFree
    # multi_intention = args.MultiIntention
    embedding_strategy_code = "CLS" if args.embedding_strategy == "cls_only" else "HYB" if args.embedding_strategy == "hybrid" else "GO" if args.embedding_strategy == "goal_oriented" else "UNK"
    training_code = "SPLIT" if args.use_seen_unseen_split else "FULL"
    # em_tol = 0.0001
    env_name =  "Traj2d" if args.Traj2d else "Reacher-v4" if args.Reacherv4 else "Pusher-v4" if args.Pusherv4 else "Humanoid-v4" if args.Humanoidv4 else "Walker2d-v4" if args.Walker2dv4 else "Unknown"
    env_id =  "Traj2d" if args.Traj2d else "Reacher-v4" if args.Reacherv4 else "Pusher-v4" if args.Pusherv4 else "Humanoid-v4" if args.Humanoidv4 else "Walker2d-v4" if args.Walker2dv4 else "UnknownEnv"
    env_code = "T2D" if args.Traj2d else "Rv4" if args.Reacherv4 else "Pv4" if args.Pusherv4 else "Hv4" if args.Humanoidv4 else "W2D" if args.Walker2dv4 else "UNK"
    # model_name = "Model-Free" if model_free else "Model-Based"
    # intention_kind = "Multi-Intention" if multi_intention else "Single Intention"
    # tr_name = "gail" if args.GAIL else "airl" if args.AIRL else "sqil"
    print(f"*** CoMIIRL approach on {env_name} ***")

    # Set the random seed for reproducibility
    SEEDS = [0,1,2,3,4]
    SEED = args.seed
    ratio = args.ratio
    SA = True
    th.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # Set the device to GPU if available
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    HIGH_CONTRAST_PREDICTED_COLORS = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', 
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]

    trajectories_directory_path = "trajs/expert_trajectories/"

    trajectory_manager = {}

    if args.Traj2d or args.Reacherv4 or args.Pusherv4:#deep approaches
        name_env = "2D-Trajectory" if args.Traj2d else "Reacher-v4" if args.Reacherv4 else "Pusher-v4" if args.Pusherv4 else "Unknown"
        file_path = f"{trajectories_directory_path}expert_imitation_trajectories_{name_env}_mode_0.pkl"
        file_path_withrew = file_path.replace(".pkl", "_withrew.pkl")
        with open(file_path, "rb") as f:
                demos_0 = pickle.load(f)
                labels_0 = np.array([10]*len(demos_0))
                print(f"Loaded {len(demos_0)} expert trajectories for mode 0")
        with open(file_path_withrew, "rb") as f:
                demos_0_withrew = pickle.load(f)
        file_path = f"{trajectories_directory_path}expert_imitation_trajectories_{name_env}_mode_1.pkl"
        file_path_withrew = file_path.replace(".pkl", "_withrew.pkl")
        with open(file_path, "rb") as f:
                demos_1 = pickle.load(f)
                labels_1 = np.array([11]*len(demos_1))
                print(f"Loaded {len(demos_1)} expert trajectories for mode 1")
        with open(file_path_withrew, "rb") as f:
                demos_1_withrew = pickle.load(f)
        file_path = f"{trajectories_directory_path}expert_imitation_trajectories_{name_env}_mode_2.pkl"
        file_path_withrew = file_path.replace(".pkl", "_withrew.pkl")
        with open(file_path, "rb") as f:
                demos_2 = pickle.load(f)
                labels_2 = np.array([12]*len(demos_2))
                print(f"Loaded {len(demos_2)} expert trajectories for mode 2")
        with open(file_path_withrew, "rb") as f:
                demos_2_withrew = pickle.load(f)
        file_path = f"{trajectories_directory_path}expert_imitation_trajectories_{name_env}_mode_3.pkl"
        file_path_withrew = file_path.replace(".pkl", "_withrew.pkl")
        with open(file_path, "rb") as f:
                demos_3 = pickle.load(f)
                labels_3 = np.array([13]*len(demos_3))
                print(f"Loaded {len(demos_3)} expert trajectories for mode 3")
        with open(file_path_withrew, "rb") as f:
                demos_3_withrew = pickle.load(f)
        file_path = f"{trajectories_directory_path}expert_imitation_trajectories_{name_env}_mode_4.pkl"
        file_path_withrew = file_path.replace(".pkl", "_withrew.pkl")
        with open(file_path, "rb") as f:
                demos_4 = pickle.load(f)
                labels_4 = np.array([14]*len(demos_4))
                print(f"Loaded {len(demos_4)} expert trajectories for mode 4")
        with open(file_path_withrew, "rb") as f:
                demos_4_withrew = pickle.load(f)
        file_path = f"{trajectories_directory_path}expert_imitation_trajectories_{name_env}_mode_5.pkl"
        file_path_withrew = file_path.replace(".pkl", "_withrew.pkl")
        with open(file_path, "rb") as f:
                demos_5 = pickle.load(f)
                labels_5 = np.array([15]*len(demos_5))
                print(f"Loaded {len(demos_5)} expert trajectories for mode 5")
        with open(file_path_withrew, "rb") as f:
                demos_5_withrew = pickle.load(f)
        K_to_split = K+unseen_modes if args.use_seen_unseen_split else K
        counts = compute_mode_counts(num_trajs, K_to_split, ratio)
        print(f"Splitting {num_trajs} trajectories into {K_to_split} modes with ratio {ratio}: {counts}")
        if args.use_seen_unseen_split:
            if args.Traj2d:
                print(f"Processing Traj2d trajectories with {unseen_modes} unseen modes. . . ")
                if unseen_modes == 1:
                    trajectories = demos_0[:counts[0]] + demos_1[:counts[1]] + demos_2[:counts[2]] + demos_3[:counts[3]] + demos_4[:counts[4]]
                    trajectories_withrew = demos_0_withrew[:counts[0]] + demos_1_withrew[:counts[1]] + demos_2_withrew[:counts[2]] + demos_3_withrew[:counts[3]] + demos_4_withrew[:counts[4]]
                    true_labels = np.concatenate((labels_0[:counts[0]], labels_1[:counts[1]], labels_2[:counts[2]], labels_3[:counts[3]], labels_4[:counts[4]]))
                elif unseen_modes == 2:
                    trajectories = demos_0[:counts[0]] + demos_1[:counts[1]] + demos_3[:counts[3]] + demos_4[:counts[4]]
                    trajectories_withrew = demos_0_withrew[:counts[0]] + demos_1_withrew[:counts[1]] + demos_3_withrew[:counts[3]] + demos_4_withrew[:counts[4]]
                    true_labels = np.concatenate((labels_0[:counts[0]], labels_1[:counts[1]], labels_3[:counts[3]], labels_4[:counts[4]]))
                elif unseen_modes == 3:
                    trajectories = demos_0[:counts[0]] + demos_3[:counts[3]] + demos_4[:counts[4]]
                    trajectories_withrew = demos_0_withrew[:counts[0]] + demos_3_withrew[:counts[3]] + demos_4_withrew[:counts[4]]
                    true_labels = np.concatenate((labels_0[:counts[0]], labels_3[:counts[3]], labels_4[:counts[4]]))#, labels_3[:counts[3]]))
                else:
                    raise ValueError("For Traj2d, when using unseen split, unseen_modes must be 1, 2, or 3.")
                unseen_trajectories_for_online = demos_0[counts[0]:counts[0]+counts[0]] + demos_1[counts[1]:counts[1]+counts[1]] + demos_2[counts[2]:counts[2]+counts[2]] + demos_3[counts[3]:counts[3]+counts[3]] + demos_4[counts[4]:counts[4]+counts[4]] + demos_5[counts[5]:counts[5]+counts[5]]
                trajectories_withrew_for_online = demos_0_withrew[counts[0]:counts[0]+counts[0]] + demos_1_withrew[counts[1]:counts[1]+counts[1]] + demos_2_withrew[counts[2]:counts[2]+counts[2]] + demos_3_withrew[counts[3]:counts[3]+counts[3]] + demos_4_withrew[counts[4]:counts[4]+counts[4]] + demos_5_withrew[counts[5]:counts[5]+counts[5]]
                true_labels_online = np.concatenate((labels_0[counts[0]:counts[0]+counts[0]], labels_1[counts[1]:counts[1]+counts[1]], labels_2[counts[2]:counts[2]+counts[2]], labels_3[counts[3]:counts[3]+counts[3]], labels_4[counts[4]:counts[4]+counts[4]], labels_5[counts[5]:counts[5]+counts[5]]))
            elif args.Reacherv4:
                print(f"Processing Reacher-v4 environment with {unseen_modes} unseen modes...")
                if unseen_modes == 1:
                    trajectories = demos_0[:counts[0]] + demos_1[:counts[1]] + demos_2[:counts[2]] + demos_3[:counts[3]] + demos_4[:counts[4]]
                    trajectories_withrew = demos_0_withrew[:counts[0]] + demos_1_withrew[:counts[1]] + demos_2_withrew[:counts[2]] + demos_3_withrew[:counts[3]] + demos_4_withrew[:counts[4]]
                    true_labels = np.concatenate((labels_0[:counts[0]], labels_1[:counts[1]], labels_2[:counts[2]], labels_3[:counts[3]], labels_4[:counts[4]]))
                elif unseen_modes == 2:
                    trajectories = demos_0[:counts[0]] + demos_1[:counts[1]] + demos_3[:counts[3]] + demos_4[:counts[4]]
                    trajectories_withrew = demos_0_withrew[:counts[0]] + demos_1_withrew[:counts[1]] + demos_3_withrew[:counts[3]] + demos_4_withrew[:counts[4]]
                    true_labels = np.concatenate((labels_0[:counts[0]], labels_1[:counts[1]], labels_3[:counts[3]], labels_4[:counts[4]]))
                elif unseen_modes == 3:
                    trajectories = demos_0[:counts[0]] + demos_2[:counts[2]] + demos_4[:counts[4]]
                    trajectories_withrew = demos_0_withrew[:counts[0]] + demos_2_withrew[:counts[2]] + demos_4_withrew[:counts[4]]
                    true_labels = np.concatenate((labels_0[:counts[0]], labels_2[:counts[2]], labels_4[:counts[4]]))#, labels_3[:counts[3]]))
                else:
                    raise ValueError("For Reacher-v4, when using unseen split, unseen_modes must be 1, 2, or 3.")
                unseen_trajectories_for_online = demos_0[counts[0]:counts[0]+counts[0]] + demos_1[counts[1]:counts[1]+counts[1]] + demos_2[counts[2]:counts[2]+counts[2]] + demos_3[counts[3]:counts[3]+counts[3]] + demos_4[counts[4]:counts[4]+counts[4]] + demos_5[counts[5]:counts[5]+counts[5]]
                trajectories_withrew_for_online = demos_0_withrew[counts[0]:counts[0]+counts[0]] + demos_1_withrew[counts[1]:counts[1]+counts[1]] + demos_2_withrew[counts[2]:counts[2]+counts[2]] + demos_3_withrew[counts[3]:counts[3]+counts[3]] + demos_4_withrew[counts[4]:counts[4]+counts[4]] + demos_5_withrew[counts[5]:counts[5]+counts[5]]
                true_labels_online = np.concatenate((labels_0[counts[0]:counts[0]+counts[0]], labels_1[counts[1]:counts[1]+counts[1]], labels_2[counts[2]:counts[2]+counts[2]], labels_3[counts[3]:counts[3]+counts[3]], labels_4[counts[4]:counts[4]+counts[4]], labels_5[counts[5]:counts[5]+counts[5]]))
            elif args.Pusherv4:
                print(f"Processing Pusher-v4 environment with {unseen_modes} unseen modes...")
                if unseen_modes == 1:
                    trajectories = demos_0[:counts[0]] + demos_1[:counts[1]] + demos_2[:counts[2]] + demos_3[:counts[3]] + demos_4[:counts[4]]
                    trajectories_withrew = demos_0_withrew[:counts[0]] + demos_1_withrew[:counts[1]] + demos_2_withrew[:counts[2]] + demos_3_withrew[:counts[3]] + demos_4_withrew[:counts[4]]
                    true_labels = np.concatenate((labels_0[:counts[0]], labels_1[:counts[1]], labels_2[:counts[2]], labels_3[:counts[3]], labels_4[:counts[4]]))
                elif unseen_modes == 2:
                    trajectories = demos_0[:counts[0]] + demos_1[:counts[1]] + demos_3[:counts[3]] + demos_4[:counts[4]]
                    trajectories_withrew = demos_0_withrew[:counts[0]] + demos_1_withrew[:counts[1]] + demos_3_withrew[:counts[3]] + demos_4_withrew[:counts[4]]
                    true_labels = np.concatenate((labels_0[:counts[0]], labels_1[:counts[1]], labels_3[:counts[3]], labels_4[:counts[4]]))
                elif unseen_modes == 3:
                    trajectories = demos_0[:counts[0]] + demos_2[:counts[2]] + demos_4[:counts[4]]
                    trajectories_withrew = demos_0_withrew[:counts[0]] + demos_2_withrew[:counts[2]] + demos_4_withrew[:counts[4]]
                    true_labels = np.concatenate((labels_0[:counts[0]], labels_2[:counts[2]], labels_4[:counts[4]]))#, labels_3[:counts[3]]))
                else:
                    raise ValueError("For Pusher-v4, when using unseen split, unseen_modes must be 1, 2, or 3.")
                
                unseen_trajectories_for_online = demos_0[counts[0]:counts[0]+counts[0]] + demos_1[counts[1]:counts[1]+counts[1]] + demos_2[counts[2]:counts[2]+counts[2]] + demos_3[counts[3]:counts[3]+counts[3]] + demos_4[counts[4]:counts[4]+counts[4]] + demos_5[counts[5]:counts[5]+counts[5]]
                trajectories_withrew_for_online = demos_0_withrew[counts[0]:counts[0]+counts[0]] + demos_1_withrew[counts[1]:counts[1]+counts[1]] + demos_2_withrew[counts[2]:counts[2]+counts[2]] + demos_3_withrew[counts[3]:counts[3]+counts[3]] + demos_4_withrew[counts[4]:counts[4]+counts[4]] + demos_5_withrew[counts[5]:counts[5]+counts[5]]
                true_labels_online = np.concatenate((labels_0[counts[0]:counts[0]+counts[0]], labels_1[counts[1]:counts[1]+counts[1]], labels_2[counts[2]:counts[2]+counts[2]], labels_3[counts[3]:counts[3]+counts[3]], labels_4[counts[4]:counts[4]+counts[4]], labels_5[counts[5]:counts[5]+counts[5]]))
            else:
                raise ValueError("Unsupported environment. Please select either Traj2d, Reacher-v4, or Pusher-v4.")
        else:
            trajectories = demos_0[:counts[0]] + demos_1[:counts[1]] + demos_2[:counts[2]] + demos_3[:counts[3]] + demos_4[:counts[4]] + demos_5[:counts[5]]
            trajectories_withrew = demos_0_withrew[:counts[0]] + demos_1_withrew[:counts[1]] + demos_2_withrew[:counts[2]] + demos_3_withrew[:counts[3]] + demos_4_withrew[:counts[4]] + demos_5_withrew[:counts[5]]
            true_labels = np.concatenate((labels_0[:counts[0]], labels_1[:counts[1]], labels_2[:counts[2]], labels_3[:counts[3]], labels_4[:counts[4]], labels_5[:counts[5]]))

            unseen_trajectories_for_online = demos_0[counts[0]:counts[0]+counts[0]] + demos_1[counts[1]:counts[1]+counts[1]] + demos_2[counts[2]:counts[2]+counts[2]] + demos_3[counts[3]:counts[3]+counts[3]] + demos_4[counts[4]:counts[4]+counts[4]] + demos_5[counts[5]:counts[5]+counts[5]]
            trajectories_withrew_for_online = demos_0_withrew[counts[0]:counts[0]+counts[0]] + demos_1_withrew[counts[1]:counts[1]+counts[1]] + demos_2_withrew[counts[2]:counts[2]+counts[2]] + demos_3_withrew[counts[3]:counts[3]+counts[3]] + demos_4_withrew[counts[4]:counts[4]+counts[4]] + demos_5_withrew[counts[5]:counts[5]+counts[5]]
            true_labels_online = np.concatenate((labels_0[counts[0]:counts[0]+counts[0]], labels_1[counts[1]:counts[1]+counts[1]], labels_2[counts[2]:counts[2]+counts[2]], labels_3[counts[3]:counts[3]+counts[3]], labels_4[counts[4]:counts[4]+counts[4]], labels_5[counts[5]:counts[5]+counts[5]]))

                        
        print(f"\n--- Calculating Original Expert Reward Statistics for {name_env} ---")
        if args.Reacherv4 or args.Pusherv4:
            mean_expert_reward, std_expert_reward = calculate_original_expert_reward_stats(trajectories_withrew)                    
        elif args.Traj2d:
            import my_envs.traj2d_gymnasium as traj2d_mod
            env = traj2d_mod.Traj(mode_idx=0)
            mean_expert_reward, std_expert_reward = calculate_expert_reward(demos_0[:counts[0]], env, mode_idx=0, env_name=env_name)

        print(f"Original Expert Reward (from stored .rews): Mean={mean_expert_reward:.4f} ± Std={std_expert_reward:.4f}")

        # true_labels = np.concatenate((labels_0[:counts[0]], labels_1[:counts[1]], labels_2[:counts[2]]))#, labels_3[:counts[3]]))
        # true_labels = np.concatenate((labels_0[:counts[0]], labels_3[:counts[3]], labels_4[:counts[4]]))#, labels_3[:counts[3]]))
        # true_labels_online = np.concatenate((labels_3[:counts[3]], labels_4[:counts[4]], labels_5[:counts[5]])) if args.Reacherv4 else np.concatenate((labels_2[:counts[2]],labels_3[:counts[3]],))
        # true_labels_online = np.concatenate((labels_1[:counts[1]], labels_2[:counts[2]], labels_3[:counts[3]], labels_5[:counts[5]]))
        # true_labels = np.concatenate((true_labels, labels_4[:counts[4]], labels_5[:counts[5]])) if args.Reacherv4 else true_labels
        input_coord_dims = demos_0[0].obs[0].shape[0]
        env_num_step = 1 #unused in SA case
        num_actions = demos_0[0].acts[0].shape[0]
        num_trajs = len(trajectories)
        print(f"Generated {len(trajectories)} expert trajectories for {name_env} with {K} modes.")

    elif args.Humanoidv4 or args.Walker2dv4:
        name_env = "Humanoid-v4" if args.Humanoidv4 else "Walker2d-v4" if args.Walker2dv4 else "Unknown"
        file_path = f"{trajectories_directory_path}expert_imitation_trajectories_{name_env}_mode_0.pkl"
        file_path_withrew = file_path.replace(".pkl", "_withrew.pkl")
        with open(file_path, "rb") as f:
                demos_0 = pickle.load(f)
                labels_0 = np.array([10]*len(demos_0))
                print(f"Loaded {len(demos_0)} expert trajectories for mode 0")
        with open(file_path_withrew, "rb") as f:
                demos_0_withrew = pickle.load(f)
        file_path = f"{trajectories_directory_path}expert_imitation_trajectories_{name_env}_mode_1.pkl"
        file_path_withrew = file_path.replace(".pkl", "_withrew.pkl")
        with open(file_path, "rb") as f:
                demos_1 = pickle.load(f)
                labels_1 = np.array([11]*len(demos_1))
                print(f"Loaded {len(demos_1)} expert trajectories for mode 1")
        with open(file_path_withrew, "rb") as f:
                demos_1_withrew = pickle.load(f)
        file_path = f"{trajectories_directory_path}expert_imitation_trajectories_{name_env}_mode_2.pkl"
        file_path_withrew = file_path.replace(".pkl", "_withrew.pkl")
        with open(file_path, "rb") as f:
                demos_2 = pickle.load(f)
                labels_2 = np.array([12]*len(demos_2))
                print(f"Loaded {len(demos_2)} expert trajectories for mode 2")
        with open(file_path_withrew, "rb") as f:
                demos_2_withrew = pickle.load(f)
        
        K_to_split = K+1 if args.use_seen_unseen_split else K
        counts = compute_mode_counts(num_trajs, K_to_split, ratio)
        print(f"Splitting {num_trajs} trajectories into {K_to_split} modes with ratio {ratio}: {counts}")
        if args.use_seen_unseen_split:
            if args.Humanoidv4:
                if unseen_modes == 1:
                    # trajectories = demos_0[:counts[0]]  + demos_1[:counts[1]] +  demos_2[:counts[2]]
                    trajectories = demos_1[:counts[1]]  +  demos_2[:counts[2]] 
                    # trajectories_withrew = demos_0_withrew[:counts[0]] + demos_1_withrew[:counts[1]] + demos_2_withrew[:counts[2]] 
                    trajectories_withrew = demos_1_withrew[:counts[1]] + demos_2_withrew[:counts[2]] 
                    # true_labels = np.concatenate((labels_0[:counts[0]], labels_1[:counts[1]], labels_2[:counts[2]]))
                    true_labels = np.concatenate((labels_1[:counts[1]], labels_2[:counts[2]]))

                    unseen_trajectories_for_online =  demos_0[counts[0]:counts[0]+counts[0]] + demos_1[counts[1]:counts[1]+counts[1]] + demos_2[counts[2]:counts[2]+counts[2]]
                    trajectories_withrew_for_online = demos_0_withrew[counts[0]:counts[0]+counts[0]] + demos_1_withrew[counts[1]:counts[1]+counts[1]] + demos_2_withrew[counts[2]:counts[2]+counts[2]]
                    true_labels_online = np.concatenate((labels_0[counts[0]:counts[0]+counts[0]], labels_1[counts[1]:counts[1]+counts[1]], labels_2[counts[2]:counts[2]+counts[2]]))
                else:
                    raise ValueError("For Humanoid-v4, when using unseen split, unseen_modes must be 1.")
            elif args.Walker2dv4:
                if unseen_modes == 1: #we keep mode 2 because that's the most confusing one
                    # trajectories = demos_0[:counts[0]]  + demos_1[:counts[1]] +  demos_2[:counts[2]]
                    trajectories = demos_1[:counts[1]]  +  demos_2[:counts[2]] 
                    # trajectories_withrew = demos_0_withrew[:counts[0]] + demos_1_withrew[:counts[1]] + demos_2_withrew[:counts[2]] 
                    trajectories_withrew = demos_1_withrew[:counts[1]] + demos_2_withrew[:counts[2]] 
                    # true_labels = np.concatenate((labels_0[:counts[0]], labels_1[:counts[1]], labels_2[:counts[2]]))
                    true_labels = np.concatenate((labels_1[:counts[1]], labels_2[:counts[2]]))

                    unseen_trajectories_for_online =  demos_0[counts[0]:counts[0]+counts[0]] + demos_1[counts[1]:counts[1]+counts[1]] + demos_2[counts[2]:counts[2]+counts[2]]
                    trajectories_withrew_for_online = demos_0_withrew[counts[0]:counts[0]+counts[0]] + demos_1_withrew[counts[1]:counts[1]+counts[1]] + demos_2_withrew[counts[2]:counts[2]+counts[2]]
                    true_labels_online = np.concatenate((labels_0[counts[0]:counts[0]+counts[0]], labels_1[counts[1]:counts[1]+counts[1]], labels_2[counts[2]:counts[2]+counts[2]]))
                else:
                    raise ValueError("For Walker2d-v4, when using unseen split, unseen_modes must be 1.")
            else:
                raise ValueError("Unsupported environment. Please select either Traj2d, Reacher-v4, Pusher-v4, Walker2d-v4 or Humanoid-v4")
        else:
            trajectories = demos_0[:counts[0]] + demos_1[:counts[1]] + demos_2[:counts[2]] 
            trajectories_withrew = demos_0_withrew[:counts[0]] + demos_1_withrew[:counts[1]] + demos_2_withrew[:counts[2]]
            true_labels = np.concatenate((labels_0[:counts[0]], labels_1[:counts[1]], labels_2[:counts[2]]))

            unseen_trajectories_for_online = demos_0[counts[0]:counts[0]+counts[0]] + demos_1[counts[1]:counts[1]+counts[1]] + demos_2[:counts[2]]
            trajectories_withrew_for_online = demos_0_withrew[counts[0]:counts[0]+counts[0]] + demos_1_withrew[counts[1]:counts[1]+counts[1]] + demos_2_withrew[counts[2]:counts[2]+counts[2]]
            true_labels_online = np.concatenate((labels_0[counts[0]:counts[0]+counts[0]], labels_1[counts[1]:counts[1]+counts[1]], labels_2[counts[2]:counts[2]+counts[2]]))

        print(f"\n--- Calculating Original Expert Reward Statistics for {name_env} ---")
        if args.Reacherv4 or args.Pusherv4 or args.Humanoidv4 or args.Walker2dv4:
            mean_expert_reward, std_expert_reward = calculate_original_expert_reward_stats(trajectories_withrew)                    
        elif args.Traj2d:
            import my_envs.traj2d_gymnasium as traj2d_mod
            env = traj2d_mod.Traj(mode_idx=0)
            mean_expert_reward, std_expert_reward = calculate_expert_reward(demos_0[:counts[0]], env, mode_idx=0, env_name=env_name)

        print(f"Original Expert Reward (from stored .rews): Mean={mean_expert_reward:.4f} ± Std={std_expert_reward:.4f}")

        # true_labels = np.concatenate((labels_0[:counts[0]], labels_1[:counts[1]], labels_2[:counts[2]]))#, labels_3[:counts[3]]))
        # true_labels = np.concatenate((labels_0[:counts[0]], labels_3[:counts[3]], labels_4[:counts[4]]))#, labels_3[:counts[3]]))
        # true_labels_online = np.concatenate((labels_3[:counts[3]], labels_4[:counts[4]], labels_5[:counts[5]])) if args.Reacherv4 else np.concatenate((labels_2[:counts[2]],labels_3[:counts[3]],))
        # true_labels_online = np.concatenate((labels_1[:counts[1]], labels_2[:counts[2]], labels_3[:counts[3]], labels_5[:counts[5]]))
        # true_labels = np.concatenate((true_labels, labels_4[:counts[4]], labels_5[:counts[5]])) if args.Reacherv4 else true_labels
        input_coord_dims = demos_0[0].obs[0].shape[0]
        env_num_step = 1 #unused in SA case
        num_actions = demos_0[0].acts[0].shape[0]
        num_trajs = len(trajectories)
        print(f"Generated {len(trajectories)} expert trajectories for {name_env} with {K} modes.") 

    else:
        raise ValueError("No available environment selected. Please select either --Two Lakes Fishing (-F), --Traj2d(-T2D), --Reacher-v4(-Rv4) or --Pusher-v4(-Pv4) or --Humanoidv4(-Hv4) or --Walker2dv4(-Wv4)")

    print("--- Preparing State-Action Tensors")
    all_states,all_actions,all_masks,all_labels,max_len = prepare_sa_trajectories(env_id,trajectories,true_labels)
    all_states_online,all_actions_online,all_masks_online,all_labels_online,max_len_online = prepare_sa_trajectories(env_id,unseen_trajectories_for_online,true_labels_online)

    if visualize_original or visualize_scalers:
        print("Trajectory flattening . . .")
        traj_flat_seen = [np.array(traj).flatten() for traj in all_states]
        traj_flat_seen_actions = [np.array(traj).flatten() for traj in all_actions]
        traj_flat_seen = np.stack(traj_flat_seen)
        traj_flat_seen_actions = np.stack(traj_flat_seen_actions)

        traj_flat_online = [np.array(traj).flatten() for traj in all_states_online]
        traj_flat_online_actions = [np.array(traj).flatten() for traj in all_actions_online]
        traj_flat_online = np.stack(traj_flat_online)
        traj_flat_online_actions = np.stack(traj_flat_online_actions)

    if visualize_original:

        # Calculate mean intra and inter distances on the flattened states (traj_flat)
        # traj_flat is already standardized and used for UMAP above
        print("Original space visualization . . .")
        features_in_all = traj_flat_seen
        concatenations_labels_distances = np.array(true_labels)
        dists = np.linalg.norm(features_in_all[:, None] - features_in_all[None, :], axis=-1)
        intra = dists[concatenations_labels_distances[:, None] == concatenations_labels_distances[None, :]]
        inter = dists[concatenations_labels_distances[:, None] != concatenations_labels_distances[None, :]]
        print("mean intra (states)", intra.mean(), "mean inter (states)", inter.mean())

        # UMAP projection
        reducer = umap.UMAP(random_state=SEED, n_neighbors=20, min_dist=0.5, n_components=2)
        umap_proj = reducer.fit_transform(traj_flat_seen)

        # Color mapping for true labels
        label_colors = {10: "tab:blue", 11: "tab:red", 12: "tab:green", 13: "tab:purple", 14: "tab:brown", 15: "tab:orange"}
        colors = [label_colors.get(lbl, "gray") for lbl in true_labels]

        plt.figure(figsize=(8, 6))
        for lbl in np.unique(true_labels):
            idx = true_labels == lbl
            plt.scatter(umap_proj[idx, 0], umap_proj[idx, 1], c=label_colors.get(lbl, "gray"), label=f"Mode {lbl-10}", alpha=0.7)
        plt.title(f"2D UMAP projection of original {env_id} trajectory space")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if visualize_scalers:
        print("Classic scalers visualization . . .")
        visualize_classic_scalers_on_flat_states(
            seen_states=traj_flat_seen,
            seen_labels=np.array(true_labels),
            online_states=traj_flat_online,
            online_labels=np.array(true_labels_online),
            seed=SEED,
            n_neighbors=20,
            min_dist=0.5
        )

    # state_scaler = "robust"
    state_scaler = "quantile_normal"
    # action_scaler = "robust"
    action_scaler = "quantile_normal"
    simple_scaler_fit = "seen"
    
    if normalize_input:


        if _is_coord_states(all_states[0]) and state_scaler != 'none':
            ss = _build_scaler(state_scaler)
            print(f"[Simple] Fitting state scaler='{state_scaler}' on {simple_scaler_fit}.")
            ss = _fit_on_flat(all_states, ss, fit_mode=simple_scaler_fit,
                            list_of_td_online=all_states_online if simple_scaler_fit=='both' else None)
            all_states = _apply_per_traj(all_states, ss)
            all_states_online = _apply_per_traj(all_states_online, ss)
            print("[Simple] States scaled.")

        # Actions: only if continuous [T,A]
        if _is_cont_actions(all_actions) and action_scaler != 'none':
            sa = _build_scaler(action_scaler)
            print(f"[Simple] Fitting action scaler='{action_scaler}' on {simple_scaler_fit}.")
            sa = _fit_on_flat(all_actions, sa, fit_mode=simple_scaler_fit,
                            list_of_td_online=all_actions_online if simple_scaler_fit=='both' else None)
            all_actions = _apply_per_traj(all_actions, sa)
            all_actions_online = _apply_per_traj(all_actions_online, sa)
            print("[Simple] Actions scaled.")
    
    print("--- Populating Trajectory Manager ---")
    for i in range(len(trajectories)):
        trajectory_manager[i]= {
            'id': i,
            'original_trajectory': trajectories[i],
            'prepared_states': all_states[i],
            'prepared_actions': all_actions[i],
            'prepared_masks': all_masks[i],
            'real_cluster_label': all_labels[i]
        }
    print("Len trajectory manager:", len(trajectory_manager))

    for i in range(len(unseen_trajectories_for_online)):
        idx = i + len(trajectories)
        trajectory_manager[idx]= {
            'id': idx,
            'original_trajectory': unseen_trajectories_for_online[i],
            'prepared_states': all_states_online[i],
            'prepared_actions': all_actions_online[i],
            'prepared_masks': all_masks_online[i],
            'real_cluster_label': all_labels_online[i]
        }
    print("Len trajectory manager after online:", len(trajectory_manager))

    # obs_shape = all_states[0][0].shape

    obs_shape = trajectories[0].obs[0].shape
    print(f"Generated {len(trajectories)} expert trajectories.")
    transformer_folder = "./methods/transformer_folder"
    model_folder = transformer_folder+f"/models/scaler_{state_scaler}/{env_code}/ntrj_{num_trajs}"
    csv_folder = f"./csvs/{env_id}_CoMIIRL_results"
    csv_file_path = os.path.join(csv_folder, f"{env_code}_ntrj_{num_trajs}_ratio_{ratio}.csv")
    os.makedirs(model_folder, exist_ok=True)
    if saving:
        os.makedirs(csv_folder, exist_ok=True)

    # if os.path.exists(csv_file_path):
    #     df_existing = pd.read_csv(csv_file_path)
    #     results = df_existing.to_dict(orient="records")
    #     print(f"Loaded existing results from {csv_file_path}")
    # else:
    #     results = []

    results = []

    # if any(rec["seed"] == SEED for rec in results):
    #     print(f"  Already have results for seed={SEED}. Exiting.")
    #     # saving = False
    #     exit()

    num_steps = env_num_step
    sos = eos = goal = 1
    max_length = num_steps+goal
    seq_max_len = max_len
    train_size = int(num_trajs*0.9)
    train_size_online = int(len(unseen_trajectories_for_online)*0.9)
    val_size = int(num_trajs*0.0)
    val_size_online = int(len(unseen_trajectories_for_online)*0.0)
    test_size = int(len(trajectories) - train_size - val_size)
    test_size_online = int(len(unseen_trajectories_for_online)- train_size_online - val_size_online)

    ### transformer hyperparameters
    #HYP for Highway and TLF
    # tr_lr = 0.0005 if args.Highway else 0.001 if args.TwoLakesFishing
    # emb_dim = 64 if args.Highway else 10 if args.TwoLakesFishing 
    # num_heads = 16 if args.Highway else 2 if args.TwoLakesFishing
    # nlayers = 6 if args.Highway else 6 if args.TwoLakesFishing
    # d_hid = 2048 if args.Highway else 512 if args.TwoLakesFishing
    tr_lr = 0.0001 if args.Traj2d else 0.0001 if args.Reacherv4 else 0.0001 if args.Pusherv4 else 0.0005
    input_channels = obs_shape[0]
    
    d_model = seq_max_len#40 # not used
    # print("d_model",d_model)
    emb_dim =  12 if args.Traj2d else 32 if args.Reacherv4 else 32 if args.Pusherv4 else 32 if args.Walker2dv4 else 64 if args.Humanoidv4 else 32
    cnn_output_dim = emb_dim
    # projection_dim = 2 #not used by BE, BEwA, NBE, NBEwA
    #only one not defined here is ntokens which is defined after the vocabulary is full
    num_heads = 4 if args.Traj2d else 4 if args.Reacherv4 else 4 if args.Pusherv4 else 4 if args.Walker2dv4 else 4 if args.Humanoidv4 else 4
    nlayers = 2 if args.Traj2d else 2 if args.Reacherv4 else 2 if args.Pusherv4 else 2 if args.Walker2dv4 else 2 if args.Humanoidv4 else 2
    d_hid = 1024 if args.Traj2d else 1024 if args.Reacherv4 else 1024 if args.Pusherv4 else 1024 if args.Walker2dv4 else 1024 if args.Humanoidv4 else 1024
    loader_batch = 64 if args.Traj2d else 64 if args.Reacherv4 else 32 if args.Pusherv4 else 64 if args.Walker2dv4 else 64 if args.Humanoidv4 else 32
    val_bptt = 8
    test_bptt = 1
    dropout = 0.1 if args.Traj2d else 0.1 if args.Reacherv4 else 0.1 if args.Pusherv4 else 0.1 if args.Walker2dv4 else 0.1 if args.Humanoidv4 else 0.1

    #Fourier feature encoding, - gaussian mapping values
    gaussian_m_state = 64 if args.Traj2d else 512 if args.Reacherv4 else 1024 if args.Pusherv4 else 1024 if args.Walker2dv4 else 1024 if args.Humanoidv4 else 1024 #bigger m -> better kernel approximation and cross-dim mixing
    gaussian_m_action = 32 if args.Traj2d else 256 if args.Reacherv4 else 512 if args.Pusherv4 else 512 if args.Walker2dv4 else 512 if args.Humanoidv4 else 512
    gaussian_sigma_state = 10 if args.Traj2d else 5 if args.Reacherv4 else 5 if args.Pusherv4 else 5 if args.Walker2dv4 else 5 if args.Humanoidv4 else 5 #smaller -> high frequency, larger -> smoother features
    gaussian_sigma_action = 10 if args.Traj2d else 5 if args.Reacherv4 else 5 if args.Pusherv4 else 5 if args.Walker2dv4 else 5 if args.Humanoidv4 else 5 #smaller -> high frequency, larger -> smoother features

    bptt = loader_batch

    config_name = f"max_len_{seq_max_len}_seed_{SEED}_ratio_{ratio}_strategy_{embedding_strategy_code}_training_{training_code}"
    config_name = config_name + f"_modes_{unseen_modes}" if args.use_seen_unseen_split else config_name


    print("--- Creating Dataloaders for State-Action Trajectories ---")
    full_dataset, train_dataset, val_dataset, test_dataset, total_dataloader, train_dataloader, val_dataloader, test_dataloader = datasets_preparation_sa(all_states,all_actions,all_masks,true_labels,train_size,val_size,test_size,loader_batch,val_bptt,test_bptt,SEED)
    full_dataset_online, train_dataset_online, val_dataset_online, test_dataset_online, total_dataloader_online, train_dataloader_online, val_dataloader_online, test_dataloader_online = datasets_preparation_sa(all_states_online,all_actions_online,all_masks_online,true_labels_online,train_size_online,val_size_online,test_size_online,loader_batch,val_bptt,test_bptt,SEED)

    print(f"*-*-*-*-*-*-*-*-* Datasets created from {len(full_dataset)} trajectories")
    variational = False
    # Create the model using the factory function for state-action models
    behaviorencoder = create_model_BECwASATyped(
        input_channels=input_channels,
        cnn_output_dim=cnn_output_dim,
        steps=max_len,  # Use the max_len from the data preparation step
        nhead=num_heads,
        d_hid=d_hid,
        emb_dim=emb_dim,
        num_actions=num_actions,
        nlayers=nlayers,
        input_coord_dims=input_coord_dims,
        dropout=dropout,
        gaussian_m_state=gaussian_m_state,
        gaussian_m_action=gaussian_m_action,
        gaussian_sigma_state=gaussian_sigma_state,
        gaussian_sigma_action=gaussian_sigma_action
    )
    
    behaviorencoder.to(device)
    transformer_total_params = sum(p.numel() for p in behaviorencoder.parameters() if p.requires_grad)
    print(behaviorencoder.model_type,"parameters ->",transformer_total_params/1e6,"M")

    # --- ENC-SA Training ---
    print("\n--- ENC-SA Training Selected ---")
    print(f"Embedding strategy: {args.embedding_strategy}")
    loss_type = "ENC_SA_TYPED"
    model_filename = f"/{behaviorencoder.model_type}_{loss_type}_{env_code}_{config_name}.pt"


    if os.path.exists(model_folder + model_filename):
        print(f"Loading existing model: {model_filename}")
        behaviorencoder = th.load(model_folder + model_filename)
        cluster_centroids = th.load(model_folder + model_filename.replace(".pt", "_centroids.pt"))
        # raw_alpha = th.load(model_folder + model_filename.replace(".pt", "_alpha.pt"))
    else:
        print(f"Model not found. Starting ENC-SA training for {model_filename}...")
        training_func = run_encoder_only_training_sa 
        behaviorencoder, _, _, cluster_centroids, _ = training_func(
            env_id=env_id,
            encoder=behaviorencoder,
            dataloader=train_dataloader,
            K=K,
            device=device,
            lr=tr_lr,
            seed=SEED,
            epochs_pre=50,
            epochs_formal=0,
            beta = 0.5 if env_id == "Traj2d" else 0.5 if env_id == "Reacherv4" else 0.5 if env_id == "Pusherv4" else 1.0 if env_id == "Walker2dv4" else 1.0 if env_id == "Humanoidv4" else 0.5, #beta for contrastive
            gamma = 1.0 if env_id == "Traj2d" else 1.0 if env_id == "Reacherv4" else 1.0 if env_id == "Pusherv4" else 1.0 if env_id == "Walker2dv4" else 1.0 if env_id == "Humanoidv4" else 1.0, #gamma for infomax
            # delta = 0.5 if env_id == "Traj2d" else 0.5 if env_id == "Reacherv4" else 0.5 if env_id == "Pusherv4" else 0.5 if env_id == "Walker2dv4" else 0.5 if env_id == "Humanoidv4" else 0.5, #delta for clustering
        )
        # Save the trained components
        th.save(behaviorencoder, model_folder + model_filename)
        th.save(cluster_centroids, model_folder + model_filename.replace(".pt", "_centroids.pt"))
        
    # --- ENC-SA Inference ---
    print("\n--- Running Inference on Dataloaders (ENC-SA) ---")
    behaviorencoder.eval()
    # autoencoder.eval()


    print("Inference on training dataloader...")

    # trajectory_manager, concatenations_seen, indices_seen = \
    #     inference_on_dataloader_cdec_sa(behaviorencoder, cluster_centroids, total_dataloader, trajectory_manager, device) if args.embedding_strategy == "cls_only" else \
    #     inference_on_dataloader_cdec_sa_cat(behaviorencoder, cluster_centroids, total_dataloader, trajectory_manager, device)

    trajectory_manager, concatenations_train, indices_train = \
        inference_on_dataloader_cdec_sa(behaviorencoder, cluster_centroids, train_dataloader, trajectory_manager, device)

    # print("Inference on validation dataloader...")
    # trajectory_manager, concatenations_val, indices_val = \
    #     inference_on_dataloader_cdec_sa(behaviorencoder, cluster_centroids, val_dataloader, trajectory_manager, device)  if args.embedding_strategy == "cls_only" else \
    #     inference_on_dataloader_cdec_sa_cat(behaviorencoder, cluster_centroids, val_dataloader, trajectory_manager, device)

    print("Inference on test dataloader...")
    trajectory_manager, concatenations_test, indices_test = \
        inference_on_dataloader_cdec_sa(behaviorencoder, cluster_centroids, test_dataloader, trajectory_manager, device)

    print("Inference on online dataloader of unseen behaviors...")
    online_indices_offset = len(trajectories)
    trajectory_manager, concatenations_online, indices_online = \
        inference_on_dataloader_cdec_sa(behaviorencoder, cluster_centroids, total_dataloader_online, trajectory_manager, device, index_offset=online_indices_offset)

    # --- Process Embeddings for Visualization ---
    print("\n--- Processing Embeddings for Visualization ---")
    if not variational:
        tj_embeddings_train_true_labels = [trajectory_manager[i]['real_cluster_label'].item() for i in indices_train]
        # tj_embeddings_val_true_labels = [trajectory_manager[i]['real_cluster_label'].item() for i in indices_val]
        tj_embeddings_test_true_labels = [trajectory_manager[i]['real_cluster_label'].item() for i in indices_test]
        # tj_embeddings_seen_true_labels = [trajectory_manager[i]['real_cluster_label'].item() for i in indices_seen]
        tj_embeddings_online_true_labels = [trajectory_manager[i]['real_cluster_label'].item() for i in indices_online]

        tj_embeddings_train_pred_labels = [trajectory_manager[i]['predicted_cluster_label'] for i in indices_train]
        # tj_embeddings_val_pred_labels = [trajectory_manager[i]['predicted_cluster_label'] for i in indices_val]
        tj_embeddings_test_pred_labels = [trajectory_manager[i]['predicted_cluster_label'] for i in indices_test]
        # tj_embeddings_seen_pred_labels = [trajectory_manager[i]['predicted_cluster_label'] for i in indices_seen]
        tj_embeddings_online_pred_labels = [trajectory_manager[i]['predicted_cluster_label'] for i in indices_online]

        tj_concatenations_train = concatenations_train.squeeze(1).cpu().numpy()
        # tj_concatenations_val = concatenations_val.squeeze(1).cpu().numpy()
        tj_concatenations_test = concatenations_test.squeeze(1).cpu().numpy()
        # tj_concatenations_seen = concatenations_seen.squeeze(1).cpu().numpy()
        tj_concatenations_online = concatenations_online.squeeze(1).cpu().numpy()

        tj_concatenations_evaluation = np.vstack([tj_concatenations_test])
        tj_embeddings_evaluation_true_labels = np.hstack([tj_embeddings_test_true_labels])
        tj_embeddings_evaluations_pred_labels = np.hstack([tj_embeddings_test_pred_labels])
        tj_concatenations_online_evaluation = np.vstack([tj_concatenations_online])
        tj_embeddings_evaluation_true_labels_with_online = np.hstack([tj_embeddings_evaluation_true_labels, tj_embeddings_online_true_labels])
        tj_embeddings_evaluations_pred_labels_with_online = np.hstack([tj_embeddings_evaluations_pred_labels, tj_embeddings_online_pred_labels])

        tj_concatenations_seen = np.vstack([tj_concatenations_train, tj_concatenations_evaluation])
        tj_embeddings_seen_true_labels = np.hstack([tj_embeddings_train_true_labels, tj_embeddings_evaluation_true_labels])
        tj_embeddings_seen_pred_labels = np.hstack([tj_embeddings_train_pred_labels, tj_embeddings_evaluations_pred_labels])
        tj_concatenations_seen_true_labels = tj_embeddings_seen_true_labels
        tj_concatenations_seen_true_labels_with_online = np.hstack([tj_embeddings_seen_true_labels, tj_embeddings_online_true_labels])
        tj_concatenations_only_online = tj_concatenations_online_evaluation
        tj_concatenations_seen_with_online = np.vstack([tj_concatenations_seen, tj_concatenations_only_online])   
    else:
        pass
    # --- Visualization and Metrics ---

    if args.Traj2d:
            # nplabels = np.array(["Mode 0" if x == 10 else "Mode 1" if x == 11 else "Mode 2" if x == 12 else "Mode 3" if x == 13 else "???" for x in tj_embeddings_train_true_labels])
            # nplabels_test = np.array(["Mode 0" if x == 10 else "Mode 1" if x == 11 else "Mode 2" if x == 12 else "Mode 3" if x == 13 else "???" for x in tj_embeddings_test_true_labels])
            # colors_train = ["tab:blue" if x == 10 else "tab:red" if x == 11 else "tab:green" if x == 12 else "tab:purple" if x == 13 else "y" for x in tj_embeddings_train_true_labels]
            # colors_test = ["tab:blue" if x == 10 else "tab:red" if x == 11 else "tab:green" if x == 12 else "tab:purple" if x == 13  else "y" for x in tj_embeddings_test_true_labels]
            # colors_evaluation = ["tab:blue" if x == 10 else "tab:red" if x == 11 else "tab:green" if x == 12 else "tab:purple" if x == 13  else "y" for x in tj_embeddings_evaluation_true_labels]
            # colors_all = ["tab:blue" if x == 10 else "tab:red" if x == 11 else "tab:green" if x == 12 else "tab:purple" if x == 13  else "y" for x in tj_embeddings_all_true_labels]
            color_label_mapping = {
                "tab:blue": "Mode 0",
                "tab:red": "Mode 1",
                "tab:green": "Mode 2",
                "tab:purple": "Mode 3",
                "tab:brown": "Mode 4",
                "tab:orange": "Mode 5",
                "y": "???"
            }
            # colors_predicted = ["b" if x == 0 else "r" if x == 1 else "g" for x in unique_predicted_labels]
    if args.Reacherv4 or args.Pusherv4:
        color_label_mapping = {
                "tab:blue": "Mode 0",
                "tab:red": "Mode 1",
                "tab:green": "Mode 2",
                "tab:purple": "Mode 3",
                "tab:brown": "Mode 4",
                "tab:orange": "Mode 5",
                "y": "???"
            }
    if args.Humanoidv4 or args.Walker2dv4:
        color_label_mapping = {
            "tab:blue": "Mode 0",
            "tab:red": "Mode 1",
            "tab:green": "Mode 2",
        }
    # elif args.Highway:#future studies
    #     if behaviorencoder.model_type != "NCBEwA" or behaviorencoder.model_type != "NCBEwACLS":
    #         # nplabels = np.array(["lane_keeping" if x == 5  else "speed_focused" if x == 6 else "safety_focused" if x == 7 else "B4" if x == 8 else "B5" if x == 9 else "B6" if x == 10  else "???" for x in tj_embeddings_train_true_labels])
    #         # nplabels_test = np.array(["lane_keeping" if x == 5  else "speed_focused" if x == 6 else "safety_focused" if x == 7 else "B4" if x == 8 else "B5" if x == 9 else "B6" if x == 10  else "???" for x in tj_embeddings_test_true_labels])
    #         # colors_train = ["tab:blue" if x == 5 else "tab:orange" if x == 6 else "tab:green" if x == 7 else "tab:red" if x == 8 else "tab:purple" if x == 9 else "tab:brown" if x == 10 else "b" for x in tj_embeddings_train_true_labels]
    #         # colors_test = ["tab:blue" if x == 5 else "tab:orange" if x == 6 else "tab:green" if x == 7 else "tab:red" if x == 8 else "tab:purple" if x == 9 else "tab:brown" if x == 10 else "b" for x in tj_embeddings_test_true_labels]
    #         # colors_evaluation = ["tab:blue" if x == 5 else "tab:orange" if x == 6 else "tab:green" if x == 7 else "tab:red" if x == 8 else "tab:purple" if x == 9 else "tab:brown" if x == 10 else "b" for x in tj_embeddings_evaluation_true_labels]
    #         # colors_all = ["tab:blue" if x == 5 else "tab:orange" if x == 6 else "tab:green" if x == 7 else "tab:red" if x == 8 else "tab:purple" if x == 9 else "tab:brown" if x == 10 else "b" for x in tj_embeddings_all_true_labels]
    #         color_label_mapping = {
    #             "tab:blue": "lane_keeping",
    #             "tab:orange": "speed_focused",
    #             "tab:green": "safety_focused",
    #             "tab:red": "B4",
    #             "tab:purple": "B5",
    #             "tab:brown": "B6",
    #             "b": "???"
    #         }
    #         # colors_predicted = ["tab:blue" if x == 0 else "tab:orange" if x == 1 else "tab:green" if x == 3 else "tab:red" if x == 4 else "tab:purple" if x == 5 else "tab:brown" if x == 6 else "b" for x in unique_predicted_labels]

    print("\n--- Generating Visualizations and Metrics ---")
    # This visualization block is the same as the one you use for your other models.
    check_unseen = args.use_seen_unseen_split
    inductive_clustering = False
    lof_check = False
    knn_density_check = False
    evaluation_metrics_clustering = False
    prefix_testing = False
    meanshift = False
    hdb = False
    gmm = False                   # Gaussian Mixture with diag cov
    controller = True


    if state_scaler == "quantile_normal":
        if check_unseen:
            granularity = 0.1 if args.Traj2d else 0.075 if args.Reacherv4 else 0.05 if args.Pusherv4  else 0.03 if args.Walker2dv4 else 0.1 if args.Humanoidv4 else 0.02
            if unseen_modes == 1:
                quantile_ms = 0.095 if args.Traj2d else 0.1 if args.Reacherv4 else 0.1 if args.Pusherv4 else 0.1 if args.Walker2dv4 else 0.2 if args.Humanoidv4 else 0.02
                granularity = 0.1 if args.Traj2d else 0.075 if args.Reacherv4 else 0.05 if args.Pusherv4  else 0.05 if args.Walker2dv4 else 0.1 if args.Humanoidv4 else 0.02
            elif unseen_modes == 2:
                quantile_ms = 0.095 if args.Traj2d else 0.1 if args.Reacherv4 else 0.1 if args.Pusherv4 else 0.15
                granularity = 0.1 if args.Traj2d else 0.075 if args.Reacherv4 else 0.05 if args.Pusherv4  else 0.1
            elif unseen_modes== 3:
                quantile_ms = 0.095 if args.Traj2d else 0.1 if args.Reacherv4 else 0.1 if args.Pusherv4 else 0.15
                granularity = 0.1 if args.Traj2d else 0.075 if args.Reacherv4 else 0.05 if args.Pusherv4  else 0.1
        else:
            granularity = 0.1 if args.Traj2d else 0.05 if args.Reacherv4 else 0.1 if args.Pusherv4 else 0.15 if args.Walker2dv4 else 0.15 if args.Humanoidv4 else 0.02
            quantile_ms = 0.095
    elif state_scaler == "robust":
        if check_unseen:
            granularity = 0.1 if args.Traj2d else 0.04 if args.Reacherv4 else 0.05 if args.Pusherv4  else 0.03 if args.Walker2dv4 else 0.1 if args.Humanoidv4 else 0.02
        else:
            granularity = 0.1 if args.Traj2d else 0.05 if args.Reacherv4 else 0.1 if args.Pusherv4 else 0.075 if args.Walker2dv4 else 0.135 if args.Humanoidv4 else 0.02


    reducer = umap.UMAP(
            random_state=SEED,
            n_neighbors=100,
            min_dist=0.5,
            n_components=3 if args.threeD else 2,
            metric='cosine',
        )

    if not check_unseen:
        umap_combined = reducer.fit_transform(tj_concatenations_seen)
    else:
        umap_combined = reducer.fit_transform(tj_concatenations_seen_with_online)
    Z_train = umap_combined[:len(tj_concatenations_train)]

    Z_evaluation  = umap_combined[len(tj_concatenations_train) :len(tj_concatenations_train) + len(tj_concatenations_evaluation)]
    Z_online = umap_combined[len(tj_concatenations_train) + len(tj_concatenations_evaluation):]

    if not check_unseen:
        indices_all = np.hstack([indices_train, indices_test])
    else:
        indices_all = np.hstack([indices_train, indices_test, indices_online]) 

    ## VISUALIZATION INCLUDING NOVELTY DETECTION IF -split True

    #LOF

    if lof_check:
        print("\n--- Density-Based Novelty Detection with LocalOutlierFactor ---")
        k_neighbors = 20  

        # Fit LOF on training data
        lof = LocalOutlierFactor(n_neighbors=k_neighbors, novelty=True,)
        lof.fit(tj_concatenations_seen)

        # Predict on online data
        lof_labels = lof.predict(tj_concatenations_online)  # -1 for outlier, 1 for inlier
        lof_scores = lof.decision_function(tj_concatenations_online)  # The lower, the more abnormal

        is_novel_lof = lof_labels == -1
        novel_indices_lof = np.where(is_novel_lof)[0]
        novel_points = tj_concatenations_online[novel_indices_lof]
        print(f"Number of novel points detected by LOF: {len(novel_indices_lof)} (out of {len(tj_concatenations_online)})")
        # Step 3: Buffer and validate novel points with MeanShift
        novel_buffer_lof = novel_points.tolist()  # Convert to list for buffering
        min_buffer_size = 5  # Minimum number of points to form a new cluster
        new_centroids = []  # To store new cluster centroids if detected
        if meanshift:
            if len(novel_buffer_lof) >= min_buffer_size:
                bandwidth = estimate_bandwidth(novel_buffer_lof, quantile=0.25)  # Same as Traj2d setting
                ms = MeanShift(bandwidth=bandwidth).fit(novel_buffer_lof)
                n_clusters = len(np.unique(ms.labels_))
                if n_clusters > 0 and -1 not in ms.labels_:  # Ensure no noise points
                    new_centroids = ms.cluster_centers_
                    print(f"New cluster detected with {n_clusters} centroid(s)")
                else:
                    print("Novel points do not form a coherent cluster yet")
            else:
                print(f"Not enough novel points ({len(novel_buffer_lof)}/{min_buffer_size}) to form a new cluster")
        elif hdb:
            novel_buffer_lof = novel_points
            if len(novel_buffer_lof) >= min_buffer_size:
                hdb_micro = hdbscan.HDBSCAN(
                    min_cluster_size=min_buffer_size,
                    min_samples=max(1, int(math.sqrt(min_buffer_size))),
                    metric='euclidean'
                ).fit(novel_buffer_lof)
                labels_micro = hdb_micro.labels_
                core_clusters = np.unique(labels_micro[labels_micro != -1])
                n_clusters = len(core_clusters)
                if n_clusters > 0:
                    # median per detected micro-cluster
                    for c in core_clusters:
                        pts = novel_buffer_lof[labels_micro == c]
                        new_centroids.append(np.median(pts, axis=0))
                    new_centroids = np.vstack(new_centroids)
                    print(f"New cluster detected with {n_clusters} centroid(s)")
                else:
                    print("Novel points do not form a coherent cluster yet")
            else:
                print(f"Not enough novel points ({len(novel_buffer_lof)}/{min_buffer_size}) to form a new cluster")

        # --- Plot 1: Train (true) + Unassigned Test + Online with Novelty ---
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d' if args.threeD else None)
        legend_handles = []

        # Plot train data by true label
        for label_val in np.unique(tj_embeddings_train_true_labels):
            props = get_label_display_properties(HIGH_CONTRAST_PREDICTED_COLORS, label_val, args, color_label_mapping, "true")
            mask = tj_embeddings_train_true_labels == label_val
            if args.threeD:
                h = ax.scatter(Z_train[mask, 0], Z_train[mask, 1], Z_train[mask, 2],
                            c=props['color'], s=15, alpha=0.7, marker='o', label=f"Train - {props['name']}")
            else:
                h = ax.scatter(Z_train[mask, 0], Z_train[mask, 1],
                            c=props['color'], s=15, alpha=0.7, marker='o', label=f"Train - {props['name']}")
            if mask.any():
                legend_handles.append(h)

        # Plot test/eval data by true label
        for label_val in np.unique(tj_embeddings_evaluation_true_labels):
            props = get_label_display_properties(HIGH_CONTRAST_PREDICTED_COLORS, label_val, args, color_label_mapping, "true")
            mask = tj_embeddings_evaluation_true_labels == label_val
            if args.threeD:
                h = ax.scatter(Z_evaluation[mask, 0], Z_evaluation[mask, 1], Z_evaluation[mask, 2],
                            c=props['color'], s=30, alpha=0.9, marker='^', label=f"Eval - {props['name']}")
            else:
                h = ax.scatter(Z_evaluation[mask, 0], Z_evaluation[mask, 1],
                            c=props['color'], s=30, alpha=0.9, marker='^', label=f"Eval - {props['name']}")
            if mask.any():
                legend_handles.append(h)

        # Plot online data by true label and novelty status
        if check_unseen:
            Z_online = reducer.transform(tj_concatenations_online)
            for label_val in np.unique(tj_embeddings_online_true_labels):
                props = get_label_display_properties(HIGH_CONTRAST_PREDICTED_COLORS, label_val, args, color_label_mapping, "true")
                mask = tj_embeddings_online_true_labels == label_val
                # Assigned (not novel)
                assigned_mask = mask & (~is_novel_lof)
                if np.any(assigned_mask):
                    if args.threeD:
                        h = ax.scatter(Z_online[assigned_mask, 0], Z_online[assigned_mask, 1], Z_online[assigned_mask, 2],
                                    c=props['color'], s=50, alpha=1.0, marker='s', label=f"Online Assigned - {props['name']}")
                    else:
                        h = ax.scatter(Z_online[assigned_mask, 0], Z_online[assigned_mask, 1],
                                    c=props['color'], s=50, alpha=1.0, marker='s', label=f"Online Assigned - {props['name']}")
                    legend_handles.append(h)
                # Novel
                novel_mask = mask & is_novel_lof
                if np.any(novel_mask):
                    if args.threeD:
                        h = ax.scatter(Z_online[novel_mask, 0], Z_online[novel_mask, 1], Z_online[novel_mask, 2],
                                    c=props['color'], s=70, alpha=1.0, marker='*', label=f"Online Novel - {props['name']}")
                    else:
                        h = ax.scatter(Z_online[novel_mask, 0], Z_online[novel_mask, 1],
                                    c=props['color'], s=70, alpha=1.0, marker='*', label=f"Online Novel - {props['name']}")
                    legend_handles.append(h)

        ax.set_title(f"{'3D' if args.threeD else '2D'} UMAP projection of {env_id} embedding space: {'Train/Test/Online' if check_unseen else 'Train/Test'} split")
        ax.legend(handles=legend_handles, loc='best')
        plt.show()
        plt.close()
    
    #MANUAL

    if knn_density_check:
        print("\n--- Setting Up Density-Based Novelty Detection ---")

        # Step 1: Compute density baseline from training data
        k_neighbors = 10  # Number of neighbors for density estimation, adjust as needed
        nn = NearestNeighbors(n_neighbors=k_neighbors + 1)  # +1 to exclude the point itself
        nn.fit(tj_concatenations_seen)
        distances, _ = nn.kneighbors(tj_concatenations_seen)

        # Local density is inversely proportional to the average distance to k-nearest neighbors
        # Use the average distance to k neighbors as a proxy for density (smaller distance = higher density)
        avg_distances_train = np.mean(distances[:, 1:], axis=1)  # Exclude distance to self
        densities_train = 1.0 / (avg_distances_train + 1e-8)  # Inverse distance, avoid division by zero

        # Compute density threshold (mean - 2 * std of training densities)
        mean_density = np.mean(densities_train)
        std_density = np.std(densities_train)
        density_threshold = mean_density - std_density
        print(f"Density Threshold: {density_threshold:.4f} (mean: {mean_density:.4f}, std: {std_density:.4f})")

        # Step 2: Evaluate density of online points
        nn_online = NearestNeighbors(n_neighbors=k_neighbors + 1)  # Same k as above
        nn_online.fit(tj_concatenations_train)  # Fit on training data to compare online points
        distances_online, _ = nn_online.kneighbors(tj_concatenations_online)
        avg_distances_online = np.mean(distances_online[:, 1:], axis=1)
        densities_online = 1.0 / (avg_distances_online + 1e-8)

        # Flag points with density below threshold as novel
        is_novel = densities_online < density_threshold
        novel_indices = np.where(is_novel)[0]
        novel_points = tj_concatenations_online[novel_indices]
        print(f"Number of novel points detected: {len(novel_indices)}")

        # Step 3: Buffer and validate novel points with MeanShift
        novel_buffer = novel_points.tolist()  # Convert to list for buffering
        min_buffer_size = 5  # Minimum number of points to form a new cluster
        new_centroids = []  # To store new cluster centroids if detected
        if meanshift:
            if len(novel_buffer) >= min_buffer_size:
                bandwidth = estimate_bandwidth(novel_buffer, quantile=0.25)  # Same as Traj2d setting
                ms = MeanShift(bandwidth=bandwidth).fit(novel_buffer)
                n_clusters = len(np.unique(ms.labels_))
                if n_clusters > 0 and -1 not in ms.labels_:  # Ensure no noise points
                    new_centroids = ms.cluster_centers_
                    print(f"New cluster detected with {n_clusters} centroid(s)")
                else:
                    print("Novel points do not form a coherent cluster yet")
            else:
                print(f"Not enough novel points ({len(novel_buffer)}/{min_buffer_size}) to form a new cluster")
        elif hdb:
            novel_buffer = novel_points
            if len(novel_buffer) >= min_buffer_size:
                hdb_micro = HDBSCAN(
                    min_cluster_size=min_buffer_size,
                    min_samples=max(1, int(math.sqrt(min_buffer_size))),
                    metric='euclidean'
                ).fit(novel_buffer)
                labels_micro = hdb_micro.labels_
                core_clusters = np.unique(labels_micro[labels_micro != -1])
                n_clusters = len(core_clusters)
                if n_clusters > 0:
                    for c in core_clusters:
                        pts = novel_buffer[labels_micro == c]
                        new_centroids.append(np.median(pts, axis=0))
                    new_centroids = np.vstack(new_centroids)
                    print(f"New cluster detected with {n_clusters} centroid(s)")
                else:
                    print("Novel points do not form a coherent cluster yet")
            else:
                print(f"Not enough novel points ({len(novel_buffer)}/{min_buffer_size}) to form a new cluster")

        # --- Plot 1: Train (true) + Unassigned Test + Online with Novelty ---
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d' if args.threeD else None)
        legend_handles = []

        # Plot train data by true label
        for label_val in np.unique(tj_embeddings_train_true_labels):
            props = get_label_display_properties(HIGH_CONTRAST_PREDICTED_COLORS, label_val, args, color_label_mapping, "true")
            mask = tj_embeddings_train_true_labels == label_val
            if args.threeD:
                h = ax.scatter(Z_train[mask, 0], Z_train[mask, 1], Z_train[mask, 2],
                            c=props['color'], s=15, alpha=0.7, marker='o', label=f"Train - {props['name']}")
            else:
                h = ax.scatter(Z_train[mask, 0], Z_train[mask, 1],
                            c=props['color'], s=15, alpha=0.7, marker='o', label=f"Train - {props['name']}")
            if mask.any():
                legend_handles.append(h)

        # Plot test/eval data by true label
        for label_val in np.unique(tj_embeddings_evaluation_true_labels):
            props = get_label_display_properties(HIGH_CONTRAST_PREDICTED_COLORS, label_val, args, color_label_mapping, "true")
            mask = tj_embeddings_evaluation_true_labels == label_val
            if args.threeD:
                h = ax.scatter(Z_evaluation[mask, 0], Z_evaluation[mask, 1], Z_evaluation[mask, 2],
                            c=props['color'], s=30, alpha=0.9, marker='^', label=f"Eval - {props['name']}")
            else:
                h = ax.scatter(Z_evaluation[mask, 0], Z_evaluation[mask, 1],
                            c=props['color'], s=30, alpha=0.9, marker='^', label=f"Eval - {props['name']}")
            if mask.any():
                legend_handles.append(h)

        # Plot online data by true label and novelty status
        if check_unseen:
            Z_online = reducer.transform(tj_concatenations_online)
            for label_val in np.unique(tj_embeddings_online_true_labels):
                props = get_label_display_properties(HIGH_CONTRAST_PREDICTED_COLORS, label_val, args, color_label_mapping, "true")
                mask = tj_embeddings_online_true_labels == label_val
                # Assigned (not novel)
                assigned_mask = mask & (~is_novel)
                if np.any(assigned_mask):
                    if args.threeD:
                        h = ax.scatter(Z_online[assigned_mask, 0], Z_online[assigned_mask, 1], Z_online[assigned_mask, 2],
                                    c=props['color'], s=50, alpha=1.0, marker='s', label=f"Online Assigned - {props['name']}")
                    else:
                        h = ax.scatter(Z_online[assigned_mask, 0], Z_online[assigned_mask, 1],
                                    c=props['color'], s=50, alpha=1.0, marker='s', label=f"Online Assigned - {props['name']}")
                    legend_handles.append(h)
                # Novel
                novel_mask = mask & is_novel
                if np.any(novel_mask):
                    if args.threeD:
                        h = ax.scatter(Z_online[novel_mask, 0], Z_online[novel_mask, 1], Z_online[novel_mask, 2],
                                    c=props['color'], s=70, alpha=1.0, marker='*', label=f"Online Novel - {props['name']}")
                    else:
                        h = ax.scatter(Z_online[novel_mask, 0], Z_online[novel_mask, 1],
                                    c=props['color'], s=70, alpha=1.0, marker='*', label=f"Online Novel - {props['name']}")
                    legend_handles.append(h)

        # # Plot learned centroids
        # if args.threeD:
        #     ax.scatter(umap_centroids[:, 0], umap_centroids[:, 1], umap_centroids[:, 2],
        #             c='red', marker='X', s=250, edgecolor='black', label='Learned Centroids', zorder=10)
        # else:
        #     ax.scatter(umap_centroids[:, 0], umap_centroids[:, 1],
        #             c='red', marker='X', s=250, edgecolor='black', label='Learned Centroids', zorder=10)

        # Plot new centroids from novel points (if any)
        # if len(new_centroids) > 0:
        #     umap_new_centroids = reducer.transform(new_centroids)
        #     if args.threeD:
        #         ax.scatter(umap_new_centroids[:, 0], umap_new_centroids[:, 1], umap_new_centroids[:, 2],
        #                 c='blue', marker='D', s=200, edgecolor='black', label='New Centroids', zorder=10)
        #     else:
        #         ax.scatter(umap_new_centroids[:, 0], umap_new_centroids[:, 1],
        #                 c='blue', marker='D', s=200, edgecolor='black', label='New Centroids', zorder=10)
        #     legend_handles.append(ax.scatter([], [], c='blue', marker='D', s=200, label='New Centroids'))

        ax.set_title(f"{'3D' if args.threeD else '2D'} UMAP projection of {env_id} embedding space: {'Train/Test/Online' if check_unseen else 'Train/Test'} split")
        ax.legend(handles=legend_handles, loc='best')
        plt.show()
        plt.close()

        # exit()

    if inductive_clustering:
        X_seen = tj_concatenations_seen
        X_online = tj_concatenations_online if check_unseen else np.zeros((0, X_seen.shape[1]))

        # L2-normalize to emulate cosine distance with Euclidean
        l2_norm = Normalizer(norm='l2')
        X_seen_n = l2_norm.fit_transform(X_seen)
        X_online_n = l2_norm.transform(X_online) if len(X_online) > 0 else X_online

        # proportion-based params on seen size
        n_seen = len(X_seen_n)
        min_cluster_size = max(5, int(granularity * n_seen))
        min_samples = max(1, int(math.sqrt(min_cluster_size)))

        # Fit HDBSCAN (euclidean on unit vectors ≈ cosine)
        hdb_model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            prediction_data=True
        ).fit(X_seen_n)

        labels_seen = hdb_model.labels_

        # Inductive assignment for online (no refit)
        if check_unseen and len(X_online_n) > 0:
            labels_online, strengths = hdbscan.approximate_predict(hdb_model, X_online_n)

            # Calibrate τ from seen core probabilities (fallback to fixed)
            tau_strength = 0.15
            if getattr(hdb_model, "probabilities_", None) is not None and np.any(labels_seen != -1):
                tau_strength = float(np.quantile(hdb_model.probabilities_[labels_seen != -1], 0.05))

            if 'is_novel_lof' in locals():
                labels_online = np.where(is_novel_lof | (strengths < tau_strength), -1, labels_online)
            else:
                labels_online = np.where(strengths < tau_strength, -1, labels_online)
        else:
            labels_online = np.zeros((0,), dtype=int)

        # Final inductive labels (seen then online) — keep local name to avoid UnboundLocalError later
        cluster_labels_ind = np.concatenate([labels_seen, labels_online])
        unique_core = np.unique(cluster_labels_ind[cluster_labels_ind != -1])
        n_clusters_ = len(unique_core)
        print(f"[HDBSCAN-inductive] core clusters (excluding noise): {n_clusters_}")

        # --- NMI metrics (inductive) ---
        try:
            # ground-truth in same order: seen then online
            y_true_all_ind = (tj_embeddings_seen_true_labels if not check_unseen
                                else np.hstack([tj_embeddings_seen_true_labels, tj_embeddings_online_true_labels]))

            # NMI over all labels (treat -1 as its own class)
            nmi_all = (normalized_mutual_info_score(y_true_all_ind, cluster_labels_ind)
                        if len(np.unique(y_true_all_ind)) > 1 and len(np.unique(cluster_labels_ind)) > 1 else np.nan)

            # NMI over core points only (exclude noise)
            mask_core = cluster_labels_ind != -1
            nmi_core = (normalized_mutual_info_score(y_true_all_ind[mask_core], cluster_labels_ind[mask_core])
                        if mask_core.sum() >= 2 and
                            len(np.unique(y_true_all_ind[mask_core])) > 1 and
                            len(np.unique(cluster_labels_ind[mask_core])) > 1
                        else np.nan)

            print(f"[Inductive NMI] all={nmi_all if not np.isnan(nmi_all) else 'nan'}  core={nmi_core if not np.isnan(nmi_core) else 'nan'}")

            # Online-only NMIs
            if check_unseen and len(labels_online) > 0:
                nmi_online_all = (normalized_mutual_info_score(tj_embeddings_online_true_labels, labels_online)
                                    if len(np.unique(tj_embeddings_online_true_labels)) > 1 and
                                        len(np.unique(labels_online)) > 1 else np.nan)

                mask_online_core = labels_online != -1
                nmi_online_core = (normalized_mutual_info_score(tj_embeddings_online_true_labels[mask_online_core],
                                                                labels_online[mask_online_core])
                                    if mask_online_core.sum() >= 2 and
                                        len(np.unique(tj_embeddings_online_true_labels[mask_online_core])) > 1 and
                                        len(np.unique(labels_online[mask_online_core])) > 1
                                    else np.nan)

                print(f"[Inductive NMI] online(all)={nmi_online_all if not np.isnan(nmi_online_all) else 'nan'}  "
                        f"online(core)={nmi_online_core if not np.isnan(nmi_online_core) else 'nan'}")

                # How many kNN-novel are flagged as HDBSCAN noise
                if 'is_novel' in locals():
                    n_noise = int((labels_online == -1).sum())
                    n_novel = int(is_novel.sum())
                    overlap = int(((labels_online == -1) & is_novel).sum())
                    print(f"[Inductive] online noise/total={n_noise}/{len(labels_online)} | "
                            f"kNN-novel/total={n_novel}/{len(labels_online)} | overlap={overlap}")
        except Exception as e:
            print(f"[Inductive NMI] failed: {e}")

        # --- Visualization: Inductive HDBSCAN assignments on UMAP ---
        try:
            # sanity prints
            seen_counts = np.unique(labels_seen, return_counts=True)
            print(f"[Inductive] seen label counts: {seen_counts}")
            if check_unseen and len(labels_online) > 0:
                online_counts = np.unique(labels_online, return_counts=True)
                print(f"[Inductive] online label counts: {online_counts}")

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d' if args.threeD else None)

            train_len = len(Z_train)
            eval_len = len(Z_evaluation)
            online_len = len(Z_online)

            lbl_train = cluster_labels_ind[:train_len]
            lbl_eval = cluster_labels_ind[train_len:train_len + eval_len]
            lbl_online = cluster_labels_ind[train_len + eval_len : train_len + eval_len + online_len]

            def _scatter_split(Z, labels, marker, name):
                if Z is None or len(Z) == 0 or len(labels) == 0:
                    return
                for lab in np.unique(labels):
                    m = labels == lab
                    if not np.any(m):
                        continue
                    if lab == -1:
                        color = 'k'
                        mk = 'x'
                        label_txt = f"{name} noise"
                    else:
                        color = plt.cm.tab10(int(lab) % 10)
                        mk = marker
                        label_txt = f"{name} C{int(lab)}"
                    if args.threeD:
                        ax.scatter(Z[m, 0], Z[m, 1], Z[m, 2],
                                    c=[color],
                                    s=50 if name != "Train" else 20,
                                    alpha=0.9 if name != "Train" else 0.7,
                                    marker=mk, label=label_txt)
                    else:
                        ax.scatter(Z[m, 0], Z[m, 1],
                                    c=[color],
                                    s=50 if name != "Train" else 20,
                                    alpha=0.9 if name != "Train" else 0.7,
                                    marker=mk, label=label_txt)

            _scatter_split(Z_train, lbl_train, 'o', 'Train')
            _scatter_split(Z_evaluation, lbl_eval, '^', 'Eval')
            if check_unseen and online_len > 0:
                _scatter_split(Z_online, lbl_online, 's', 'Online')

            ax.set_title(f"{'3D' if args.threeD else '2D'} UMAP: Inductive HDBSCAN (seen fit, novelty-gated)")
            ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
            if args.threeD:
                ax.set_zlabel("UMAP-3")
            ax.legend(loc='best', fontsize=8)
            plt.tight_layout()
            plt.show()
            plt.close()
        except Exception as e:
            print(f"[Inductive Viz] failed: {e}")
    
    if evaluation_metrics_clustering:

        novelty_results = {}
        train_label_set = set(np.unique(tj_embeddings_train_true_labels))
        y_online_true_novel = np.array([1 if lab not in train_label_set else 0 for lab in tj_embeddings_online_true_labels])  # 1 = novel

        # (A) LOF (if available)
        if 'lof_scores' in locals():
            lof_anomaly_scores = -lof_scores  # higher => more novel
            try:
                novelty_results['lof_auc'] = roc_auc_score(y_online_true_novel, lof_anomaly_scores)
                novelty_results['lof_aupr'] = average_precision_score(y_online_true_novel, lof_anomaly_scores)
                y_pred_lof = (lof_labels == -1).astype(int)
                novelty_results['lof_precision'] = precision_score(y_online_true_novel, y_pred_lof, zero_division=0)
                novelty_results['lof_recall'] = recall_score(y_online_true_novel, y_pred_lof, zero_division=0)
                novelty_results['lof_f1'] = f1_score(y_online_true_novel, y_pred_lof, zero_division=0)
                novelty_results['lof_tp'] = int(((y_pred_lof == 1) & (y_online_true_novel == 1)).sum())
                novelty_results['lof_fp'] = int(((y_pred_lof == 1) & (y_online_true_novel == 0)).sum())
                novelty_results['lof_fn'] = int(((y_pred_lof == 0) & (y_online_true_novel == 1)).sum())
            except ValueError:
                for k in ['lof_auc','lof_aupr','lof_precision','lof_recall','lof_f1']:
                    novelty_results[k] = np.nan

        # (B) kNN density (already computed): lower density => more novel
        if 'densities_online' in locals():
            density_anomaly_scores = -densities_online  # higher => more novel
            try:
                novelty_results['knn_density_auc'] = roc_auc_score(y_online_true_novel, density_anomaly_scores)
                novelty_results['knn_density_aupr'] = average_precision_score(y_online_true_novel, density_anomaly_scores)
                y_pred_knn = (densities_online < density_threshold).astype(int)
                novelty_results['knn_density_precision'] = precision_score(y_online_true_novel, y_pred_knn, zero_division=0)
                novelty_results['knn_density_recall'] = recall_score(y_online_true_novel, y_pred_knn, zero_division=0)
                novelty_results['knn_density_f1'] = f1_score(y_online_true_novel, y_pred_knn, zero_division=0)
                novelty_results['knn_density_tp'] = int(((y_pred_knn == 1) & (y_online_true_novel == 1)).sum())
                novelty_results['knn_density_fp'] = int(((y_pred_knn == 1) & (y_online_true_novel == 0)).sum())
                novelty_results['knn_density_fn'] = int(((y_pred_knn == 0) & (y_online_true_novel == 1)).sum())
            except ValueError:
                for k in ['knn_density_auc','knn_density_aupr','knn_density_precision','knn_density_recall','knn_density_f1']:
                    novelty_results[k] = np.nan

        # (C) Mahalanobis (Ledoit-Wolf shrinkage)
        try:
            lw = LedoitWolf().fit(tj_concatenations_train)
            diff = tj_concatenations_online - lw.location_
            maha_scores = np.einsum('ij,jk,ik->i', diff, lw.precision_, diff)  # higher = more novel
            novelty_results['maha_auc'] = roc_auc_score(y_online_true_novel, maha_scores)
            novelty_results['maha_aupr'] = average_precision_score(y_online_true_novel, maha_scores)
        except Exception:
            novelty_results['maha_auc'] = np.nan
            novelty_results['maha_aupr'] = np.nan

        # (D) Isolation Forest
        try:
            iso = IsolationForest(random_state=SEED, n_estimators=200, contamination='auto')
            iso.fit(tj_concatenations_train)
            # decision_function: higher means less anomalous; invert
            iso_scores = -iso.decision_function(tj_concatenations_online)
            novelty_results['iforest_auc'] = roc_auc_score(y_online_true_novel, iso_scores)
            novelty_results['iforest_aupr'] = average_precision_score(y_online_true_novel, iso_scores)
        except Exception:
            novelty_results['iforest_auc'] = np.nan
            novelty_results['iforest_aupr'] = np.nan

        # (E) One-Class SVM (may be slower; keep small nu)
        try:
            ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
            ocsvm.fit(tj_concatenations_train)
            # decision_function: positive inlier, negative outlier
            ocsvm_scores = -ocsvm.decision_function(tj_concatenations_online)
            novelty_results['ocsvm_auc'] = roc_auc_score(y_online_true_novel, ocsvm_scores)
            novelty_results['ocsvm_aupr'] = average_precision_score(y_online_true_novel, ocsvm_scores)
        except Exception:
            novelty_results['ocsvm_auc'] = np.nan
            novelty_results['ocsvm_aupr'] = np.nan

        print("[Novelty Metrics] (online stream)")
        for k_, v_ in novelty_results.items():
            if isinstance(v_, float):
                print(f"  {k_}: {v_:.4f}")
            else:
                print(f"  {k_}: {v_}")

        # Internal clustering metrics in ORIGINAL space (train + eval only)
        # (Exclude online to avoid biasing with unseen novelties.)
        raw_scores_dir = f"./novelty_raw/{env_code}"
        os.makedirs(raw_scores_dir, exist_ok=True)

        # Build per-sample arrays explicitly (avoid nested locals() bug)
        N = len(y_online_true_novel)
        lof_anomaly_scores_arr = (-lof_scores) if 'lof_scores' in locals() else np.full(N, np.nan)
        density_anomaly_scores_arr = (-densities_online) if 'densities_online' in locals() else np.full(N, np.nan)
        maha_scores_arr = maha_scores if 'maha_scores' in locals() else np.full(N, np.nan)
        iso_scores_arr = iso_scores if 'iso_scores' in locals() else np.full(N, np.nan)
        ocsvm_scores_arr = ocsvm_scores if 'ocsvm_scores' in locals() else np.full(N, np.nan)

        # Long-form per-sample CSV (novelty_raw)
        novelty_df = pd.DataFrame({
            "env": env_code,
            "seed": SEED,
            "ratio": ratio,
            "index": np.arange(N),
            "y_true_novel": y_online_true_novel,
            "lof_score": lof_anomaly_scores_arr,
            "knn_density_score": density_anomaly_scores_arr,
            "maha_score": maha_scores_arr,
            "iforest_score": iso_scores_arr,
            "ocsvm_score": ocsvm_scores_arr,
            "density_threshold": np.full(N, density_threshold if 'density_threshold' in locals() else np.nan),
        })
        novelty_csv_path = os.path.join(raw_scores_dir, "novelty_raw.csv")
        novelty_df.to_csv(novelty_csv_path, mode="a", header=not os.path.exists(novelty_csv_path), index=False)
        print(f"Appended novelty_raw CSV: {novelty_csv_path}")

        # Also append a one-row summary CSV with aggregate metrics
        novelty_summary_row = {"env": env_code, "seed": SEED, "ratio": ratio}
        novelty_summary_row.update(novelty_results)
        novelty_summary_df = pd.DataFrame([novelty_summary_row])
        novelty_summary_csv = os.path.join(raw_scores_dir, "novelty_summary.csv")
        novelty_summary_df.to_csv(novelty_summary_csv, mode="a", header=not os.path.exists(novelty_summary_csv), index=False)
        print(f"Appended novelty_summary CSV: {novelty_summary_csv}")

        # np.savez_compressed(
        #     os.path.join(raw_scores_dir, f"raw_anomaly_seed_{SEED}_ratio_{ratio}.npz"),
        #     y_online_true_novel=y_online_true_novel,
        #     lof_scores=(lof_anomaly_scores if 'lof_anomaly_scores' in locals() else None),
        #     density_scores=(density_anomaly_scores if 'density_anomaly_scores' in locals() else None),
        #     maha_scores=(maha_scores if 'maha_scores' in locals() else None),
        #     iforest_scores=(iso_scores if 'iso_scores' in locals() else None),
        #     ocsvm_scores=(ocsvm_scores if 'ocsvm_scores' in locals() else None),
        #     density_threshold=np.array([density_threshold]) if 'density_threshold' in locals() else None
        # )
        # print("Saved raw anomaly scores for ROC/AUPR plotting.")

        # Internal clustering metrics (original space train+eval)
        try:
            X_internal = np.vstack([tj_concatenations_train, tj_concatenations_evaluation])
            y_internal_true = np.hstack([tj_embeddings_train_true_labels, tj_embeddings_evaluation_true_labels])
            sil_cos = silhouette_score(X_internal, y_internal_true, metric='cosine') if len(np.unique(y_internal_true)) > 1 else np.nan
            dbi = davies_bouldin_score(X_internal, y_internal_true)
            chi = calinski_harabasz_score(X_internal, y_internal_true)
        except Exception:
            sil_cos = dbi = chi = np.nan
        print(f"[Internal Original-Space] Silhouette(cosine)={sil_cos:.4f}  DBI={dbi:.4f}  CHI={chi:.1f}")

        # Save internal original-space metrics (one row per run; accumulates across seeds/ratios)
        internal_row = {
            "env": env_code,
            "seed": SEED,
            "ratio": ratio,
            "silhouette_cosine": sil_cos,
            "davies_bouldin": dbi,
            "calinski_harabasz": chi,
        }
        internal_df = pd.DataFrame([internal_row])
        internal_csv = os.path.join(raw_scores_dir, "internal_original_space.csv")
        internal_df.to_csv(internal_csv, mode="a", header=not os.path.exists(internal_csv), index=False)
        print(f"Appended internal_original_space CSV: {internal_csv}")

    if meanshift:
                if not check_unseen:
                    bandwidth = estimate_bandwidth(tj_concatenations_seen, quantile=quantile_ms) #0.5 for TLF, 0.25 for Traj2d
                    ms = MeanShift(bandwidth=bandwidth).fit(tj_concatenations_seen)
                else:
                    bandwidth = estimate_bandwidth(tj_concatenations_seen_with_online, quantile=quantile_ms) #0.125 for Traj2d, 0.15 for Rv4
                    ms = MeanShift(bandwidth=bandwidth).fit(tj_concatenations_seen_with_online)

                cluster_labels = ms.labels_
                cluster_centers = ms.cluster_centers_
                labels_unique = np.unique(cluster_labels)
                n_clusters_ = len(labels_unique)
                print("number of estimated clusters : %d" % n_clusters_)

    elif hdb:
        X_cluster = tj_concatenations_seen_with_online if check_unseen else tj_concatenations_seen
        nX = len(X_cluster)
        min_cluster_size = max(5, int(granularity * nX))  # smallest behavior granularity ~2% of data
        min_samples = max(1, int(math.sqrt(min_cluster_size)))  # stricter core definition

        hdb = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='cosine'
        ).fit(X_cluster)

        cluster_labels = hdb.labels_  # -1 = noise
        # Build representative centers (median per cluster, ignore noise)
        unique_core = np.unique(cluster_labels[cluster_labels != -1])
        centers = []
        for c in unique_core:
            pts = X_cluster[cluster_labels == c]
            centers.append(np.median(pts, axis=0))
        cluster_centers = np.vstack(centers) if len(centers) > 0 else np.zeros((0, X_cluster.shape[1]))
        final_center_ids = unique_core.copy()

        n_clusters_ = len(unique_core)
        print(f"[HDBSCAN] number of estimated clusters (excluding noise): {n_clusters_}")

    elif gmm:
        # Gaussian Mixture on unit vectors (diag cov works well)
        X_cluster = tj_concatenations_seen_with_online if check_unseen else tj_concatenations_seen
        gm = GaussianMixture(
            n_components=K_known, covariance_type='diag',
            random_state=SEED, n_init=5, reg_covar=1e-6
        ).fit(X_cluster)
        cluster_labels = gm.predict(X_cluster)
        cluster_centers = gm.means_
        cluster_centers = cluster_centers / (np.linalg.norm(cluster_centers, axis=1, keepdims=True) + 1e-8)
        n_clusters_ = K_known
        print(f"[GMM] K={n_clusters_} | counts={np.unique(cluster_labels, return_counts=True)}")

    elif controller:
        # 1) Fit seen-only HDBSCAN, build registry
        X_seen = tj_concatenations_seen
        model_seen, labels_seen, core_ids, centers_seen = _fit_hdbscan_seen(X_seen, granularity=granularity, seed=SEED)
        registry = _build_registry(X_seen, labels_seen, core_ids, centers_seen)

        # 2) Inductive assignment for online with novelty gating
        if check_unseen:
            X_online = tj_concatenations_online
            
            # --- STAGE 1: EVALUATE INITIAL NOVELTY DETECTION ---
            print("\n--- Evaluating Initial Novelty Detection (Before Learning) ---")
            maha_p = 0.99 if args.Traj2d else 0.99 if args.Reacherv4 else 0.99 if args.Pusherv4 else 0.9 if args.Walker2dv4 else 0.99 if args.Humanoidv4 else 0.99
            labels_online_initial, novel_mask_initial, novelty_scores_initial = _assign_or_flag_online(X_online, registry, maha_p=maha_p)

            num_novel_initial = np.sum(novel_mask_initial)
            num_assigned_initial = len(X_online) - num_novel_initial
            print(f"Initial pass: {num_novel_initial} points flagged as novel, {num_assigned_initial} points assigned.")

            y_online_true = np.array(tj_embeddings_online_true_labels)
            train_label_set = set(np.unique(tj_embeddings_seen_true_labels))
            y_online_true_novel = np.array([1 if lab not in train_label_set else 0 for lab in y_online_true])

            try:
                # Evaluate based on the initial hard mask
                auc_initial = roc_auc_score(y_online_true_novel, novel_mask_initial.astype(int))
                aupr_initial = average_precision_score(y_online_true_novel, novel_mask_initial.astype(int))
                
                # Evaluate based on the continuous novelty score
                auc_s_initial = roc_auc_score(y_online_true_novel, novelty_scores_initial)
                aupr_s_initial = average_precision_score(y_online_true_novel, novelty_scores_initial)
            except Exception as e:
                print(f"Could not compute initial novelty metrics: {e}")
                auc_initial, aupr_initial, auc_s_initial, aupr_s_initial = (np.nan, np.nan, np.nan, np.nan)
            
            try:
                outp = _plot_novelty_metrics(
                    y_true=y_online_true_novel,
                    continuous_scores=novelty_scores_initial,
                    hard_mask=novel_mask_initial,
                    out_prefix=f"{env_code}_seed{SEED}_initial",
                    title=f"{env_code} initial novelty detection (seed={SEED})",
                    show=args.visualize_clusters
                )
            except Exception as e:
                print(f"Plotting helper failed: {e}")

            # --- STAGE 2: SPAWN NEW CLUSTERS AND RE-ASSIGN ---
            print("\n--- Spawning and Re-assigning Novel Points ---")
            novel_points = X_online[novel_mask_initial]
            new_entries = _spawn_new_clusters_from_buffer(novel_points, min_cluster_size=5)
            
            # Start with the initial assignments
            labels_online = labels_online_initial
            novelty_scores = novelty_scores_initial

            if len(new_entries) > 0:
                print(f"Spawning {len(new_entries)} new cluster(s) from novel points.")
                # Register new cluster IDs after current max
                next_id = int(max(registry.keys())) + 1 if registry else 0
                for entry in new_entries:
                    registry[next_id] = entry
                    next_id += 1
                
                # Re-assign ALL online points against the UPDATED registry
                labels_online, novel_mask, novelty_scores = _assign_or_flag_online(X_online, registry, maha_p=0.99)

        else:
            labels_online = np.zeros((0,), dtype=int)
            novel_mask = np.zeros((0,), dtype=bool)
            novelty_scores = np.zeros((0,), dtype=float)

        # 3) Final labels
        cluster_labels = np.concatenate([labels_seen, labels_online], axis=0)
        unique_core = np.unique(cluster_labels[cluster_labels != -1])
        n_clusters_ = len(unique_core)
        reg_ids = sorted([cid for cid in registry.keys()])
        cluster_centers = np.stack([registry[cid]["center"] for cid in reg_ids], axis=0) if len(reg_ids) > 0 else np.zeros((0, X_seen.shape[1]))
        cluster_centers = cluster_centers / (np.linalg.norm(cluster_centers, axis=1, keepdims=True) + 1e-8)

        print(f"[Controller] core clusters (seen): {len(core_ids)} | final unique IDs (excluding noise): {n_clusters_}")

        # --- STAGE 3: EVALUATE FINAL CLUSTERING QUALITY ---
        print("\n--- Evaluating Final Clustering Quality (After Learning) ---")
        y_true_all = np.concatenate([tj_embeddings_seen_true_labels, tj_embeddings_online_true_labels]) if check_unseen else tj_embeddings_seen_true_labels
        
        # Evaluate NMI/ARI on all data points (seen + online)
        # This shows if the new cluster was correctly separated from the old ones.
        mask_core_final = cluster_labels != -1
        if mask_core_final.sum() > 1 and len(np.unique(y_true_all[mask_core_final])) > 1 and len(np.unique(cluster_labels[mask_core_final])) > 1:
            nmi_final = normalized_mutual_info_score(y_true_all[mask_core_final], cluster_labels[mask_core_final])
            ari_final = adjusted_rand_score(y_true_all[mask_core_final], cluster_labels[mask_core_final])
        else:
            nmi_final, ari_final = np.nan, np.nan

        if args.visualize_clusters:
            Z_seen = reducer.transform(tj_concatenations_seen)
            Z_online_viz = reducer.transform(tj_concatenations_online) if check_unseen else np.zeros((0, Z_seen.shape[1]))
            visualize_controller_output(
                Z_seen=Z_seen,
                Z_online=Z_online_viz,
                labels_seen=labels_seen,
                labels_online=labels_online,
                novelty_scores=novelty_scores,
                registry=registry,
                reducer=reducer,
                title=f"Controller Output on {env_id}",
                is_3d=args.threeD
            )
    
    for idx, kmeans_label in zip(indices_all, cluster_labels):
        trajectory_manager[idx]['kmeans_cluster_label'] = int(kmeans_label)
    unique_labels = np.unique(cluster_labels)

    train_len = len(Z_train)
    eval_len = len(Z_evaluation)
    train_ac_labels = cluster_labels[:train_len]
    eval_ac_labels = cluster_labels[train_len:train_len+eval_len]

    if prefix_testing:
        print("\n[Prefix / Early Evaluation - Online-only]")
        assert args.embedding_strategy == "cls_only" or args.embedding_strategy == "goal_oriented", "Prefix API supports cls_only embeddings."

        # Baselines/detectors from seen full embeddings:
        # - LOF on seen (train+eval) for density-based novelty
        # - kNN on train for density thresholding
        # - LedoitWolf/IForest/OCSVM on train
        train_label_set = set(np.unique(tj_embeddings_train_true_labels))

        if 'lof' not in locals():
            lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
            lof.fit(tj_concatenations_seen)  # seen = train + eval

        if 'nn_online' not in locals() or 'density_threshold' not in locals():
            nn_online = NearestNeighbors(n_neighbors=11).fit(tj_concatenations_train)
            d_tr, _ = nn_online.kneighbors(tj_concatenations_train)
            dens_tr = 1.0 / (np.mean(d_tr[:, 1:], axis=1) + 1e-8)
            density_threshold = dens_tr.mean() - dens_tr.std()

        if 'lw' not in locals():
            lw = LedoitWolf().fit(tj_concatenations_train)

        if 'iso' not in locals():
            iso = IsolationForest(random_state=SEED, n_estimators=200, contamination='auto')
            iso.fit(tj_concatenations_train)

        if 'ocsvm' not in locals():
            ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
            ocsvm.fit(tj_concatenations_train)

        prefix_fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
        online_indices_offset = len(trajectories)

        # rows: (frac, lof_auc, lof_aupr, knn_auc, knn_aupr, maha_auc, maha_aupr, if_auc, if_aupr, oc_auc, oc_aupr)
        prefix_summary = []

        for frac in prefix_fractions:
            tag = f"_p{int(round(frac*100))}"

            # Only ONLINE prefix embeddings; seen remain full-length
            _, concat_online_p, idx_online_p = inference_on_dataloader_cdec_sa_prefix(
                behaviorencoder, cluster_centroids, total_dataloader_online, trajectory_manager, device,
                prefix_fraction=frac, index_offset=online_indices_offset, tag_suffix=tag
            )

            X_online_p = concat_online_p.squeeze(1).cpu().numpy()

            # --- HDBSCAN per-prefix on [seen + online_prefix] and save labels per item ---
            X_seen_full = tj_concatenations_seen  # train+eval (full-length seen set)
            X_cluster_p = np.vstack([X_seen_full, X_online_p])
            nXp = len(X_cluster_p)
            min_cluster_size_p = max(5, int(0.02 * nXp))
            min_samples_p = max(1, int(math.sqrt(min_cluster_size_p)))

            hdb_prefix = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size_p,
                min_samples=min_samples_p,
                metric='euclidean'
            ).fit(X_cluster_p)

            labels_prefix_all = hdb_prefix.labels_
            labels_prefix_online = labels_prefix_all[len(X_seen_full):]  # slice online part

            # Store per-prefix cluster label in trajectory_manager with the same key pattern used elsewhere
            for i_idx, lbl in zip(idx_online_p, labels_prefix_online):
                trajectory_manager[i_idx][f'kmeans_cluster_label{tag}'] = int(lbl)

            # Ground-truth novelty vs seen train labels
            y_online_true_novel = np.array([
                1 if int(trajectory_manager[i]['real_cluster_label']) not in train_label_set else 0
                for i in idx_online_p
            ])

            # LOF (higher score => more normal, invert)
            try:
                lof_scores_p = lof.decision_function(X_online_p)
                lof_auc = roc_auc_score(y_online_true_novel, -lof_scores_p)
                lof_aupr = average_precision_score(y_online_true_novel, -lof_scores_p)
            except Exception:
                lof_auc, lof_aupr = np.nan, np.nan

            # kNN density vs train (invert density for anomaly)
            try:
                d_on_p, _ = nn_online.kneighbors(X_online_p)
                dens_on_p = 1.0 / (np.mean(d_on_p[:, 1:], axis=1) + 1e-8)
                knn_auc = roc_auc_score(y_online_true_novel, -dens_on_p)
                knn_aupr = average_precision_score(y_online_true_novel, -dens_on_p)
            except Exception:
                knn_auc, knn_aupr = np.nan, np.nan

            # Mahalanobis
            try:
                diff_p = X_online_p - lw.location_
                maha_p = np.einsum('ij,jk,ik->i', diff_p, lw.precision_, diff_p)
                maha_auc = roc_auc_score(y_online_true_novel, maha_p)
                maha_aupr = average_precision_score(y_online_true_novel, maha_p)
            except Exception:
                maha_auc, maha_aupr = np.nan, np.nan

            # Isolation Forest
            try:
                if_scores_p = -iso.decision_function(X_online_p)
                if_auc = roc_auc_score(y_online_true_novel, if_scores_p)
                if_aupr = average_precision_score(y_online_true_novel, if_scores_p)
            except Exception:
                if_auc, if_aupr = np.nan, np.nan

            # One-Class SVM
            try:
                oc_scores_p = -ocsvm.decision_function(X_online_p)
                oc_auc = roc_auc_score(y_online_true_novel, oc_scores_p)
                oc_aupr = average_precision_score(y_online_true_novel, oc_scores_p)
            except Exception:
                oc_auc, oc_aupr = np.nan, np.nan

            # --- ARI/NMI on online prefix vs. true labels ---
            y_online_true = np.array([int(trajectory_manager[i]['real_cluster_label']) for i in idx_online_p])
            pred_key = f'kmeans_cluster_label{tag}'
            y_online_pred = []
            for i in idx_online_p:
                if pred_key in trajectory_manager[i]:
                    y_online_pred.append(int(trajectory_manager[i][pred_key]))
                elif 'kmeans_cluster_label' in trajectory_manager[i]:
                    y_online_pred.append(int(trajectory_manager[i]['kmeans_cluster_label']))
                else:
                    y_online_pred.append(-1)
            y_online_pred = np.array(y_online_pred)

            try:
                if len(np.unique(y_online_true)) > 1 and len(np.unique(y_online_pred)) > 1:
                    ari_online = adjusted_rand_score(y_online_true, y_online_pred)
                    nmi_online = normalized_mutual_info_score(y_online_true, y_online_pred)
                else:
                    ari_online, nmi_online = np.nan, np.nan
            except Exception:
                ari_online, nmi_online = np.nan, np.nan

            # keep tuple order, then add ARI/NMI at the end
            prefix_summary.append((
                frac, lof_auc, lof_aupr, knn_auc, knn_aupr,
                maha_auc, maha_aupr, if_auc, if_aupr, oc_auc, oc_aupr,
                ari_online, nmi_online
            ))
            print(f" prefix={frac:.2f} | LOF AUC={lof_auc:.3f} AUPR={lof_aupr:.3f} | "
                    f"kNN AUC={knn_auc:.3f} AUPR={knn_aupr:.3f} | "
                    f"MAHA AUC={maha_auc:.3f} AUPR={maha_aupr:.3f} | "
                    f"IF AUC={if_auc:.3f} AUPR={if_aupr:.3f} | "
                    f"OCSVM AUC={oc_auc:.3f} AUPR={oc_aupr:.3f} | "
                    f"ARI={ari_online if not np.isnan(ari_online) else 'nan'} "
                    f"NMI={nmi_online if not np.isnan(nmi_online) else 'nan'}")

        # Save online-only prefix metrics
        prefix_out_dir = f"./prefix_eval/{env_code}"
        os.makedirs(prefix_out_dir, exist_ok=True)

        prefix_df = pd.DataFrame({
            "env": env_code,
            "seed": SEED,
            "ratio": ratio,
            "fraction": np.array([r[0] for r in prefix_summary]),
            "lof_auc": np.array([r[1] for r in prefix_summary]),
            "lof_aupr": np.array([r[2] for r in prefix_summary]),
            "knn_auc": np.array([r[3] for r in prefix_summary]),
            "knn_aupr": np.array([r[4] for r in prefix_summary]),
            "maha_auc": np.array([r[5] for r in prefix_summary]),
            "maha_aupr": np.array([r[6] for r in prefix_summary]),
            "if_auc": np.array([r[7] for r in prefix_summary]),
            "if_aupr": np.array([r[8] for r in prefix_summary]),
            "oc_auc": np.array([r[9] for r in prefix_summary]),
            "oc_aupr": np.array([r[10] for r in prefix_summary]),
            "ari_online": np.array([r[11] for r in prefix_summary]),
            "nmi_online": np.array([r[12] for r in prefix_summary]),
        })
        prefix_csv_path = os.path.join(prefix_out_dir, "prefix_analysis.csv")
        prefix_df.to_csv(prefix_csv_path, mode="a", header=not os.path.exists(prefix_csv_path), index=False)
        print(f"Appended prefix_analysis CSV: {prefix_csv_path}")
                        

    # --- Metrics print for controller ---
    if controller:
        print("************************************Cluster Controller Prediction************************************")
        y_true_all = tj_concatenations_seen_true_labels if not check_unseen else tj_concatenations_seen_true_labels_with_online
        print(f"Predicted clusters (K excl. noise) = {len(np.unique(cluster_labels[cluster_labels!=-1]))} ",
                np.unique(cluster_labels, return_counts=True))
        
        # --- Seen Data Quality ---
        print("\n--- Seen Data Performance ---")
        mask_seen_core = (np.arange(len(cluster_labels)) < len(tj_concatenations_seen)) & (cluster_labels != -1)
        if mask_seen_core.sum() >= 2 and len(np.unique(cluster_labels[mask_seen_core])) >= 2:
            ari_seen_core = adjusted_rand_score(y_true_all[mask_seen_core], cluster_labels[mask_seen_core])
            nmi_seen_core = normalized_mutual_info_score(y_true_all[mask_seen_core], cluster_labels[mask_seen_core])
            print(f"ARI (seen core): {ari_seen_core:.4f}")
            print(f"NMI (seen core): {nmi_seen_core:.4f}")
        else:
            print("Insufficient seen core structure for ARI/NMI.")

        # --- Online Data Quality ---
        if check_unseen:
            print("\n--- Online Data Performance ---")
            # Initial Novelty Detection
            print(f"Initial Gating AUC: {auc_initial:.4f}, AUPR: {aupr_initial:.4f}")
            print(f"Initial Score AUC: {auc_s_initial:.4f}, Score AUPR: {aupr_s_initial:.4f}")
            
            # Final Clustering Quality
            print(f"Final NMI (all core points): {nmi_final:.4f}")
            print(f"Final ARI (all core points): {ari_final:.4f}")

            # Optional ARI/NMI on assigned online points that belong to seen classes
            mask_assigned = (cluster_labels[len(tj_concatenations_seen):] != -1)
            mask_seenclass = np.array([int(lab) in train_label_set for lab in y_online_true])
            mask_eval = mask_assigned & mask_seenclass
            if np.any(mask_eval) and len(np.unique(cluster_labels[len(tj_concatenations_seen):][mask_eval])) >= 2 and len(np.unique(y_online_true[mask_eval])) >= 2:
                ari_online_assigned = adjusted_rand_score(y_online_true[mask_eval], cluster_labels[len(tj_concatenations_seen):][mask_eval])
                nmi_online_assigned = normalized_mutual_info_score(y_online_true[mask_eval], cluster_labels[len(tj_concatenations_seen):][mask_eval])
                print(f"ARI (online assigned, seen classes): {ari_online_assigned:.4f}")
                print(f"NMI (online assigned, seen classes): {nmi_online_assigned:.4f}")

    elif meanshift:
        print("************************************MeanShift Prediction************************************")
        print(f"Predicted Labels with estimated K = {n_clusters_} ",np.unique(cluster_labels,return_counts=True))
        print("ARI",adjusted_rand_score(tj_concatenations_seen_true_labels if not check_unseen else tj_concatenations_seen_true_labels_with_online,cluster_labels))
        print("SILHOUETTE",silhouette_score(umap_combined,cluster_labels))
        print("SILHOUETTE TRUE",silhouette_score(umap_combined,tj_concatenations_seen_true_labels if not check_unseen else tj_concatenations_seen_true_labels_with_online))
        print("NMI",normalized_mutual_info_score(tj_concatenations_seen_true_labels if not check_unseen else tj_concatenations_seen_true_labels_with_online,cluster_labels))
        print("HOMOGENEITY",homogeneity_score(tj_concatenations_seen_true_labels if not check_unseen else tj_concatenations_seen_true_labels_with_online,cluster_labels))
        print("COMPLETENESS",completeness_score(tj_concatenations_seen_true_labels if not check_unseen else tj_concatenations_seen_true_labels_with_online,cluster_labels))
        print("V MEASURE",v_measure_score(tj_concatenations_seen_true_labels if not check_unseen else tj_concatenations_seen_true_labels_with_online,cluster_labels))
    elif hdb:
        print("************************************HDBSCAN Prediction************************************")
        y_true_all = tj_concatenations_seen_true_labels if not check_unseen else tj_concatenations_seen_true_labels_with_online
        mask_core = cluster_labels != -1
        if mask_core.sum() >= 2 and len(np.unique(cluster_labels[mask_core])) >= 2:
            print(f"Predicted core clusters (K) = {n_clusters_} ", np.unique(cluster_labels[mask_core], return_counts=True))
            print("ARI (core)", adjusted_rand_score(y_true_all[mask_core], cluster_labels[mask_core]))
            print("SILHOUETTE (core)", silhouette_score(umap_combined[mask_core], cluster_labels[mask_core]))
            print("SILHOUETTE TRUE", silhouette_score(umap_combined, y_true_all))
            print("NMI (core)", normalized_mutual_info_score(y_true_all[mask_core], cluster_labels[mask_core]))
            print("HOMOGENEITY (core)", homogeneity_score(y_true_all[mask_core], cluster_labels[mask_core]))
            print("COMPLETENESS (core)", completeness_score(y_true_all[mask_core], cluster_labels[mask_core]))
            print("V MEASURE (core)", v_measure_score(y_true_all[mask_core], cluster_labels[mask_core]))
        else:
            print("Insufficient core clustering structure for external metrics (need >=2 clusters).")
    elif gmm:
        print("************************************Gaussian Mixture Prediction************************************")
        y_true_all = tj_concatenations_seen_true_labels if not check_unseen else tj_concatenations_seen_true_labels_with_online
        print(f"Predicted clusters (K) = {n_clusters_} ", np.unique(cluster_labels, return_counts=True))
        if len(np.unique(cluster_labels)) >= 2:
            print("ARI", adjusted_rand_score(y_true_all, cluster_labels))
            print("SILHOUETTE", silhouette_score(umap_combined, cluster_labels))
            print("SILHOUETTE TRUE", silhouette_score(umap_combined, y_true_all))
            print("NMI", normalized_mutual_info_score(y_true_all, cluster_labels))
            print("HOMOGENEITY", homogeneity_score(y_true_all, cluster_labels))
            print("COMPLETENESS", completeness_score(y_true_all, cluster_labels))
            print("V MEASURE", v_measure_score(y_true_all, cluster_labels))
        else:
            print("Insufficient cluster variety for external metrics (need >=2 clusters).")

    print("*******************************************************************************************")
    # exit("Exiting after Traj2d clustering and IRL training.")

    if saving:
        final_df = pd.DataFrame(results)
        final_df.to_csv(csv_file_path, index=False)
        print(f"\nAll experiments done. Final results saved to {csv_file_path}")

if __name__ == "__main__":
    main()