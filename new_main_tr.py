import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.data import DataLoader
from torch.nn.utils import parametrizations as parametrize
import umap
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from sklearn.manifold import trustworthiness
from scipy.stats import wasserstein_distance

from imitation.data.types import Trajectory
from morl_behavior_objective.methods.behaviorencoder import (
    BehaviorEncoderCLSattnSATyped,
    DeepInfoMaxLoss,
    InstanceLoss,
)
from morl_behavior_objective.methods.behaviorencoder_utils import (
    apply_per_traj,
    build_scaler,
    datasets_preparation_sa,
    fit_on_flat,
    is_cont_actions,
    is_coord_states,
    prepare_sa_trajectories,
    expand_interleaved_mask,
)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---------- Decoder Definition ----------
class TrajectoryDecoder(nn.Module):
    def __init__(self, emb_dim, state_dim, action_dim, max_len,spec_norm=False):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_len = max_len
        self.out_features = max_len * (state_dim + action_dim)

        self.net = nn.Sequential(
            nn.Linear(emb_dim, 1024) if not spec_norm else parametrize.spectral_norm(nn.Linear(emb_dim, 1024)),
            nn.GELU(),
            nn.Linear(1024, 1024) if not spec_norm else parametrize.spectral_norm(nn.Linear(1024, 1024)),
            nn.GELU(),
            nn.Linear(1024, self.out_features) if not spec_norm else parametrize.spectral_norm(nn.Linear(1024, self.out_features)),
        )

    def forward(self, cls_embedding):
        # cls_embedding shape: [B, emb_dim]
        flat_recon = self.net(cls_embedding)
        # flat_recon shape: [B, max_len * (state_dim + action_dim)]

        # Reshape to [B, max_len, state_dim + action_dim]
        recon = flat_recon.view(-1, self.max_len, self.state_dim + self.action_dim)

        states = recon[..., : self.state_dim]
        actions = recon[..., self.state_dim :]
        return states, actions

# ---------- Loss Functions ----------
def extract_scalar_returns(returns):
    """
    Extracts a single scalar value from various return formats.
    """
    values = []
    for ret in returns:
        if ret is None:
            return None
        # Handle tensor inputs from the dataloader
        if isinstance(ret, th.Tensor):
            ret = ret.cpu().numpy()
        
        if isinstance(ret, dict):
            if "scalar" in ret:
                values.append(float(ret["scalar"]))
            elif "sum" in ret:
                values.append(float(ret["sum"]))
            else:
                return None
        elif isinstance(ret, (list, tuple, np.ndarray)):
            if len(ret) == 0:
                return None
            values.append(float(ret[0]))
        else:
            values.append(float(ret))
    return values

def loss_vol_simplified(z_normalized):
    # z_normalized are the embeddings already on the hypersphere
    s = z_normalized.std(0)
    eta = 1e-6  # Small constant to avoid log(0)
    # The geometric mean is still a valid measure of spread on the sphere
    return th.exp(th.log(s + eta).mean())

def pole_penalty_loss(z):
    """
    Penalizes embeddings for having one dimension dominate the others.
    This discourages points from clustering at the poles of the hypersphere.
    Assumes z is L2-normalized.
    """
    # z shape: [B, emb_dim]
    # For each embedding, find the maximum absolute value across its dimensions.
    # A value close to 1.0 means it's near a pole.
    max_abs_vals, _ = th.max(th.abs(z), dim=1)
    
    # The loss is the mean of these maximum values. Minimizing this pushes
    # points away from the poles.
    return max_abs_vals.mean()


def decorrelation_loss(z):
    """
    Encourages different dimensions of the embeddings to be uncorrelated.
    Assumes z is centered (mean=0) across the batch.
    """
    # z shape: [B, emb_dim]
    z = z - z.mean(dim=0) # Center the batch
    cov_matrix = (z.T @ z) / (len(z) - 1) # [emb_dim, emb_dim]
    
    # We want the off-diagonal elements to be zero.
    # Penalize the sum of the squares of the off-diagonal elements.
    off_diag_mask = ~th.eye(z.shape[1], dtype=th.bool, device=z.device)
    loss = cov_matrix[off_diag_mask].pow(2).sum() / z.shape[1]
    return loss

def segment_contrastive_loss(
    full_cls_emb: th.Tensor,
    encoder: nn.Module,
    states: th.Tensor,
    actions: th.Tensor,
    masks: th.Tensor,
    L: int,
    contrastive_loss_fn : nn.Module,
    num_segments: int = 2,
    pairwise_segments:bool = False,
):
    """
    Global-vs-Segment contrastive loss.
    Contrasts the CLS embedding of the full trajectory against the CLS embeddings
    of num_segments random segments of length L from that same trajectory.
    """
    B, T, _ = states.shape
    device = states.device

    # 1) Calculate valid lengths and sample start indices
    valid_lengths = (~masks).sum(dim=1)
    max_starts = (valid_lengths - L).clamp(min=0)
    
    if max_starts.sum() == 0:
        return th.tensor(0.0, device=device)

    rand = th.rand(B, num_segments, device=device)
    starts = (rand * max_starts.unsqueeze(1).float()).long().clamp(max=max_starts.unsqueeze(1))

    seg_batches = []
    for k in range(num_segments):
        idx_k = starts[:, k]
        # Create padded tensors for segments to ensure consistent shape for the encoder
        seg_states = th.zeros(B, L, states.shape[2], device=device)
        seg_actions = th.zeros(B, L, actions.shape[2], device=device)
        seg_masks = th.ones(B, L, dtype=th.bool, device=device) # All padded initially

        for b in range(B):
            # Only process if a valid segment can be extracted
            if max_starts[b] > 0:
                start_idx = idx_k[b]
                end_idx = start_idx + L
                seg_states[b] = states[b, start_idx:end_idx]
                seg_actions[b] = actions[b, start_idx:end_idx]
                seg_masks[b, :] = False # This segment is not padded
        
        seg_batches.append((seg_states, seg_actions, seg_masks))

    # 3) Encode segments → CLS
    z_full = F.normalize(full_cls_emb, dim=1)
    z_segs = []
    for k in range(num_segments):
        # We need to run the encoder in eval mode for segments if we don't want dropout here
        # but for consistency with the main passes, we keep it in train mode.
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
        loss_pairwise = loss_pairwise / float(pairs) if pairs > 0 else 0.0

    loss = loss_full_vs_segs + loss_pairwise
    return loss

def datasets_preparation_ret(
    all_states, all_actions, all_masks, all_labels,
    train_size, val_size, test_size,
    loader_batch, val_bptt, test_bptt, seed,
    returns=None
):
    """
    Prepares datasets and DataLoaders for training, validation, and testing,
    including trajectory returns.
    """
    

    # Ensure returns are provided and have the correct length
    if returns is None or len(returns) != len(all_states):
        raise ValueError("Returns must be provided and match the number of trajectories.")

    # Convert returns to a tensor
    returns_tensor = th.tensor(np.array(returns), dtype=th.float32)

    # Create a full dataset including returns
    full_dataset = TensorDataset(all_states, all_actions, all_masks, all_labels, returns_tensor)

    # Split dataset into training, validation, and test sets
    if val_size > 0 or test_size > 0:
        train_set, val_set, test_set = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=th.Generator().manual_seed(seed)
        )
    else:
        train_set, val_set, test_set = full_dataset, None, None

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=loader_batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=val_bptt) if val_set else None
    test_loader = DataLoader(test_set, batch_size=test_bptt) if test_set else None

    return full_dataset, train_set, val_set, test_set, train_loader, val_loader, test_loader, None

# ---------- Training & Evaluation ----------
def train_epoch(
    encoder, decoder, loader, optim, device, info_loss_fn, dim_loss_fn,
    recon_weight, info_weight, dim_weight, vol_weight, decorr_weight=0.0, least_volumes=False,
    segment_weight=0.0, env_id=None
):
    encoder.train()
    decoder.train()
    dim_loss_fn.train()
    total_loss = 0.0

    for states, actions, masks, _, returns in loader:
        states, actions, masks = states.to(device), actions.to(device), masks.to(device)

        # --- Two stochastic forward passes for contrastive learning ---
        all_tokens1, _, _, _, cls_emb1, _ = encoder(states, actions, src_key_padding_mask=masks)
        all_tokens2, _, _, _, cls_emb2, _ = encoder(states, actions, src_key_padding_mask=masks)
        
        # Decoder can use either embedding, let's use the first one
        states_rec, actions_rec = decoder(cls_emb1)

        # --- Reconstruction Loss ---
        mask_flat = masks.view(masks.size(0), -1).unsqueeze(-1)
        valid = (~mask_flat).float()
        tgt_states = states
        tgt_actions = actions
        rec_state = ((states_rec - tgt_states).pow(2) * valid).sum() / valid.sum().clamp(min=1.0)
        rec_action = ((actions_rec - tgt_actions).pow(2) * valid).sum() / valid.sum().clamp(min=1.0)
        recon_loss = rec_state + rec_action

        # --- CORRECTED InfoNCE Loss ---
        info_loss = info_loss_fn(cls_emb1, cls_emb2)

        # --- Deep InfoMax Loss (averaged over both views) ---
        interleaved_mask = expand_interleaved_mask(masks)
        dim_loss1 = dim_loss_fn(cls_emb1, all_tokens1[:, 1:, :], interleaved_mask)
        dim_loss2 = dim_loss_fn(cls_emb2, all_tokens2[:, 1:, :], interleaved_mask)
        dim_loss = (dim_loss1 + dim_loss2) / 2.0

        # --- NEW Segment Contrastive Loss ---
        seg_loss = th.tensor(0.0, device=device)
        # if segment_weight > 0.0:
        #     # Define segment length L based on environment
        #     L_min, L_max = (2, 12) if "dst" in env_id or "sea" in env_id else (8, 16)
        #     T = states.shape[1]
        #     current_L_max = min(L_max, T - 1)
        #     L = th.randint(L_min, current_L_max + 1, (1,)).item() if current_L_max >= L_min else L_min
            
        #     seg_loss1 = segment_contrastive_loss(cls_emb1, encoder, states, actions, masks, L, info_loss_fn, num_segments=4, pairwise_segments=True)
        #     seg_loss2 = segment_contrastive_loss(cls_emb2, encoder, states, actions, masks, L, info_loss_fn, num_segments=4, pairwise_segments=True)
        #     seg_loss = (seg_loss1 + seg_loss2) / 2.0

        # --- Other losses (unchanged) ---
        decorr_loss = decorrelation_loss(cls_emb1) if decorr_weight > 0.0 else th.tensor(0.0, device=device)
        vol_loss = th.tensor(0.0, device=device)
        if least_volumes:
            vol_loss = loss_vol_simplified(cls_emb1)

        # --- Total Weighted Loss ---
        loss = (
            recon_weight * recon_loss
            + info_weight * info_loss
            + dim_weight * dim_loss
            + segment_weight * seg_loss
            + vol_weight * vol_loss
            + decorr_weight * decorr_loss
        )
        
        optim.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        th.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optim.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)

def train_epoch_no_norm(
    encoder, decoder, loader, optim, device, info_loss_fn, dim_loss_fn,
    recon_weight, info_weight, dim_weight, vol_weight, decorr_weight=0.0, least_volumes=False,
    segment_weight=0.0, env_id=None
):
    """
    Training epoch that assumes the encoder outputs are NOT normalized.
    Normalization is applied selectively for losses that require it.
    """
    encoder.train()
    decoder.train()
    dim_loss_fn.train()
    total_loss = 0.0

    for states, actions, masks, _, returns in loader:
        states, actions, masks = states.to(device), actions.to(device), masks.to(device)

        # --- Two stochastic forward passes for contrastive learning ---
        # Encoder returns unnormalized embeddings
        all_tokens1, _, _, _, cls_emb1_raw, _ = encoder(states, actions, src_key_padding_mask=masks)
        all_tokens2, _, _, _, cls_emb2_raw, _ = encoder(states, actions, src_key_padding_mask=masks)
        
        # --- Reconstruction Loss ---
        # The decoder receives the RAW (unnormalized) embedding to use magnitude information.
        states_rec, actions_rec = decoder(cls_emb1_raw)
        mask_flat = masks.view(masks.size(0), -1).unsqueeze(-1)
        valid = (~mask_flat).float()
        tgt_states = states
        tgt_actions = actions
        rec_state = ((states_rec - tgt_states).pow(2) * valid).sum() / valid.sum().clamp(min=1.0)
        rec_action = ((actions_rec - tgt_actions).pow(2) * valid).sum() / valid.sum().clamp(min=1.0)
        recon_loss = rec_state + rec_action

        # --- Loss Calculations Requiring Normalization ---
        # Normalize the embeddings before passing them to these specific losses.
        cls_emb1_norm = F.normalize(cls_emb1_raw, p=2, dim=1)
        cls_emb2_norm = F.normalize(cls_emb2_raw, p=2, dim=1)

        # --- InfoNCE Loss (uses normalized embeddings) ---
        info_loss = info_loss_fn(cls_emb1_norm, cls_emb2_norm)

        # --- Deep InfoMax Loss (uses normalized global embedding) ---
        interleaved_mask = expand_interleaved_mask(masks)
        # Note: all_tokens are also unnormalized, but DIM loss contrasts a global summary
        # with local features, so we normalize the global part.
        dim_loss1 = dim_loss_fn(cls_emb1_norm, all_tokens1[:, 1:, :], interleaved_mask)
        dim_loss2 = dim_loss_fn(cls_emb2_norm, all_tokens2[:, 1:, :], interleaved_mask)
        dim_loss = (dim_loss1 + dim_loss2) / 2.0

        # --- Other losses (using normalized embeddings) ---
        # These losses are designed to shape the geometry on the hypersphere.
        decorr_loss = decorrelation_loss(cls_emb1_norm) if decorr_weight > 0.0 else th.tensor(0.0, device=device)
        vol_loss = th.tensor(0.0, device=device)
        if least_volumes:
            vol_loss = loss_vol_simplified(cls_emb1_norm)

        # --- Segment Contrastive Loss (handles its own normalization internally) ---
        seg_loss = th.tensor(0.0, device=device)
        # if segment_weight > 0.0:
        #     ... (This part remains unchanged as the function normalizes internally)

        # --- Total Weighted Loss ---
        loss = (
            recon_weight * recon_loss
            + info_weight * info_loss
            + dim_weight * dim_loss
            + segment_weight * seg_loss
            + vol_weight * vol_loss
            + decorr_weight * decorr_loss
        )
        
        optim.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        th.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optim.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@th.no_grad()
def get_all_embeddings(encoder, loader, device):
    encoder.eval()
    all_cls_embs, all_policies = [], []
    for states, actions, masks, labels, _ in loader:
        states, actions, masks = states.to(device), actions.to(device), masks.to(device)
        all_tokens, _, _, _, cls_emb, _ = encoder(states, actions, src_key_padding_mask=masks)
        all_cls_embs.append(cls_emb.cpu().numpy())
        all_policies.extend(labels.tolist())
    return np.vstack(all_cls_embs), np.array(all_policies)

def visualize_trajectory_embeddings(embeddings, policies, emb_dim, title="Trajectory Embeddings (pre-aggregation)", highlight_pids=None):
    """Visualizes the raw, un-aggregated trajectory embeddings."""
    if emb_dim < 2:
        print("Embedding dimension < 2; skipping raw trajectory visualization.")
        return

    unique_policies = np.unique(policies)
    palette = sns.color_palette("tab10", n_colors=max(len(unique_policies), 3))
    color_map = {pid: palette[i % len(palette)] for i, pid in enumerate(unique_policies)}
    colors = [color_map[pid] for pid in policies]

    fig = plt.figure(figsize=(8, 6))
    
    plot_embeddings = embeddings
    # Use UMAP for dimensionality reduction if emb_dim > 3
    if emb_dim > 3:
        print(f"Reducing {emb_dim}D -> 3D with UMAP for raw embedding visualization.")
        reducer = umap.UMAP(n_components=3, random_state=42)
        plot_embeddings = reducer.fit_transform(embeddings)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(plot_embeddings[:, 0], plot_embeddings[:, 1], plot_embeddings[:, 2], c=colors, alpha=0.5)
        ax.set_xlabel("UMAP Dim 1"); ax.set_ylabel("UMAP Dim 2"); ax.set_zlabel("UMAP Dim 3")
    elif emb_dim == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=colors, alpha=0.5)
        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); ax.set_zlabel("Dim 3")
    else: # emb_dim == 2
        ax = fig.add_subplot(111)
        ax.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, alpha=0.5)
        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")

    # Add labels for highlighted policies
    if highlight_pids is not None:
        print(f"Highlighting policy IDs: {highlight_pids}")
        for i, pid in enumerate(policies):
            if pid in highlight_pids:
                point = plot_embeddings[i]
                if emb_dim == 2:
                    ax.text(point[0], point[1], str(pid), color='black', fontsize=9, ha='center', va='center', weight='bold')
                else: # 3D or UMAP 3D
                    ax.text(point[0], point[1], point[2], str(pid), color='black', fontsize=9, ha='center', va='center', weight='bold')

    # Create dummy artists for legend
    for pid in unique_policies:
        ax.scatter([], [], c=[color_map[pid]], label=f"Policy {pid}")

    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.show()

def aggregate_and_visualize_policy_embeddings(
    embeddings, policies, emb_dim, save_path=None
):
    policy_latents = {}
    unique_policies = np.unique(policies)
    for pid in unique_policies:
        policy_latents[int(pid)] = embeddings[policies == pid].mean(axis=0)

    if emb_dim < 2:
        print("Embedding dimension < 2; skipping aggregated visualization.")
        return policy_latents

    agg_embeddings = np.array(list(policy_latents.values()))
    agg_ids = np.array(list(policy_latents.keys()))

    palette = sns.color_palette("tab10", n_colors=max(len(unique_policies), 3))
    color_map = {pid: palette[i % len(palette)] for i, pid in enumerate(unique_policies)}

    fig = plt.figure(figsize=(8, 6))
    
    # Handle different dimensions for plotting
    plot_title = "Aggregated Policy Embeddings"
    if emb_dim > 3:
        plot_title = f"Aggregated Policy Embeddings ({emb_dim}D -> 3D via UMAP)"
        print(f"Reducing {emb_dim}D -> 3D with UMAP for aggregated visualization.")
        reducer = umap.UMAP(n_components=3, random_state=42)
        plot_embeddings = reducer.fit_transform(agg_embeddings)
        ax = fig.add_subplot(111, projection="3d")
        for i, pid in enumerate(agg_ids):
            pts = plot_embeddings[i, :]
            ax.scatter(pts[0], pts[1], pts[2], c=[color_map[pid]], label=f"Policy {pid}", s=100, alpha=0.9)
            ax.text(pts[0], pts[1], pts[2], str(pid), color='black', fontsize=12, ha='right', va='bottom')
        ax.set_xlabel("UMAP Dim 1"); ax.set_ylabel("UMAP Dim 2"); ax.set_zlabel("UMAP Dim 3")
    elif emb_dim == 3:
        plot_title = "Aggregated Policy Embeddings (3D)"
        ax = fig.add_subplot(111, projection="3d")
        for i, pid in enumerate(agg_ids):
            pts = agg_embeddings[i, :3]
            ax.scatter(pts[0], pts[1], pts[2], c=[color_map[pid]], label=f"Policy {pid}", s=100, alpha=0.9)
            ax.text(pts[0], pts[1], pts[2], str(pid), color='black', fontsize=12, ha='right', va='bottom')
        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); ax.set_zlabel("Dim 3")
    else: # emb_dim == 2
        plot_title = "Aggregated Policy Embeddings (2D)"
        ax = fig.add_subplot(111)
        for i, pid in enumerate(agg_ids):
            pts = agg_embeddings[i, :2]
            ax.scatter(pts[0], pts[1], c=[color_map[pid]], label=f"Policy {pid}", s=100, alpha=0.9)
            ax.text(pts[0], pts[1], str(pid), color='black', fontsize=12, ha='right', va='bottom')
        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")

    ax.set_title(plot_title)
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved aggregated embedding visualization to {save_path}")
    plt.show()

    return {k: v.tolist() for k, v in policy_latents.items()}


# ---------- Main Execution Block ----------
def main():
    parser = argparse.ArgumentParser()
    # Add all arguments from main_morl_BE.py for consistency
    arg_env = parser.add_argument_group('Environment Selection')
    arg_env.add_argument("-DSTC","--DeepSeaTreasureConcave", help="DeepSeaTreasureConcave environment",action="store_true")
    arg_env.add_argument("-DSTS","--DeepSeaTreasureSmooth", help="DeepSeaTreasureSmooth environment",action="store_true")
    arg_env.add_argument("-DSTLR","--DeepSeaTreasureLeftRight", help="DeepSeaTreasureLeftRight environment",action="store_true")
    arg_env.add_argument("-MHC","--MOHalfCheetah", help="MO-HalfCheetah environment",action="store_true")
    arg_env.add_argument("-MHW","--MOHighway", help="MO-Highway environment",action="store_true")
    arg_env.add_argument("-MHo2","--MOHopper2obj", help="MO-Hopper environment with 2 objectives",action="store_true")
    arg_env.add_argument("-MHo","--MOHopper", help="MO-Hopper environment",action="store_true")
    arg_env.add_argument("-RG", "--ResourceGathering", help="ResourceGathering environment", action="store_true")

    arg_hyp = parser.add_argument_group('Training Hyperparameters')
    arg_hyp.add_argument("--shorter",action="store_true", help="Use shorter trajectories for halfcheetah")
    arg_hyp.add_argument("--viz_policy_ids", type=int, nargs='+', default=None, help="List of policy IDs to label in the pre-aggregation visualization.")
    arg_hyp.add_argument("--device", default="cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu")
    arg_hyp.add_argument("--epochs", type=int, default=501)
    arg_hyp.add_argument("--spec_norm", action="store_true", help="Use spectral normalization in the decoder")
    arg_hyp.add_argument("--least_volumes", action="store_true", help="Encourage least volume embeddings")
    arg_hyp.add_argument("--batch_size", type=int, default=32)
    arg_hyp.add_argument("--lr", type=float, default=3e-4)
    arg_hyp.add_argument("--emb_dim", type=int, default=3)
    arg_hyp.add_argument("--nheads", type=int, default=3)
    arg_hyp.add_argument("--nlayers", type=int, default=2)
    arg_hyp.add_argument("--d_hid", type=int, default=1024)
    arg_hyp.add_argument("--dropout", type=float, default=0.1)
    arg_hyp.add_argument("--recon_weight", type=float, default=0.1)
    arg_hyp.add_argument("--info_weight", type=float, default=1.0)
    arg_hyp.add_argument("--dim_weight", type=float, default=1.0)
    arg_hyp.add_argument("--segment_weight", type=float, default=0.0)
    arg_hyp.add_argument("--vol_weight", type=float, default=0.0)
    arg_hyp.add_argument("--decorr_weight", type=float, default=0.0)
    arg_hyp.add_argument("--temperature", type=float, default=0.05)
    arg_hyp.add_argument("--state_scaler", default="quantile_normal")
    arg_hyp.add_argument("--action_scaler", default="quantile_normal")
    arg_hyp.add_argument("--scaler_fit", default="seen", choices=["seen", "both"])
    arg_hyp.add_argument("--model_dir", default="models/no_topo/")
    arg_hyp.add_argument("--model_prefix", default="be")
    arg_hyp.add_argument("--train", action="store_true")
    arg_hyp.add_argument("--seed", type=int, default=0)
    arg_hyp.add_argument("--save_data", action="store_true", help="Save the aggregated policy embeddings to a JSON file")
    args = parser.parse_args()

    # --- Seeding ---
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = th.device(args.device)
    print(f"Using device: {device}")
    env_code = "DSTC" if args.DeepSeaTreasureConcave else "DSTS" if args.DeepSeaTreasureSmooth else "DSTLR" if args.DeepSeaTreasureLeftRight else \
               "MHC" if args.MOHalfCheetah else "MHW" if args.MOHighway else "MHo2" if args.MOHopper2obj else "MHo" if args.MOHopper \
               else "RG" if args.ResourceGathering else "UNK"

    # --- Data Loading and Preparation (from main_morl_BE.py) ---
    if args.DeepSeaTreasureConcave or args.DeepSeaTreasureSmooth or args.DeepSeaTreasureLeftRight:
        if args.DeepSeaTreasureLeftRight:
            name_env = "left_right_dst"
            trajectories_directory_path = f"trajectories/{name_env}/"
            embeddings_folder_path = trajectories_directory_path + "embeddings/"
            os.makedirs(embeddings_folder_path, exist_ok=True)
            num_policies = 6
            env_id = "left-right-dst-v0"
        else: # Concave or Smooth
            name_env = "dst_concave" if args.DeepSeaTreasureConcave else "smooth"
            trajectories_directory_path = f"trajectories/{name_env}/"
            embeddings_folder_path = trajectories_directory_path + "embeddings/"
            os.makedirs(embeddings_folder_path, exist_ok=True)
            num_policies = 10
            env_id = "deep-sea-treasure-v0" if args.DeepSeaTreasureConcave else "dst-smooth-v0"

        trajectories, true_labels, obj_feats_list = [], [], []
        for i in range(num_policies):
            if args.DeepSeaTreasureLeftRight:
                file_path = os.path.join(trajectories_directory_path, f"{name_env}_{i}.json")
            else:
                file_path = os.path.join(trajectories_directory_path, f"dst_{i}.json" if args.DeepSeaTreasureConcave else f"smooth_{i}.json")
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            ret_vec = data.get('return', None)
            for states, actions in data['trajectories']:
                obs = np.array(list(states) + [states[-1]], dtype=np.float32)
                acts = np.array(actions, dtype=np.float32)
                traj = Trajectory(obs=obs, acts=acts, infos=None, terminal=True)
                trajectories.append(traj)
                true_labels.append(i)
                if ret_vec is not None:
                    # Ensure obj_feats_list gets populated for every trajectory, even if return is the same for a policy
                    obj_feats_list.append(np.asarray(ret_vec, dtype=np.float64))  
    elif args.MOHalfCheetah:
        name_env = "mo-halfcheetah-v4" if not args.shorter else "mo-halfcheetah-v4_100steps"
        trajectories_directory_path = f"trajectories/morld/{name_env}/"
        embeddings_folder_path = trajectories_directory_path + "embeddings/"
        os.makedirs(embeddings_folder_path, exist_ok=True)
        num_policies = 31 if not args.shorter else 80
        env_id = "mo-halfcheetah-v4" if not args.shorter else "mo-halfcheetah-v4_100steps"

        trajectories, true_labels, obj_feats_list = [], [], []
        for i in range(num_policies):
            file_path = os.path.join(trajectories_directory_path, f"policy_{i}.json")
            with open(file_path, 'r') as f:
                data = json.load(f)
            ret_vec = data.get('return', None)
            for states, actions in data['trajectories']:
                obs = np.array(list(states) + [states[-1]], dtype=np.float32)
                acts = np.array(actions, dtype=np.float32)
                # print("LEN CHEETAH OBS AND ACTS:", obs.shape, acts.shape)
                if not args.shorter:
                    obs = obs[:201]
                    acts = acts[:200]
                    timesteps = obs.shape[0] - 1
                else:
                    timesteps = 100
                # print("Truncated HalfCheetah trajectories to length 100 for faster training.")
                traj = Trajectory(obs=obs, acts=acts, infos=None, terminal=True)
                trajectories.append(traj)
                true_labels.append(i)
                if ret_vec is not None:
                    obj_feats_list.append(np.asarray(ret_vec, dtype=np.float64))
    elif args.MOHighway:
        name_env = "mo-highway-fast-v0" if not args.shorter else "mo-highway-fast-v0_100steps"
        trajectories_directory_path = f"trajectories/morld/{name_env}_test/"
        embeddings_folder_path = trajectories_directory_path + "embeddings/"
        os.makedirs(embeddings_folder_path, exist_ok=True)
        num_policies = 41# if not args.shorter else 80
        env_id = "mo-highway-fast-v0" if not args.shorter else "mo-highway-fast-v0_100steps"

        trajectories, true_labels, obj_feats_list = [], [], []
        for i in range(num_policies):
            file_path = os.path.join(trajectories_directory_path, f"policy_{i}.json")
            with open(file_path, 'r') as f:
                data = json.load(f)
            ret_vec = data.get('return', None)
            for states, actions in data['trajectories']:
                obs = np.array(list(states) + [states[-1]], dtype=np.float32)
                acts = np.array(actions, dtype=np.float32)
                if acts.ndim ==1:
                    acts = acts.reshape(-1,1)
                # print("Truncated HalfCheetah trajectories to length 100 for faster training.")
                traj = Trajectory(obs=obs, acts=acts, infos=None, terminal=True)
                trajectories.append(traj)
                true_labels.append(i)
                if ret_vec is not None:
                    obj_feats_list.append(np.asarray(ret_vec, dtype=np.float64))
    elif args.MOHopper or args.MOHopper2obj:
        name_env = "mo-hopper-v5" if args.MOHopper else "mo-hopper-2obj-v5"
        trajectories_directory_path = f"trajectories/morld/{name_env}/"
        embeddings_folder_path = trajectories_directory_path + "embeddings/"
        os.makedirs(embeddings_folder_path, exist_ok=True)
        num_policies = 35 if args.MOHopper2obj else 160
        env_id = "mo-hopper-v5" if not args.MOHopper2obj else "mo-hopper-2obj-v5"

        trajectories, true_labels, obj_feats_list = [], [], []
        for i in range(num_policies):
            file_path = os.path.join(trajectories_directory_path, f"policy_{i}.json")
            with open(file_path, 'r') as f:
                data = json.load(f)
            ret_vec = data.get('return', None)
            for states, actions in data['trajectories']:
                obs = np.array(list(states) + [states[-1]], dtype=np.float32)
                acts = np.array(actions, dtype=np.float32)
                if acts.ndim ==1:
                    acts = acts.reshape(-1,1)
                # print("Truncated HalfCheetah trajectories to length 100 for faster training.")
                traj = Trajectory(obs=obs, acts=acts, infos=None, terminal=True)
                trajectories.append(traj)
                true_labels.append(i)
                if ret_vec is not None:
                    obj_feats_list.append(np.asarray(ret_vec, dtype=np.float64))
    elif args.ResourceGathering:
        name_env = "resource_gathering"
        trajectories_directory_path = f"trajectories/{name_env}/"
        embeddings_folder_path = trajectories_directory_path + "embeddings/"
        os.makedirs(embeddings_folder_path, exist_ok=True)
        num_policies = 6
        env_id = "resource_gathering"
        
        # Dynamically find files based on their ID prefix
        all_json_files = sorted([f for f in os.listdir(trajectories_directory_path) if f.endswith('.json')])

        trajectories, true_labels, obj_feats_list = [], [], []
        for i in range(num_policies):
            # Find the file that starts with the current policy index
            try:
                file_name = next(f for f in all_json_files if f.startswith(f"{i}_"))
            except StopIteration:
                print(f"Warning: Could not find a trajectory file for policy ID {i} in {trajectories_directory_path}")
                continue

            file_path = os.path.join(trajectories_directory_path, file_name)
            with open(file_path, 'r') as f:
                data = json.load(f)
            ret_vec = data.get('return', None)
            for states, actions in data['trajectories']:
                obs = np.array(list(states) + [states[-1]], dtype=np.float32)
                acts = np.array(actions, dtype=np.float32)
                if acts.ndim ==1:
                    acts = acts.reshape(-1,1)
                traj = Trajectory(obs=obs, acts=acts, infos=None, terminal=True)
                trajectories.append(traj)
                true_labels.append(i)
                if ret_vec is not None:
                    obj_feats_list.append(np.asarray(ret_vec, dtype=np.float64))
    else:
        raise ValueError("Please select a valid environment")

    all_states_raw = [t.obs for t in trajectories]
    all_actions_raw = [t.acts for t in trajectories]

    # --- Normalization ---
    if args.state_scaler != 'none':
        ss = build_scaler(args.state_scaler)
        ss = fit_on_flat(all_states_raw, ss, fit_mode=args.scaler_fit)
        all_states_norm = apply_per_traj(all_states_raw, ss)
    else:
        all_states_norm = all_states_raw

    if args.action_scaler != 'none' and is_cont_actions(all_actions_raw):
        sa = build_scaler(args.action_scaler)
        sa = fit_on_flat(all_actions_raw, sa, fit_mode=args.scaler_fit)
        all_actions_norm = apply_per_traj(all_actions_raw, sa)
    else:
        all_actions_norm = all_actions_raw

    norm_trajectories = [Trajectory(obs=s, acts=a, infos=None, terminal=True) for s, a in zip(all_states_norm, all_actions_norm)]

    # --- Dataset and DataLoader ---
    all_states, all_actions, all_masks, all_labels, max_len = prepare_sa_trajectories(env_id, norm_trajectories, np.array(true_labels))
    if args.MOHighway or args.MOHopper or args.MOHopper2obj or args.ResourceGathering:
        timesteps = max_len - 1
        print("Trajectory timesteps:", timesteps)
    returns_per_traj = []
    policy_map = {i:[] for i in range(num_policies)}
    for i, label in enumerate(true_labels):
        policy_map[label].append(i)
    
    for i in range(len(trajectories)):
        policy_id = true_labels[i]
        returns_per_traj.append(obj_feats_list[policy_id])

    full_dataset, _, _, _, loader, _, _, _ = datasets_preparation_ret(
        all_states, all_actions, all_masks, all_labels,
        train_size=len(all_states), val_size=0, test_size=0,
        loader_batch=args.batch_size, val_bptt=1, test_bptt=1, seed=args.seed,
        returns=returns_per_traj
    )

    # --- Model, Losses, and Optimizer ---
    input_coord_dims = trajectories[0].obs.shape[1]
    print("Input coordinate dimensions:", input_coord_dims)
    print("Input action dimensions:", trajectories[0].acts.shape)
    num_actions = trajectories[0].acts.shape[1] if trajectories[0].acts.ndim > 1 else 1
    
    encoder = BehaviorEncoderCLSattnSATyped(
        input_channels=input_coord_dims, cnn_output_dim=args.emb_dim,steps=max_len, max_len=max_len,
        nhead=args.nheads, d_hid=args.d_hid, emb_dim=args.emb_dim,
        num_actions=num_actions, nlayers=args.nlayers, dropout=args.dropout,
        input_coord_dims=input_coord_dims
    ).to(device)

    decoder = TrajectoryDecoder(args.emb_dim, input_coord_dims, num_actions, max_len,spec_norm=args.spec_norm).to(device)
    info_loss_fn = InstanceLoss(args.temperature, device=device)
    dim_loss_fn = DeepInfoMaxLoss(args.emb_dim).to(device)

    params = list(encoder.parameters()) + list(decoder.parameters()) + list(dim_loss_fn.parameters())
    optim = th.optim.AdamW(params, lr=args.lr)

    # --- Training or Loading ---
    model_name = f"{args.model_prefix}_{env_code}_{args.emb_dim}d_{args.nheads}h_l{args.nlayers}_e{args.epochs}.pt"
    model_name = model_name.replace(".pt", f"_decorr{int(args.decorr_weight)}.pt")  if args.decorr_weight > 0.0 else model_name
    model_name = model_name.replace(".pt", f"_lower_t.pt") if args.temperature ==0.1 else  model_name.replace(".pt", f"_two_t.pt") if args.temperature ==0.2 else model_name.replace(".pt", f"_pf_t.pt") if args.temperature ==0.05 else model_name.replace(".pt", f"_h_t.pt") if args.temperature ==0.01 else model_name
    model_name = model_name.replace(".pt", "_specnorm.pt") if args.spec_norm else model_name
    model_name = model_name.replace(".pt", "_leastvol.pt") if args.least_volumes else model_name
    model_name = model_name.replace(".pt", f"_ts{timesteps}.pt") if args.MOHalfCheetah else model_name
    model_name = model_name.replace(".pt", "_shorter.pt") if args.shorter else model_name
    model_path = os.path.join(args.model_dir, model_name)
    print(encoder.model_type,"parameters ->",sum(p.numel() for p in encoder.parameters() if p.requires_grad)/1e6,"M")

    if args.train:
        print(f"Starting training for {args.epochs} epochs...")
        pbar = tqdm(range(args.epochs))
        for epoch in pbar:
            loss = train_epoch(
                encoder, decoder, loader, optim, device, info_loss_fn, dim_loss_fn,
                args.recon_weight, args.info_weight, args.dim_weight, args.vol_weight, args.decorr_weight, args.least_volumes, args.segment_weight, env_id
            )
            pbar.set_description(f"Epoch {epoch+1}/{args.epochs} | Loss: {loss:.4f}")
        os.makedirs(args.model_dir, exist_ok=True)
        th.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "dim_disc": dim_loss_fn.state_dict()}, model_path)
        print(f"Saved checkpoint to {model_path}")
    else:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Please train first with --train.")
        ckpt = th.load(model_path, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        print(f"Loaded checkpoint from {model_path}")

    # --- Evaluation and Visualization ---
    print("Evaluating embeddings on the full dataset...")
    embeddings, policies = get_all_embeddings(encoder, loader, device)

    unique = np.unique(policies)
    # per-policy mean and adjacent distances
    policy_means = np.array([embeddings[policies==pid].mean(0) for pid in unique])
    adj_dists = np.linalg.norm(policy_means[1:] - policy_means[:-1], axis=1)
    print("Adjacent mean embedding distances:", adj_dists)

    print("Visualizing raw trajectory embeddings (pre-aggregation)...")
    visualize_trajectory_embeddings(embeddings, policies, args.emb_dim)
    if args.viz_policy_ids is not None:
        print(f"Visualizing raw trajectory embeddings (pre-aggregation) highlighting policy {args.viz_policy_ids}...")
        visualize_trajectory_embeddings(embeddings, policies, args.emb_dim, highlight_pids=args.viz_policy_ids)

    print("Aggregating and visualizing policy embeddings...")
    image_dir = f"./images/no_topo_no_norm/{env_id}/"
    os.makedirs(image_dir, exist_ok=True)
    viz_path = os.path.join(image_dir, f"aggregated_embeddings_{args.emb_dim}d_e{args.epochs}.png")
    viz_path = viz_path.replace(".png", f"_decorr{int(args.decorr_weight)}.png") if args.decorr_weight > 0.0 else viz_path
    viz_path = viz_path.replace(".png", f"_lower_t.png") if args.temperature == 0.1 else viz_path.replace(".png", f"_two_t.png") if args.temperature == 0.2 else viz_path.replace(".png", f"_pf_t.png") if args.temperature == 0.05 else viz_path.replace(".png", f"_h_t.png") if args.temperature == 0.01 else viz_path
    viz_path = viz_path.replace(".png", "_specnorm.png") if args.spec_norm else viz_path
    viz_path = viz_path.replace(".png", "_leastvol.png") if args.least_volumes else viz_path
    viz_path = viz_path.replace(".png", f"_ts{timesteps}.png") if args.MOHalfCheetah else viz_path
    policy_latents = aggregate_and_visualize_policy_embeddings(
        embeddings, policies, args.emb_dim, save_path=viz_path if args.save_data else None
    )

    # --- Save aggregated embeddings in the detailed JSON format ---
    print("Saving aggregated policy latents to JSON...")
    
    def _to_list(x):
        if x is None:
            return None
        if isinstance(x, (th.Tensor, np.ndarray)):
            return x.tolist()
        # If it's already a list or other JSON-serializable type, return as is.
        if isinstance(x, (list, tuple, int, float, str)):
            return x
        # Fallback for other types, though ideally everything is covered above.
        return str(x)

    output_data = []
    unique_policies = np.unique(true_labels)

    for pid in unique_policies:
        # Find the index of the first trajectory for this policy
        representative_idx = np.where(true_labels == pid)[0][0]
        
        # Get the representative trajectory's data
        traj = trajectories[representative_idx]
        ret = obj_feats_list[representative_idx]
        
        # Get the aggregated embedding for this policy
        agg_embedding = policy_latents.get(int(pid))

        if agg_embedding is not None:
            entry = {
                "States": _to_list(traj.obs),
                "Actions": _to_list(traj.acts),
                "Return": _to_list(ret),
                "T_Embedding": _to_list(agg_embedding)
            }
            output_data.append(entry)

    json_path = os.path.join(embeddings_folder_path, f"{name_env}_{args.emb_dim}D_l{args.nlayers}_embeddings_e{args.epochs}.json")
    json_path = json_path.replace(".json", f"_decorr{int(args.decorr_weight)}.json") if args.decorr_weight > 0.0 else json_path
    json_path = json_path.replace(".json", f"_lower_t.json") if args.temperature == 0.1 else json_path.replace(".json", f"_two_t.json") if args.temperature == 0.2 else json_path.replace(".json", f"_pf_t.json") if args.temperature == 0.05 else json_path.replace(".json", f"_h_t.json") if args.temperature == 0.01 else json_path
    json_path = json_path.replace(".json", "_specnorm.json") if args.spec_norm else json_path
    json_path = json_path.replace(".json", "_leastvol.json") if args.least_volumes else json_path
    json_path = json_path.replace(".json", f"_ts{timesteps}.json") if args.MOHalfCheetah else json_path
    json_path = json_path.replace(".json", "_shorter.json") if args.shorter else json_path
    if args.save_data:
        with open(json_path, "w") as fh:
            json.dump(output_data, fh, indent=2)
        print(f"Saved aggregated policy latents to {json_path}")

    # -----------------------------------------------------------------
    # --- NEW: Wasserstein-based (Distributional) Analysis ---
    # -----------------------------------------------------------------
    print("\n" + "-"*30)
    print("--- Starting Wasserstein-based (Distributional) Analysis ---")

    # 1. Define K for neighborhood analysis
    # We choose a small k, e.g., 5 neighbors
    k_neighbors = 2
    if num_policies <= k_neighbors:
        print(f"Warning: k_neighbors={k_neighbors} is >= num_policies={num_policies}. Setting k=max(1, num_policies-2).")
        k_neighbors = max(1, num_policies - 2)

    # 2. Get unique policy IDs (already sorted by np.unique)
    unique_pids = np.unique(policies)
    N = len(unique_pids)
    
    # 3. Create the list of policy "point clouds" (distributions)
    # policy_clouds is a list of [n_trajs_per_policy, emb_dim] arrays
    print("Grouping trajectory embeddings by policy...")
    policy_clouds = [embeddings[policies == pid] for pid in unique_pids]
    emb_dim = policy_clouds[0].shape[1]

    # 4. Create the objective space vectors
    # (Ensuring they are in the same order as unique_pids)
    policy_objectives = np.array(
        [obj_feats_list[np.where(true_labels == pid)[0][0]] for pid in unique_pids]
    )
    # Ensure objectives are 2D
    if policy_objectives.ndim == 1:
        policy_objectives = policy_objectives.reshape(-1, 1)

    # 5. Pre-compute the Objective Space Distance Matrix (D_obj)
    print("Calculating N^2 Objective Distances (Euclidean)...")
    D_obj_precomputed = pairwise_distances(policy_objectives, metric='euclidean')

    # 6. Pre-compute the Behavioral Space Distance Matrix (D_emb_w)
    # We use a sum of 1D Wasserstein distances as a proxy for the multi-D distance
    # This avoids adding heavy new dependencies like POT.
    print(f"Calculating N^2 Behavioral Distances (Wasserstein, {emb_dim}D)...")
    D_emb_wasserstein = np.zeros((N, N))
    
    # Use tqdm for a progress bar on the outer loop
    pbar = tqdm(range(N), desc="Wasserstein Matrix")
    for i in pbar:
        for j in range(i + 1, N):
            dist = 0.0
            cloud_i = policy_clouds[i]
            cloud_j = policy_clouds[j]
            
            # Calculate 1D Wasserstein distance for each dimension and sum them
            for d in range(emb_dim):
                dist += wasserstein_distance(cloud_i[:, d], cloud_j[:, d])
            
            D_emb_wasserstein[i, j] = dist
            D_emb_wasserstein[j, i] = dist # The matrix is symmetric

    # 7. Calculate Distributional Trustworthiness and Continuity
    print("Calculating Trustworthiness and Continuity...")
    
    # Trustworthiness: How many of the k-nearest neighbors in D_obj
    # are preserved as neighbors in D_emb_w?
    trust_w = trustworthiness(
        D_obj_precomputed, D_emb_wasserstein, n_neighbors=k_neighbors, metric='precomputed'
    )
    
    # Continuity: How many of the k-nearest neighbors in D_emb_w
    # are preserved as neighbors in D_obj? (Just swap the matrices)
    cont_w = trustworthiness(
        D_emb_wasserstein, D_obj_precomputed, n_neighbors=k_neighbors, metric='precomputed'
    )

    print(f"  Distributional Trustworthiness (k={k_neighbors}): {trust_w:.4f}")
    print(f"  Distributional Continuity (k={k_neighbors}):     {cont_w:.4f}")

    # 8. Calculate and Plot the Distributional L-Plot
    print("Calculating and plotting Distributional L-Plot...")
    
    # Get adjacent-pair distances from the precomputed matrices
    adj_d_obj = np.array([D_obj_precomputed[i, i+1] for i in range(N - 1)])
    adj_d_emb_w = np.array([D_emb_wasserstein[i, i+1] for i in range(N - 1)])

    # Calculate the ratio, adding a small epsilon to prevent division by zero
    L_plot_wasserstein = adj_d_emb_w / (adj_d_obj + 1e-9)

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.bar(range(N - 1), L_plot_wasserstein, color='C1', alpha=0.7)
    plt.xlabel("Policy $i \\to i+1$")
    plt.ylabel("Lipschitz Constant (Wasserstein)")
    plt.title(f"Distributional Lipschitz Constants ({env_code})")
    plt.tight_layout()
    
    # Save the plot
    plot_save_path = os.path.join(image_dir, f"lipschitz_wasserstein_{args.emb_dim}d_e{args.epochs}.png")
    plt.savefig(plot_save_path)
    print(f"Saved Wasserstein L-Plot to {plot_save_path}")
    plt.show()

    print("Calculating and plotting Objective vs. Behavioral distance scatter plot...")

    # --- Plotting with Raw Distances and Independent Scales ---
    plt.figure(figsize=(8, 6))
    
    # --- Add Labels for Adjacent Policies ---
    # Get the unnormalized adjacent distances (already calculated for the L-plot)
    adj_d_obj = np.array([D_obj_precomputed[i, i+1] for i in range(N - 1)])
    adj_d_emb_w = np.array([D_emb_wasserstein[i, i+1] for i in range(N - 1)])

    # Highlight and label the adjacent points using raw distances
    plt.scatter(adj_d_obj, adj_d_emb_w, color='orange', s=60, edgecolors='k', zorder=5, label='Adjacent Pairs (i -> i+1)')
    for i in range(N - 1):
        plt.text(adj_d_obj[i], adj_d_emb_w[i], f" {i}→{i+1}", fontsize=9, va='bottom')

    # The y=x line and equal axis scaling have been removed to allow for independent scales.
    
    plt.xlabel("Objective Space Distance (Euclidean)")
    plt.ylabel("Behavioral Space Distance (Wasserstein)")
    plt.title(f"Raw Objective vs. Behavioral Distances ({env_code})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save the plot
    scatter_save_path = os.path.join(image_dir, f"distance_scatter_raw_{args.emb_dim}d_e{args.epochs}.png")
    plt.savefig(scatter_save_path)
    print(f"Saved Raw Distance Scatter Plot to {scatter_save_path}")
    plt.show()
    
    print("--- Distributional Analysis Complete ---")
    # --- End of NEW Section ---

if __name__ == "__main__":
    main()