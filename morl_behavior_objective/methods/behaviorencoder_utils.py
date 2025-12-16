import os
import json
from imitation.data.types import Trajectory  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np  # type: ignore[import]
import seaborn as sns  # type: ignore[import]
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)  # type: ignore[import]
import torch as th  # type: ignore[import]
from torch import nn  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]
from torch.utils.data import TensorDataset, DataLoader, random_split  # type: ignore[import]
from torch.utils.data import DataLoader  # type: ignore[import]
from tqdm import tqdm  # type: ignore[import]
import umap  # type: ignore[import]

from .behaviorencoder import *
from typing import List, Dict
import argparse
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import]


def prepare_sa_trajectories(
    env_id: str,
    trajectories: list[Trajectory],
    true_labels: np.ndarray,
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """Prepares state-action trajectories for a typed transformer model.
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
        all_actions = th.zeros((num_trajs, max_len, action_dim), dtype=th.float32)
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
                num_copy = min(len(acts_np), effective_len - 1)
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
            num_copy = min(len(acts_np), effective_len - 1)
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


def expand_interleaved_mask(original_mask: th.Tensor) -> th.Tensor:
    """Expands a mask for interleaved states and actions.

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
    pairwise_segments: bool = False,
    contrastive_loss_fn: nn.Module = InstanceLoss(0.5, device=th.device("cpu")),
):
    """Global-vs-Segment contrastive loss.
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
        seg_states = th.stack([states[b, idx_k[b] : idx_k[b] + L] for b in range(B)], dim=0)
        seg_actions = th.stack([actions[b, idx_k[b] : idx_k[b] + L] for b in range(B)], dim=0)
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


def build_scaler(name: str, seed: int = 0, feature_range=(-1.0, 1.0), n_quantiles: int = 1000):
    name = name.lower()
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler(feature_range=tuple(feature_range))
    if name == "maxabs":
        return MaxAbsScaler()
    if name == "robust":
        return RobustScaler(quantile_range=(25, 75))
    if name == "quantile_uniform":
        return QuantileTransformer(n_quantiles=n_quantiles, output_distribution="uniform", random_state=seed)
    if name == "quantile_normal":
        return QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal", random_state=seed)
    if name == "power_yeo":
        return PowerTransformer(method="yeo-johnson", standardize=True)
    if name == "l2norm":
        return Normalizer(norm="l2")
    if name == "none":
        return None
    raise ValueError(f"Unknown scaler: {name}")


def is_coord_states(x) -> bool:
    x = np.asarray(x)
    return x.ndim == 2  # [T, D]


def is_cont_actions(a_list) -> bool:
    a0 = np.asarray(a_list[0])
    return a0.ndim == 2 and np.issubdtype(a0.dtype, np.floating)


def fit_on_flat(list_of_td, scaler, fit_mode="seen", list_of_td_online=None):
    # Flatten: stack all [T,D] -> [sum_T, D]
    X_seen = np.vstack([np.asarray(s) for s in list_of_td])
    if fit_mode == "both" and list_of_td_online is not None:
        X_online = np.vstack([np.asarray(s) for s in list_of_td_online])
        X_fit = np.vstack([X_seen, X_online])
    else:
        X_fit = X_seen
    scaler.fit(X_fit)
    return scaler


def apply_per_traj(list_of_td, scaler):
    if scaler is None:
        return list_of_td
    out = []
    for s in list_of_td:
        s_np = np.asarray(s)
        if s_np.ndim != 2:
            out.append(s)  # leave grids or discrete as-is
            continue
        T, D = s_np.shape
        out.append(scaler.transform(s_np.reshape(-1, D)).reshape(T, D))
    return out


def datasets_preparation_ret(
    all_states,
    all_actions,
    all_masks,
    all_labels,
    train_size,
    val_size,
    test_size,
    loader_batch,
    val_bptt,
    test_bptt,
    seed,
    returns=None,
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
            full_dataset, [train_size, val_size, test_size], generator=th.Generator().manual_seed(seed)
        )
    else:
        train_set, val_set, test_set = full_dataset, None, None

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=loader_batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=val_bptt) if val_set else None
    test_loader = DataLoader(test_set, batch_size=test_bptt) if test_set else None

    return full_dataset, train_set, val_set, test_set, train_loader, val_loader, test_loader, None


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


# ---------- Loss Functions ----------
# Note: loss_vol_simplified and decorrelation_loss are no longer called
# They are left here for posterity but could be removed.
def loss_vol_simplified(z_normalized):
    # z_normalized are the embeddings already on the hypersphere
    s = z_normalized.std(0)
    eta = 1e-6  # Small constant to avoid log(0)
    # The geometric mean is still a valid measure of spread on the sphere
    return th.exp(th.log(s + eta).mean())


def decorrelation_loss(z):
    """
    Encourages different dimensions of the embeddings to be uncorrelated.
    Assumes z is centered (mean=0) across the batch.
    """
    # z shape: [B, emb_dim]
    z = z - z.mean(dim=0)  # Center the batch
    cov_matrix = (z.T @ z) / (len(z) - 1)  # [emb_dim, emb_dim]

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
    contrastive_loss_fn: nn.Module,
    num_segments: int = 2,
    pairwise_segments: bool = False,
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
        seg_masks = th.ones(B, L, dtype=th.bool, device=device)  # All padded initially

        for b in range(B):
            # Only process if a valid segment can be extracted
            if max_starts[b] > 0:
                start_idx = idx_k[b]
                end_idx = start_idx + L
                seg_states[b] = states[b, start_idx:end_idx]
                seg_actions[b] = actions[b, start_idx:end_idx]
                seg_masks[b, :] = False  # This segment is not padded

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



def train_epoch(
    encoder,
    decoder,
    loader,
    optim,
    device,
    info_loss_fn,
    dim_loss_fn,
    vc_loss_fn,
    recon_weight,
    info_weight,
    dim_weight,
    segment_weight=0.0,
    env_id=None,
    vc_weight=0.05,
    cls_norm=True
):
    encoder.train()
    decoder.train()
    dim_loss_fn.train()
    total_loss = 0.0

    # Check if encoder is MLP baseline (no CLS token in output)
    is_mlp_baseline = hasattr(encoder, 'model_type') and "MLP" in encoder.model_type


    for states, actions, masks, _, returns in loader:
        states, actions, masks = states.to(device), actions.to(device), masks.to(device)

        if actions[0][0].shape[0] == 1:
            discrete_actions = True
        else:
            discrete_actions = False

        # --- Two stochastic forward passes for contrastive learning ---
        all_tokens1, _, _, _, cls_emb1_raw, _ = encoder(states, actions, src_key_padding_mask=masks)
        all_tokens2, _, _, _, cls_emb2_raw, _ = encoder(states, actions, src_key_padding_mask=masks)
        if cls_norm:
            cls_emb1 = cls_emb1_raw
            cls_emb2 = cls_emb2_raw
        else:
            cls_emb1 = F.normalize(cls_emb1_raw, dim=1)
            cls_emb2 = F.normalize(cls_emb2_raw, dim=1)

        # Decoder can use either embedding, let's use the first one
        if cls_norm:
            states_rec, actions_rec = decoder(cls_emb1)
        else:
            states_rec, actions_rec = decoder(cls_emb1_raw)

        # --- IMPROVED: Mask-Aware Reconstruction Loss ---
        # Calculate valid lengths for each trajectory in the batch
        valid_lengths = (~masks).sum(dim=1)  # [B]

        # Compute loss only over valid positions for each trajectory
        rec_state_loss = 0.0
        rec_action_loss = 0.0
        total_valid_steps = 0

        B = states.size(0)
        for b in range(B):
            valid_len = int(valid_lengths[b].item())
            if valid_len == 0:
                continue

            # Loss for this trajectory (only over valid steps)
            rec_state_loss += (states_rec[b, :valid_len] - states[b, :valid_len]).pow(2).sum()
            if discrete_actions:
                # For discrete actions, use one-hot encoding for reconstruction loss
                act_logits = actions_rec[b, :valid_len]
                act_targets = actions[b, :valid_len].long().squeeze(-1)
                rec_action_loss += F.cross_entropy(act_logits, act_targets, reduction='sum')
                # print(f"Batch {b}, rec_action_loss: {rec_action_loss}")
            else:
                rec_action_loss += (actions_rec[b, :valid_len] - actions[b, :valid_len]).pow(2).sum()
            total_valid_steps += valid_len

        # Average over all valid steps in the batch
        if recon_weight > 0.0:
            rec_state_loss = rec_state_loss / max(total_valid_steps, 1)
            rec_action_loss = rec_action_loss / max(total_valid_steps, 1)
            recon_loss = rec_state_loss + rec_action_loss
        else:
            recon_loss = th.tensor(0.0, device=device)

        # --- CORRECTED InfoNCE Loss ---
        if info_weight > 0.0:
            info_loss = info_loss_fn(cls_emb1, cls_emb2)
        else:
            info_loss = th.tensor(0.0, device=device)

        # --- Deep InfoMax Loss (averaged over both views) ---
        interleaved_mask = expand_interleaved_mask(masks)

        if dim_weight > 0.0:
            if is_mlp_baseline:
                # MLP baseline: all_tokens already excludes any CLS-like token
                # Shape is [B, 2*T, D], mask is [B, 2*T]
                local_tokens1 = all_tokens1
                local_tokens2 = all_tokens2
                #for MLPT
                num_tokens = local_tokens1.shape[1]  # 2*num_local_tokens
                interleaved_mask = th.zeros(B, num_tokens, dtype=th.bool, device=device)
            else:
                # Transformer: first token is CLS, skip it for local tokens
                # Shape is [B, 1 + 2*T, D], we want [B, 2*T, D]
                local_tokens1 = all_tokens1[:, 1:, :]
                local_tokens2 = all_tokens2[:, 1:, :]
            dim_loss1 = dim_loss_fn(cls_emb1, local_tokens1, interleaved_mask)
            dim_loss2 = dim_loss_fn(cls_emb2, local_tokens2, interleaved_mask)
            dim_loss = (dim_loss1 + dim_loss2) / 2.0
        else:
            dim_loss = th.tensor(0.0, device=device)

        # --- Segment Contrastive Loss ---
        seg_loss = th.tensor(0.0, device=device)
        if segment_weight > 0.0:
            # Define segment length L based on environment
            L_min, L_max = (1, 4) if "dst" in env_id or "sea" in env_id else (8, 16)
            T = states.shape[1]
            current_L_max = min(L_max, T - 1)
            L = th.randint(L_min, current_L_max + 1, (1,)).item() if current_L_max >= L_min else L_min

        # vc_reg = (vc_loss_fn(cls_emb1_raw) + vc_loss_fn(cls_emb2_raw)) / 2.0
        if vc_weight > 0.0:
            vc_reg = vc_loss_fn(cls_emb1_raw)
        else:
            vc_reg = th.tensor(0.0, device=device)

        # --- Total Weighted Loss ---
        loss = (
            recon_weight * recon_loss
            + info_weight * info_loss
            + dim_weight * dim_loss
            + segment_weight * seg_loss
            + vc_weight * vc_reg
        )

        optim.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        th.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        th.nn.utils.clip_grad_norm_(dim_loss_fn.parameters(), 1.0)
        optim.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)

def train_epoch_vicreg_only(
    encoder,
    decoder,  # Kept for API compatibility, but not used
    loader,
    optim,
    device,
    info_loss_fn,  # Kept for API compatibility, but not used
    dim_loss_fn,   # Kept for API compatibility, but not used
    vc_loss_fn,    # Kept for API compatibility, but not used
    recon_weight,  # Ignored
    info_weight,   # Ignored
    dim_weight,    # Ignored
    segment_weight=0.0,  # Ignored
    env_id=None,   # Kept for API compatibility
    vc_weight=0.05,  # Ignored - we use VICReg weights instead
    cls_norm=True,  # Ignored - VICReg needs unnormalized embeddings
    # VICReg specific weights
    vicreg_inv_weight: float = 25.0,
    vicreg_var_weight: float = 50.0,
    vicreg_cov_weight: float = 5.0,
):
    """
    Training epoch using ONLY VICReg loss (Invariance + Variance + Covariance).
    
    The two forward passes with dropout act as the two "augmented views".
    
    Args:
        encoder: The behavior encoder model
        decoder: Not used, kept for API compatibility
        loader: DataLoader
        optim: Optimizer
        device: torch device
        info_loss_fn: Not used, kept for API compatibility
        dim_loss_fn: Not used, kept for API compatibility
        vc_loss_fn: Not used, kept for API compatibility
        recon_weight: Ignored
        info_weight: Ignored
        dim_weight: Ignored
        segment_weight: Ignored
        env_id: Not used, kept for API compatibility
        vc_weight: Ignored
        cls_norm: Ignored (VICReg needs raw embeddings)
        vicreg_inv_weight: Weight for invariance loss (default: 25.0)
        vicreg_var_weight: Weight for variance loss (default: 25.0)
        vicreg_cov_weight: Weight for covariance loss (default: 1.0)
        
    Returns:
        avg_loss: Average loss over the epoch
    """
    encoder.train()
    total_loss = 0.0
    total_inv = 0.0
    total_var = 0.0
    total_cov = 0.0
    total_info = 0.0
    num_batches = 0
    
    eps = 1e-4

    for states, actions, masks, _, returns in loader:
        states, actions, masks = states.to(device), actions.to(device), masks.to(device)
        num_batches += 1

        # --- Two stochastic forward passes (dropout creates two "views") ---
        _, _, _, _, cls_emb1_raw, _ = encoder(states, actions, src_key_padding_mask=masks)
        _, _, _, _, cls_emb2_raw, _ = encoder(states, actions, src_key_padding_mask=masks)
        
        # VICReg operates on RAW (unnormalized) embeddings
        z1 = cls_emb1_raw
        z2 = cls_emb2_raw
        B, D = z1.shape

        # === 1. Invariance Loss ===
        # MSE between the two views (same sample should have same embedding)
        inv_loss = F.mse_loss(z1, z2)

        # === 2. Variance Loss ===
        # Force std of each dimension >= 1 (across batch)
        std_z1 = th.sqrt(z1.var(dim=0) + eps)
        std_z2 = th.sqrt(z2.var(dim=0) + eps)
        var_loss = th.mean(F.relu(1 - std_z1)) + th.mean(F.relu(1 - std_z2))

        # === 3. Covariance Loss ===
        # Force off-diagonal elements of covariance matrix to be zero
        z1_centered = z1 - z1.mean(dim=0)
        z2_centered = z2 - z2.mean(dim=0)

        cov_z1 = (z1_centered.T @ z1_centered) / (B - 1)  # [D, D]
        cov_z2 = (z2_centered.T @ z2_centered) / (B - 1)  # [D, D]

        # Off-diagonal elements
        off_diag_mask = ~th.eye(D, dtype=th.bool, device=z1.device)
        cov_loss = (cov_z1[off_diag_mask].pow(2).sum() + cov_z2[off_diag_mask].pow(2).sum()) / D

        #InfoNCE loss l2 normalize embeddings
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)
        info_loss = info_loss_fn(z1_norm, z2_norm)

        # === Total Loss ===
        loss = (
            vicreg_inv_weight * inv_loss +
            vicreg_var_weight * var_loss +
            vicreg_cov_weight * cov_loss +
            info_weight * info_loss
        )

        optim.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optim.step()
        
        total_loss += loss.item()
        total_inv += inv_loss.item()
        total_var += var_loss.item()
        total_cov += cov_loss.item()
        total_info += info_loss.item()

    avg_loss = total_loss / max(num_batches, 1)
    
    # Optional: Print breakdown every epoch
    # print(f"  VICReg breakdown - Inv: {total_inv/num_batches:.4f}, "
    #       f"Var: {total_var/num_batches:.4f}, Cov: {total_cov/num_batches:.4f}")

    return avg_loss

def train_epoch_no_norm(
    encoder,
    decoder,
    loader,
    optim,
    device,
    info_loss_fn,
    dim_loss_fn,
    recon_weight,
    info_weight,
    dim_weight,
    segment_weight=0.0,
    env_id=None,
):  # Removed vol_weight, decorr_weight, least_volumes
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

        # --- Other losses (removed) ---
        # decorr_loss = th.tensor(0.0, device=device)
        # vol_loss = th.tensor(0.0, device=device)

        # --- Segment Contrastive Loss ---
        seg_loss = th.tensor(0.0, device=device)
        if segment_weight > 0.0:
            # Define segment length L based on environment
            L_min, L_max = (4, 12) if "dst" in env_id or "sea" in env_id else (8, 16)
            T = states.shape[1]
            current_L_max = min(L_max, T - 1)
            L = th.randint(L_min, current_L_max + 1, (1,)).item() if current_L_max >= L_min else L_min

        # --- Total Weighted Loss ---
        loss = recon_weight * recon_loss + info_weight * info_loss + dim_weight * dim_loss + segment_weight * seg_loss

        optim.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        th.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optim.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def set_train_epoch(
    set_encoder: PolicySetEncoder,
    policy_data: Dict,
    set_optim: th.optim.Optimizer,
    info_loss_fn_set: InstanceLoss,
    dim_loss_fn_set: DeepInfoMaxLoss,
    set_info_weight: float,
    set_dim_weight: float,
) -> Tuple[float, float, float]:
    """
    Performs a single training epoch for the PolicySetEncoder.
    """
    set_encoder.train()
    dim_loss_fn_set.train()
    set_optim.zero_grad()

    total_dim_loss_val = 0.0
    total_loss_for_epoch = 0.0
    z_p_views_1, z_p_views_2 = [], []

    pids_shuffled = np.random.permutation(list(policy_data.keys()))
    valid_pids = 0

    for pid in pids_shuffled:
        data = policy_data[pid]
        emb_set = data["embeddings"]

        if emb_set.shape[0] < 1:  # Safety check (policy has no trajs)
            continue
        valid_pids += 1

        # --- Two stochastic forward passes ---
        # z_p: [1, D], all_tokens: [1, 1+N, D]
        z_p_v1, all_tokens_v1 = set_encoder(emb_set)
        z_p_v2, all_tokens_v2 = set_encoder(emb_set)

        # --- Store for InfoNCE ---
        z_p_views_1.append(z_p_v1)
        z_p_views_2.append(z_p_v2)

        # --- Calculate DIM Loss ---
        # Get local tokens (excluding [CLS] token)
        local_tokens_v1 = all_tokens_v1[:, 1:, :]  # Shape: [1, N, D]
        local_tokens_v2 = all_tokens_v2[:, 1:, :]  # Shape: [1, N, D]

        # --- THIS IS THE FIX ---
        # 1. Create a "no padding" mask.
        # Shape must be [B, T] -> [1, N]
        B1, T1, _ = local_tokens_v1.shape
        mask_v1 = th.zeros(B1, T1, dtype=th.bool, device=local_tokens_v1.device)

        B2, T2, _ = local_tokens_v2.shape
        mask_v2 = th.zeros(B2, T2, dtype=th.bool, device=local_tokens_v2.device)

        # 2. Call the loss function WITHOUT squeeze() and WITH the mask.
        dim_loss_1 = dim_loss_fn_set(z_p_v1, local_tokens_v1, mask_v1)
        dim_loss_2 = dim_loss_fn_set(z_p_v2, local_tokens_v2, mask_v2)
        # --- END OF FIX ---

        loss_dim = (dim_loss_1 + dim_loss_2) / 2
        total_dim_loss_val += loss_dim.item()

        # Add to total loss, scaled
        total_loss_for_epoch += set_dim_weight * loss_dim

    if valid_pids == 0:
        print("Warning: No valid policies found in epoch.")
        return 0.0, 0.0, 0.0

    # Average the total DIM loss
    total_loss_for_epoch = total_loss_for_epoch / valid_pids

    # --- Calculate InfoNCE loss ---
    # Stack all collected views
    Z1 = th.cat(z_p_views_1, dim=0)  # [Num_Policies, D]
    Z2 = th.cat(z_p_views_2, dim=0)  # [Num_Policies, D]

    loss_infonce = info_loss_fn_set(Z1, Z2)

    # Add to total loss
    total_loss_for_epoch += set_info_weight * loss_infonce

    # --- 3. Single Backward pass and Step ---
    total_loss_for_epoch.backward()
    set_optim.step()

    avg_dim_loss = total_dim_loss_val / valid_pids

    return total_loss_for_epoch.item(), avg_dim_loss, loss_infonce.item()


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


def visualize_trajectory_embeddings(
    embeddings, policies, emb_dim, title="Trajectory Embeddings (pre-aggregation)", highlight_pids=None
):
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
        ax.set_xlabel("UMAP Dim 1")
        ax.set_ylabel("UMAP Dim 2")
        ax.set_zlabel("UMAP Dim 3")
    elif emb_dim == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=colors, alpha=0.5)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")
    else:  # emb_dim == 2
        ax = fig.add_subplot(111)
        ax.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, alpha=0.5)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

    # Add labels for highlighted policies
    if highlight_pids is not None:
        print(f"Highlighting policy IDs: {highlight_pids}")
        for i, pid in enumerate(policies):
            if pid in highlight_pids:
                point = plot_embeddings[i]
                if emb_dim == 2:
                    ax.text(
                        point[0], point[1], str(pid), color="black", fontsize=9, ha="center", va="center", weight="bold"
                    )
                else:  # 3D or UMAP 3D
                    ax.text(
                        point[0],
                        point[1],
                        point[2],
                        str(pid),
                        color="black",
                        fontsize=9,
                        ha="center",
                        va="center",
                        weight="bold",
                    )

    # Create dummy artists for legend
    for pid in unique_policies:
        ax.scatter([], [], c=[color_map[pid]], label=f"Policy {pid}")

    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.show()


def visualize_policy_embeddings(
    policy_latents: Dict[int, np.ndarray], emb_dim: int, title: str = "Aggregated Policy Embeddings", save_path: str = None
):
    """
    Visualizes a dictionary of aggregated policy embeddings.
    """
    if emb_dim < 2:
        print(f"Embedding dimension < 2; skipping visualization for '{title}'.")
        return

    agg_embeddings = np.array(list(policy_latents.values()))
    agg_ids = np.array(list(policy_latents.keys()))

    # Ensure embeddings are 2D
    if agg_embeddings.ndim == 1:
        agg_embeddings = agg_embeddings.reshape(-1, 1)

    unique_policies = np.unique(agg_ids)
    palette = sns.color_palette("tab10", n_colors=max(len(unique_policies), 3))
    color_map = {pid: palette[i % len(palette)] for i, pid in enumerate(unique_policies)}

    fig = plt.figure(figsize=(8, 6))

    # Handle different dimensions for plotting
    plot_title = title
    if emb_dim > 3:
        plot_title = f"{title} ({emb_dim}D -> 3D UMAP)"
        print(f"Reducing {emb_dim}D -> 3D with UMAP for '{title}' visualization.")
        reducer = umap.UMAP(n_components=3, random_state=42)
        plot_embeddings = reducer.fit_transform(agg_embeddings)
        ax = fig.add_subplot(111, projection="3d")
        for i, pid in enumerate(agg_ids):
            pts = plot_embeddings[i, :]
            ax.scatter(pts[0], pts[1], pts[2], c=[color_map[pid]], label=f"Policy {pid}", s=100, alpha=0.9)
            ax.text(pts[0], pts[1], pts[2], str(pid), color="black", fontsize=12, ha="right", va="bottom")
        ax.set_xlabel("UMAP Dim 1")
        ax.set_ylabel("UMAP Dim 2")
        ax.set_zlabel("UMAP Dim 3")
    elif emb_dim == 3:
        plot_title = f"{title} (3D)"
        ax = fig.add_subplot(111, projection="3d")
        for i, pid in enumerate(agg_ids):
            pts = agg_embeddings[i, :3]
            ax.scatter(pts[0], pts[1], pts[2], c=[color_map[pid]], label=f"Policy {pid}", s=100, alpha=0.9)
            ax.text(pts[0], pts[1], pts[2], str(pid), color="black", fontsize=12, ha="right", va="bottom")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")
    else:  # emb_dim == 2
        plot_title = f"{title} (2D)"
        ax = fig.add_subplot(111)
        for i, pid in enumerate(agg_ids):
            pts = agg_embeddings[i, :2]
            ax.scatter(pts[0], pts[1], c=[color_map[pid]], label=f"Policy {pid}", s=100, alpha=0.9)
            ax.text(pts[0], pts[1], str(pid), color="black", fontsize=12, ha="right", va="bottom")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

    ax.set_title(plot_title)
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved aggregated embedding visualization to {save_path}")
    # plt.show()
    plt.close()


def visualize_policy_embeddings_with_lines(
    policy_latents: Dict[int, np.ndarray], emb_dim: int, title: str = "Aggregated Policy Embeddings", save_path: str = None
):
    """
    Visualizes a dictionary of aggregated policy embeddings with CONNECTING LINES.
    """
    if emb_dim < 2:
        print(f"Embedding dimension < 2; skipping visualization for '{title}'.")
        return

    # 1. Extract and Sort by Policy ID (Crucial for lines!)
    sorted_pids = sorted(policy_latents.keys())
    agg_embeddings = np.array([policy_latents[pid] for pid in sorted_pids])

    # Ensure embeddings are 2D
    if agg_embeddings.ndim == 1:
        agg_embeddings = agg_embeddings.reshape(-1, 1)

    unique_policies = np.unique(sorted_pids)
    palette = sns.color_palette("tab10", n_colors=max(len(unique_policies), 3))
    # Use modulo to cycle colors if K > 10
    color_map = {pid: palette[i % len(palette)] for i, pid in enumerate(unique_policies)}

    fig = plt.figure(figsize=(8, 6))

    # Handle different dimensions for plotting
    plot_title = title

    # Check if we need UMAP (for >3 dims) or direct plotting
    if emb_dim > 3:
        plot_title = f"{title} ({emb_dim}D -> 3D UMAP)"
        print(f"Reducing {emb_dim}D -> 3D with UMAP for '{title}' visualization.")
        reducer = umap.UMAP(n_components=3, random_state=42)
        plot_embeddings = reducer.fit_transform(agg_embeddings)
        ax = fig.add_subplot(111, projection="3d")

        # --- DRAW LINES (UMAP Space) ---
        ax.plot(
            plot_embeddings[:, 0],
            plot_embeddings[:, 1],
            plot_embeddings[:, 2],
            color="gray",
            alpha=0.5,
            linewidth=1.5,
            linestyle="--",
        )

        for i, pid in enumerate(sorted_pids):
            pts = plot_embeddings[i, :]
            ax.scatter(pts[0], pts[1], pts[2], c=[color_map[pid]], label=f"Policy {pid}", s=100, alpha=0.9)
            ax.text(pts[0], pts[1], pts[2], str(pid), color="black", fontsize=12, ha="right", va="bottom")
        ax.set_xlabel("UMAP Dim 1")
        ax.set_ylabel("UMAP Dim 2")
        ax.set_zlabel("UMAP Dim 3")

    elif emb_dim == 3:
        plot_title = f"{title} (3D)"
        ax = fig.add_subplot(111, projection="3d")

        # --- DRAW LINES (Direct 3D Space) ---
        # This connects P0 -> P1 -> P2 ...
        ax.plot(
            agg_embeddings[:, 0],
            agg_embeddings[:, 1],
            agg_embeddings[:, 2],
            color="black",
            alpha=0.4,
            linewidth=1,
            linestyle="--",
        )

        for i, pid in enumerate(sorted_pids):
            pts = agg_embeddings[i, :3]
            ax.scatter(pts[0], pts[1], pts[2], c=[color_map[pid]], label=f"Policy {pid}", s=100, alpha=0.9)
            ax.text(pts[0], pts[1], pts[2], str(pid), color="black", fontsize=12, ha="right", va="bottom")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")

    else:  # emb_dim == 2
        plot_title = f"{title} (2D)"
        ax = fig.add_subplot(111)

        # --- DRAW LINES (2D Space) ---
        ax.plot(agg_embeddings[:, 0], agg_embeddings[:, 1], color="gray", alpha=0.5, linewidth=1.5, linestyle="--")

        for i, pid in enumerate(sorted_pids):
            pts = agg_embeddings[i, :2]
            ax.scatter(pts[0], pts[1], c=[color_map[pid]], label=f"Policy {pid}", s=100, alpha=0.9)
            ax.text(pts[0], pts[1], str(pid), color="black", fontsize=12, ha="right", va="bottom")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

    ax.set_title(plot_title)
    # ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5)) # Optional: comment out if legend is too big
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved aggregated embedding visualization to {save_path}")
    plt.show()
    plt.close()


def get_mean_policy_embeddings(embeddings: np.ndarray, policies: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Aggregates policy embeddings by *mean*.
    Returns a dictionary of the mean embeddings.
    """
    policy_latents = {}
    unique_policies = np.unique(policies)
    for pid in unique_policies:
        policy_latents[int(pid)] = embeddings[policies == pid].mean(axis=0)
    return policy_latents


def load_environment_data(args: argparse.Namespace) -> Tuple:
    """
    Loads trajectories, labels, and objective features based on args.
    Assumes all envs use 'policy_{id}.json' naming.
    """
    trajectories, true_labels, obj_feats_list = [], [], []

    # --- 1. Define base paths and metadata ---
    if args.DeepSeaTreasureConcave:
        name_env = "dst_concave"
        num_policies = 10
        env_id = "deep-sea-treasure-v0"
        base_trajectories_path = f"trajectories/{name_env}/"
        is_seeded_env = False
    elif args.DeepSeaTreasureSmooth:
        name_env = "smooth"
        num_policies = 10
        env_id = "dst-smooth-v0"
        base_trajectories_path = f"trajectories/{name_env}/"
        is_seeded_env = False
    elif args.DeepSeaTreasureLeftRight:
        name_env = "left_right_dst"
        num_policies = 6
        env_id = "left-right-dst-v0"
        base_trajectories_path = f"trajectories/{name_env}/"
        is_seeded_env = False
    elif args.DeepSeaTreasureLouvre:
        name_env = "louvre_dst"
        num_policies = 10
        env_id = "louvre_dst"
        base_trajectories_path = f"trajectories/{name_env}/"
        is_seeded_env = False
    elif args.MOHalfCheetah:
        name_env = "mo-halfcheetah-v5"
        num_policies = (
            80 #if args.seed == 0 else 99 if args.seed == 1 else 92 if args.seed == 2 else 103 if args.seed == 3 else 93
        )
        env_id = "mo-halfcheetah-v5"
        base_trajectories_path = f"trajectories/morld/{name_env}/"
        is_seeded_env = True
    elif args.MOHighway:
        name_env = "mo-highway-fast-v0"
        num_policies = (
            41 #if args.seed == 0 else 41 if args.seed == 1 else 40 if args.seed == 2 else 43 if args.seed == 3 else 51
        )
        env_id = "mo-highway-fast-v0"
        base_trajectories_path = f"trajectories/morld/{name_env}/"  # Highway's base path
        is_seeded_env = True
    elif args.MOHopper:
        name_env = "mo-hopper-v5"
        num_policies = (
            160 #if args.seed == 0 else 157 if args.seed == 1 else 144 if args.seed == 2 else 182 if args.seed == 3 else 203
        )
        env_id = "mo-hopper-v5"
        base_trajectories_path = f"trajectories/morld/{name_env}/"
        is_seeded_env = True
    elif args.MOHopper2obj:
        name_env = "mo-hopper-2obj-v5"
        num_policies = (
            35 #if args.seed == 0 else 26 if args.seed == 1 else 32 if args.seed == 2 else 24 if args.seed == 3 else 25
        )
        env_id = "mo-hopper-2obj-v5"
        base_trajectories_path = f"trajectories/morld/{name_env}/"
        is_seeded_env = True
    elif args.ResourceGathering:
        name_env = "resource_gathering"
        num_policies = 6
        env_id = "resource_gathering"
        base_trajectories_path = f"trajectories/{name_env}/"
        is_seeded_env = False
    else:
        raise ValueError("Please select a valid environment")

    # --- 2. Define specific paths for loading and saving ---

    # Path to load trajectories from (e.g., .../seed0/)
    if is_seeded_env:
        # trajectories_directory_path = os.path.join(base_trajectories_path, f"seed{args.seed}")
        trajectories_directory_path = os.path.join(base_trajectories_path, f"seed0") # using only seed0 trajectories we can evaluate policies over different transformers
    else:
        trajectories_directory_path = base_trajectories_path

    # Path to save embeddings to (e.g., .../embeddings/)
    # This now correctly uses the base path, so it's a sibling to seed{s}
    embeddings_folder_path = os.path.join(base_trajectories_path, "embeddings")
    os.makedirs(embeddings_folder_path, exist_ok=True)

    print(f"Loading data from: {trajectories_directory_path}")
    print(f"Saving embeddings to: {embeddings_folder_path}")

    # --- 3. Generalized Loading Loop ---
    for i in range(num_policies):
        file_path = os.path.join(trajectories_directory_path, f"policy_{i}.json")
        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}")
            continue

        with open(file_path, "r") as f:
            data = json.load(f)
        ret_vec = data.get("return", None)

        # Store the return vector *once* per policy
        if ret_vec is not None:
            obj_feats_list.append(np.asarray(ret_vec, dtype=np.float64))

        if args.DeepSeaTreasureConcave or args.DeepSeaTreasureSmooth or args.DeepSeaTreasureLeftRight:
            loaded_trajectories = data["trajectories"]
        else:
            loaded_trajectories = data["trajectories"]

        for states, actions in loaded_trajectories:
            obs = np.array(list(states) + [states[-1]], dtype=np.float32)
            # acts = np.array(actions, dtype=np.float32)
            if is_seeded_env:
                acts = np.array(actions, dtype=np.float32)
            else:
                acts = np.array(actions)
            if acts.ndim == 1:
                acts = acts.reshape(-1, 1)
            traj = Trajectory(obs=obs, acts=acts, infos=None, terminal=True)
            trajectories.append(traj)
            true_labels.append(i)  # The policy ID is 'i'

    if len(obj_feats_list) != num_policies:
        # Fallback in case some files were missing or didn't have a 'return' key
        print(f"Warning: Found {len(obj_feats_list)} return vectors, expected {num_policies}.")
        # Try to fill missing ones if possible
        if not obj_feats_list:  # Handle case where no returns were found
            obj_feats_list = [np.array([0.0, 0.0])]  # Add a dummy one
        while len(obj_feats_list) < num_policies:
            obj_feats_list.append(obj_feats_list[0])  # Just duplicate the first one

    # Map the per-policy returns to each trajectory
    policy_to_return = {i: obj_feats_list[i] for i in range(len(obj_feats_list))}
    obj_feats_per_traj = [policy_to_return[label] for label in true_labels]

    return trajectories, true_labels, obj_feats_per_traj, env_id, name_env, num_policies, embeddings_folder_path


# ---------- Clean Filename Generation ----------
def generate_experiment_name(args: argparse.Namespace, env_code: str) -> str:
    """
    Generates a consistent, unique base name for models and outputs.
    Removed fixed hyperparameters from name.
    """
    name = f"{args.model_prefix}_{env_code}_e{args.epochs}"
    return name


# ---------- JSON Saving Helper ----------
def _to_list(x):
    """Helper to convert items to JSON-serializable lists."""
    if x is None:
        return None
    if isinstance(x, (th.Tensor, np.ndarray)):
        return x.tolist()
    if isinstance(x, (list, tuple, int, float, str, dict)):
        return x
    return str(x)


def save_policy_latents_to_json(
    latents_dict: Dict[int, list],
    trajectories: List[Trajectory],
    true_labels: List[int],
    obj_feats_list: List[np.ndarray],
    file_path: str,
):
    """
    Saves aggregated policy latents to a JSON file in the specified format.

    Args:
        latents_dict: {policy_id: aggregated_embedding_list}
        trajectories: The original list of all trajectories
        true_labels: The list of policy IDs for each trajectory
        obj_feats_list: The list of objective returns for each trajectory
        file_path: The full path to save the JSON file.
    """
    print(f"Saving aggregated policy latents to {file_path}...")
    output_data = []

    # Get a map of policy_id -> index of first trajectory
    policy_id_to_first_idx = {label: i for i, label in reversed(list(enumerate(true_labels)))}

    for pid_int, agg_embedding in latents_dict.items():
        pid = int(pid_int)  # Ensure key is int
        # Find the representative trajectory's data
        try:
            representative_idx = policy_id_to_first_idx[pid]
        except KeyError:
            print(f"Warning: No trajectory found for policy ID {pid}. Skipping save.")
            continue

        traj = trajectories[representative_idx]
        ret = obj_feats_list[representative_idx]

        entry = {
            "States": _to_list(traj.obs),
            "Actions": _to_list(traj.acts),
            "Return": _to_list(ret),
            "T_Embedding": _to_list(agg_embedding),
        }
        output_data.append(entry)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as fh:
        json.dump(output_data, fh, indent=2)
    print(f"Saved {len(output_data)} policy latents to {file_path}")
