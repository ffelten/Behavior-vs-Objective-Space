import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.utils import data
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
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
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler,
    QuantileTransformer, PowerTransformer, Normalizer
)
from sklearn.covariance import LedoitWolf
from scipy.stats import chi2
import sklearn.calibration as skcal
from sklearn.metrics import pairwise_distances
from collections import Counter
from sklearn.cluster import KMeans,SpectralClustering,DBSCAN, AgglomerativeClustering # type: ignore[import]
from functools import partial
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
from sklearn.cluster import KMeans,SpectralClustering,HDBSCAN,DBSCAN, AgglomerativeClustering
import umap #type: ignore[import]
from scipy.spatial import ConvexHull
from scipy.spatial.distance import jensenshannon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from sklearn.preprocessing import StandardScaler
from .behaviorencoder import *
from imitation.data.types import TrajectoryWithRew,Trajectory
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
    tau = 0.1 if env_id == "Traj2d" else 0.3 if env_id == "Reacher-v4" else 0.3 if env_id == "Pusherv4" else 0.0005 if env_id == "Walker2dv4" else 0.0005 if env_id == "Humanoidv4" else 0.1
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


def build_prefix_mask(masks: th.Tensor, prefix_fraction: float) -> th.Tensor:
    """
    Given original masks [B, T] (True = padded), return new masks that keep only the
    first ceil(prefix_fraction * valid_len) unmasked for each sequence.
    prefix_fraction in (0, 1], e.g., 0.3 keeps 30% of valid steps.
    """
    assert 0 < prefix_fraction <= 1.0, "prefix_fraction must be in (0, 1]."
    B, T = masks.shape
    new_masks = th.ones_like(masks)  # start all masked (True)
    for i in range(B):
        L = int((~masks[i]).sum().item())  # number of valid (unmasked) steps
        if L == 0:
            continue
        k = int(np.ceil(L * float(prefix_fraction)))
        k = max(1, min(L, k))
        new_masks[i, :k] = False
    return new_masks

def inference_on_dataloader_cdec_sa_prefix(
    encoder,
    cluster_centroids,
    dataloader,
    trajectory_manager,
    device,
    prefix_fraction: float,
    index_offset: int = 0,
    tag_suffix=None,
):
    """
    Same as inference_on_dataloader_cdec_sa, but only uses a prefix of each trajectory.
    The prefix is the first ceil(prefix_fraction * valid_len) steps according to the mask.

    Args:
      prefix_fraction: fraction of the valid (non-padded) length to keep in (0,1].
      tag_suffix: optional suffix added to trajectory_manager keys to avoid overwriting,
                  e.g., '_p10' for 10%, '_p50' for 50%. If None, defaults to f'_{int(100*prefix_fraction)}p'.
    """
    encoder.eval()
    if tag_suffix is None:
        tag_suffix = f"_{int(round(100*prefix_fraction))}p"

    all_concatenations = []
    all_indices = []

    with th.no_grad():
        for batch in tqdm(dataloader, desc=f"Inference (CDEC-SA prefix {int(round(100*prefix_fraction))}%)"):
            states, actions, masks, _, indices_batch = [b.to(device) for b in batch]

            pm = build_prefix_mask(masks, prefix_fraction)

            if encoder.model_type != "BE_SA_VDT":
                _, _, _, _, cls_emb, _ = encoder(states, actions, src_key_padding_mask=pm)
            else:
                _, _, _, _, cls_emb, _, _, _ = encoder(states, actions, src_key_padding_mask=pm)

            cls_emb = F.normalize(cls_emb, p=2, dim=1)

            sims = F.cosine_similarity(cls_emb.unsqueeze(1), cluster_centroids.unsqueeze(0), dim=-1)
            predicted_labels = sims.argmax(dim=1)

            all_concatenations.append(cls_emb.cpu())
            all_indices.append(indices_batch.cpu() + index_offset)

            for i in range(len(indices_batch)):
                traj_idx = indices_batch[i].item() + index_offset
                trajectory_manager[traj_idx][f'cls_emb{tag_suffix}'] = cls_emb[i].cpu().numpy()
                trajectory_manager[traj_idx][f'predicted_cluster_label{tag_suffix}'] = predicted_labels[i].item()
                trajectory_manager[traj_idx][f'concatenation{tag_suffix}'] = cls_emb[i].cpu().numpy()

    final_concatenations = th.cat(all_concatenations, dim=0)
    final_indices = th.cat(all_indices, dim=0).numpy()
    return trajectory_manager, final_concatenations, final_indices


def fit_hdbscan_seen(X_seen: np.ndarray, granularity: float, seed: int):
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

def build_registry(X_seen: np.ndarray, labels: np.ndarray, core_ids: np.ndarray, centers: np.ndarray) -> dict:
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

def assign_or_flag_online(X_online: np.ndarray, registry: dict, maha_p: float = 0.99):
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

def spawn_new_clusters_from_buffer(novel_points: np.ndarray, min_cluster_size: int = 5) -> list[dict]:
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

def plot_novelty_metrics(y_true, continuous_scores, hard_mask, out_prefix, title="", show=True):
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

def build_scaler(name: str, seed: int = 0, feature_range=(-1.0, 1.0), n_quantiles: int = 1000):
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

def is_coord_states(x) -> bool:
    x = np.asarray(x)
    return x.ndim == 2  # [T, D]

def is_cont_actions(a_list) -> bool:
    a0 = np.asarray(a_list[0])
    return a0.ndim == 2 and np.issubdtype(a0.dtype, np.floating)

def fit_on_flat(list_of_td, scaler, fit_mode='seen', list_of_td_online=None):
    # Flatten: stack all [T,D] -> [sum_T, D]
    X_seen = np.vstack([np.asarray(s) for s in list_of_td])
    if fit_mode == 'both' and list_of_td_online is not None:
        X_online = np.vstack([np.asarray(s) for s in list_of_td_online])
        X_fit = np.vstack([X_seen, X_online])
    else:
        X_fit = X_seen
    scaler.fit(X_fit)
    return scaler

def apply_per_traj(list_of_td, scaler):
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
