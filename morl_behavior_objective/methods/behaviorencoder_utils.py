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
    calinski_harabasz_score, adjusted_rand_score, accuracy_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    adjusted_mutual_info_score
)
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
from imitation.data.types import Trajectory
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
