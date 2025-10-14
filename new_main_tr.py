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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from torch.utils.data import DataLoader

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

# ---------- Decoder Definition ----------
class TrajectoryDecoder(nn.Module):
    def __init__(self, emb_dim, state_dim, action_dim, max_len):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_len = max_len
        self.out_features = max_len * (state_dim + action_dim)

        self.net = nn.Sequential(
            nn.Linear(emb_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, self.out_features),
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
    from torch.utils.data import TensorDataset, DataLoader, random_split

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
    recon_weight, info_weight, dim_weight, topo_weight
):
    encoder.train()
    decoder.train()
    dim_loss_fn.train()
    total_loss = 0.0

    for states, actions, masks, _, returns in loader:
        states, actions, masks = states.to(device), actions.to(device), masks.to(device)

        # Encoder returns the full sequence of tokens AND the specific CLS embedding
        all_tokens, _, _, _, cls_emb, _ = encoder(states, actions, src_key_padding_mask=masks)
        
        # Decoder now takes ONLY the CLS embedding to reconstruct the trajectory
        states_rec, actions_rec = decoder(cls_emb)

        # Reconstruction Loss
        # The shapes should now match: states_rec is [B, T, D_s] and states is [B, T, D_s]
        mask_flat = masks.view(masks.size(0), -1).unsqueeze(-1)
        valid = (~mask_flat).float()
        
        # Ensure target shapes are correct for comparison
        tgt_states = states
        tgt_actions = actions

        rec_state = ((states_rec - tgt_states).pow(2) * valid).sum() / valid.sum().clamp(min=1.0)
        rec_action = ((actions_rec - tgt_actions).pow(2) * valid).sum() / valid.sum().clamp(min=1.0)
        recon_loss = rec_state + rec_action

        # InfoNCE and DIM Loss (these still use the full token sequences)
        info_loss = info_loss_fn(cls_emb, cls_emb)
        # The DeepInfoMax loss needs the full sequence of tokens from the encoder
        interleaved_mask = expand_interleaved_mask(masks)
        dim_loss = dim_loss_fn(cls_emb, all_tokens[:, 1:, :], interleaved_mask) # Exclude CLS from local tokens

        # Topology Loss
        topo_loss = th.tensor(0.0, device=device)
        return_vals = extract_scalar_returns(returns)
        if return_vals is not None and len(return_vals) >= 3:
            ret_tensor = th.tensor(return_vals, device=device, dtype=th.float32)
            # Sort embeddings based on their scalar return
            order = th.argsort(ret_tensor)
            
            # Calculate difference vectors between adjacent embeddings in the sorted sequence
            diffs = cls_emb[order[1:]] - cls_emb[order[:-1]]
            
            if diffs.size(0) > 1:
                # Penalize deviations from a straight line by maximizing cosine similarity
                # between consecutive difference vectors.
                # 1 - cos(theta) is minimized when theta is 0 (vectors are parallel).
                topo_loss = (
                    1.0
                    - F.cosine_similarity(diffs[:-1], diffs[1:], dim=-1).mean()
                )

        # Total Weighted Loss
        loss = (
            recon_weight * recon_loss
            + info_weight * info_loss
            + dim_weight * dim_loss
            + topo_weight * topo_loss
        )

        optim.zero_grad()
        loss.backward()
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

def aggregate_and_visualize_policy_embeddings(
    embeddings, policies, emb_dim, save_path=None
):
    policy_latents = {}
    unique_policies = np.unique(policies)
    for pid in unique_policies:
        policy_latents[int(pid)] = embeddings[policies == pid].mean(axis=0)

    if emb_dim < 3:
        print("Embedding dimension < 3; skipping 3D visualization.")
        return policy_latents

    agg_embeddings = np.array(list(policy_latents.values()))
    agg_ids = np.array(list(policy_latents.keys()))

    palette = sns.color_palette("tab10", n_colors=max(len(unique_policies), 3))
    color_map = {pid: palette[i % len(palette)] for i, pid in enumerate(unique_policies)}

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    for i, pid in enumerate(agg_ids):
        pts = agg_embeddings[i, :3]
        ax.scatter(
            pts[0], pts[1], pts[2],
            c=[color_map[pid]], label=f"Policy {pid}", s=100, alpha=0.9
        )

    ax.set_title("Aggregated Policy Embeddings (3D)")
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); ax.set_zlabel("Dim 3")
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
    # Add other envs if needed...

    arg_hyp = parser.add_argument_group('Training Hyperparameters')
    arg_hyp.add_argument("--traj_dir", default="trajectories/dst_concave")
    arg_hyp.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    arg_hyp.add_argument("--epochs", type=int, default=200)
    arg_hyp.add_argument("--batch_size", type=int, default=32)
    arg_hyp.add_argument("--lr", type=float, default=3e-4)
    arg_hyp.add_argument("--emb_dim", type=int, default=3)
    arg_hyp.add_argument("--nheads", type=int, default=3)
    arg_hyp.add_argument("--nlayers", type=int, default=1)
    arg_hyp.add_argument("--d_hid", type=int, default=1024)
    arg_hyp.add_argument("--dropout", type=float, default=0.1)
    arg_hyp.add_argument("--recon_weight", type=float, default=1.0)
    arg_hyp.add_argument("--info_weight", type=float, default=0.1)
    arg_hyp.add_argument("--dim_weight", type=float, default=0.1)
    arg_hyp.add_argument("--topo_weight", type=float, default=0.2)
    arg_hyp.add_argument("--temperature", type=float, default=0.1)
    arg_hyp.add_argument("--state_scaler", default="quantile_normal")
    arg_hyp.add_argument("--action_scaler", default="quantile_normal")
    arg_hyp.add_argument("--scaler_fit", default="seen", choices=["seen", "both"])
    arg_hyp.add_argument("--model_dir", default="models/")
    arg_hyp.add_argument("--model_prefix", default="be_")
    arg_hyp.add_argument("--train", action="store_true")
    arg_hyp.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # --- Seeding ---
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = th.device(args.device)
    print(f"Using device: {device}")

    # --- Data Loading and Preparation (from main_morl_BE.py) ---
    if args.DeepSeaTreasureConcave or args.DeepSeaTreasureSmooth:
        name_env = "dst_concave" if args.DeepSeaTreasureConcave else "smooth"
        trajectories_directory_path = f"trajectories/{name_env}/"
        num_policies = 10
        env_id = "deep-sea-treasure-v0" if args.DeepSeaTreasureConcave else "dst-smooth-v0"

        trajectories, true_labels, obj_feats_list = [], [], []
        for i in range(num_policies):
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
                    obj_feats_list.append(np.asarray(ret_vec, dtype=np.float64))
    else:
        raise ValueError("Please select a valid environment, e.g., --DeepSeaTreasureConcave")

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
    print("MAXIMUM LENGTH OF TRAJECTORIES:", max_len)
    
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
    num_actions = trajectories[0].acts.shape[1]

    encoder = BehaviorEncoderCLSattnSATyped(
        input_channels=input_coord_dims, cnn_output_dim=args.emb_dim, steps=max_len,
        nhead=args.nheads, d_hid=args.d_hid, emb_dim=args.emb_dim,
        num_actions=num_actions, nlayers=args.nlayers, dropout=args.dropout,
        input_coord_dims=input_coord_dims
    ).to(device)

    decoder = TrajectoryDecoder(args.emb_dim, input_coord_dims, num_actions, max_len).to(device)
    info_loss_fn = InstanceLoss(args.temperature, device=device)
    dim_loss_fn = DeepInfoMaxLoss(args.emb_dim).to(device)

    params = list(encoder.parameters()) + list(decoder.parameters()) + list(dim_loss_fn.parameters())
    optim = th.optim.AdamW(params, lr=args.lr)

    # --- Training or Loading ---
    model_name = f"{args.model_prefix}_{args.emb_dim}d_{args.nheads}h_e{args.epochs}.pt"
    model_path = os.path.join(args.model_dir, model_name)

    if args.train:
        print(f"Starting training for {args.epochs} epochs...")
        for epoch in range(args.epochs):
            loss = train_epoch(
                encoder, decoder, loader, optim, device, info_loss_fn, dim_loss_fn,
                args.recon_weight, args.info_weight, args.dim_weight, args.topo_weight
            )
            print(f"Epoch {epoch+1}/{args.epochs}: loss={loss:.4f}")
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

    print("Aggregating and visualizing policy embeddings...")
    image_dir = f"./images/{env_id}/"
    viz_path = os.path.join(image_dir, f"aggregated_embeddings_{args.emb_dim}d.png")
    policy_latents = aggregate_and_visualize_policy_embeddings(
        embeddings, policies, args.emb_dim, save_path=viz_path
    )

    json_path = os.path.join(trajectories_directory_path, f"policy_latents_{args.emb_dim}d.json")
    with open(json_path, "w") as fh:
        json.dump(policy_latents, fh, indent=2)
    print(f"Saved aggregated policy latents to {json_path}")

if __name__ == "__main__":
    main()