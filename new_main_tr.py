import argparse
import os
import random

import numpy as np #type: ignore[import]
import torch as th #type: ignore[import]
from tqdm import tqdm #type: ignore[import]
from sklearn.metrics import pairwise_distances #type: ignore[import]
from sklearn.manifold import trustworthiness #type: ignore[import]

from imitation.data.types import Trajectory #type: ignore[import]
from morl_behavior_objective.methods.behaviorencoder import (
    BehaviorEncoderCLSattnSATyped,
    DeepInfoMaxLoss,
    InstanceLoss,
    TrajectoryDecoder,
    PolicySetEncoder,
    VarianceCovarianceLoss
)
from morl_behavior_objective.methods.behaviorencoder_utils import (
    apply_per_traj,
    build_scaler,
    fit_on_flat,
    is_cont_actions,
    prepare_sa_trajectories,
    datasets_preparation_ret,
    train_epoch,
    train_epoch_no_norm,
    set_train_epoch,
    get_all_embeddings,
    get_mean_policy_embeddings,
    load_environment_data,
    generate_experiment_name,
    save_policy_latents_to_json,
    visualize_policy_embeddings_with_lines,
    visualize_trajectory_embeddings,
    visualize_policy_embeddings,
)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)




# ---------- Main Execution Block ----------
def main():
    parser = argparse.ArgumentParser()
    # Add all arguments from main_morl_BE.py for consistency
    arg_env = parser.add_argument_group('Environment Selection')
    arg_env.add_argument("-DSTC","--DeepSeaTreasureConcave", help="DeepSeaTreasureConcave environment",action="store_true") #no exp
    arg_env.add_argument("-DSTS","--DeepSeaTreasureSmooth", help="DeepSeaTreasureSmooth environment",action="store_true")
    arg_env.add_argument("-DSTLR","--DeepSeaTreasureLeftRight", help="DeepSeaTreasureLeftRight environment",action="store_true")
    arg_env.add_argument("-MHC","--MOHalfCheetah", help="MO-HalfCheetah environment",action="store_true")
    arg_env.add_argument("-MHW","--MOHighway", help="MO-Highway environment",action="store_true")
    arg_env.add_argument("-MHo2","--MOHopper2obj", help="MO-Hopper environment with 2 objectives",action="store_true")
    arg_env.add_argument("-MHo","--MOHopper", help="MO-Hopper environment",action="store_true")
    arg_env.add_argument("-RG", "--ResourceGathering", help="ResourceGathering environment", action="store_true") #no exp

    arg_viz = parser.add_argument_group('Visualizations and Directories')
    arg_viz.add_argument("--viz_policy_ids", type=int, nargs='+', default=None, help="List of policy IDs to label in the pre-aggregation visualization.")
    arg_viz.add_argument("--visualize", action="store_true", help="Whether to visualize trajectory embeddings before aggregation")
    arg_viz.add_argument("--model_dir", default="models/no_topo/")
    arg_viz.add_argument("--model_prefix", default="be")
    arg_viz.add_argument("--save_data", action="store_true", help="Save the aggregated policy embeddings to a JSON file")
    arg_viz.add_argument("--with_lines", action="store_true", help="Whether to connect policy embeddings with lines in the visualization")

    arg_hyp = parser.add_argument_group('Training Hyperparameters')
    arg_hyp.add_argument("--device", default="cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu")
    arg_hyp.add_argument("--epochs", type=int, default=500)
    arg_hyp.add_argument("--batch_size", type=int, default=32)
    arg_hyp.add_argument("--lr", type=float, default=3e-4)
    arg_hyp.add_argument("--emb_dim", type=int, default=3)
    arg_hyp.add_argument("--nheads", type=int, default=1)
    arg_hyp.add_argument("--nlayers", type=int, default=2)
    arg_hyp.add_argument("--d_hid", type=int, default=128)# dsts 64 dtslr 32
    arg_hyp.add_argument("--dropout", type=float, default=0.1)
    arg_hyp.add_argument("--recon_weight", type=float, default=0.1)
    arg_hyp.add_argument("--info_weight", type=float, default=1.0)
    arg_hyp.add_argument("--dim_weight", type=float, default=1.0)
    arg_hyp.add_argument("--segment_weight", type=float, default=0.0)
    arg_hyp.add_argument("--temperature", type=float, default=0.05)
    arg_hyp.add_argument("--state_scaler", default="quantile_normal")
    arg_hyp.add_argument("--action_scaler", default="quantile_normal")
    arg_hyp.add_argument("--scaler_fit", default="seen", choices=["seen", "both"])
    arg_hyp.add_argument("--train", action="store_true")
    arg_hyp.add_argument("--seed", type=int, default=0)
    
    # --- Hyperparams for the Set Encoder ---
    arg_set = parser.add_argument_group('Set Encoder Hyperparameters')
    arg_set.add_argument("--use_set_encoder", action="store_true", help="Whether to train a policy-level set encoder")
    arg_set.add_argument("--train_set", action="store_true", help="Whether to train the policy-level set encoder")
    arg_set.add_argument("--set_epochs", type=int, default=200, help="Number of epochs to train the policy-level set encoder")
    arg_set.add_argument("--set_lr", type=float, default=1e-3, help="Learning rate for the set encoder")
    arg_set.add_argument("--set_info_weight", type=float, default=1.0, help="InfoNCE loss weight for set encoder")
    arg_set.add_argument("--set_dim_weight", type=float, default=1.0, help="DIM loss weight for set encoder")
    arg_set.add_argument("--set_n_layers", type=int, default=2, help="Number of layers for set encoder transformer")
    arg_set.add_argument("--set_n_heads", type=int, default=4, help="Number of heads for set encoder transformer")
    arg_set.add_argument("--set_d_hid", type=int, default=1024, help="Hidden dimension for set encoder transformer")

    args = parser.parse_args()

    # --- Seeding ---
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = th.device(args.device)
    print(f"Using device: {device} | Seed: {args.seed}")
    
    # --- Bug Fix Init ---
    learned_policy_latents = {}
    
    env_code = "DSTC" if args.DeepSeaTreasureConcave else "DSTS" if args.DeepSeaTreasureSmooth else "DSTLR" if args.DeepSeaTreasureLeftRight else \
               "MHC" if args.MOHalfCheetah else "MHW" if args.MOHighway else "MHo2" if args.MOHopper2obj else "MHo" if args.MOHopper \
               else "RG" if args.ResourceGathering else "UNK"
    
    if env_code == "DSTC" or env_code == "DSTS" or env_code == "DSTLR":
        gaussian_m_state = 64
        gaussian_m_action = 32
        gaussian_sigma_state = 10 
        gaussian_sigma_action = 10
    else:
        gaussian_m_state = 128
        gaussian_m_action = 32
        gaussian_sigma_state = 5 
        gaussian_sigma_action = 5
    
    model_dir = args.model_dir+f"{env_code}/"
    args.model_dir = model_dir
    os.makedirs(args.model_dir, exist_ok=True)

    # --- Data Loading and Preparation (Refactored) ---
    trajectories, true_labels, obj_feats_per_traj, env_id, name_env, num_policies, embeddings_folder_path = load_environment_data(args)
    
    # Get one return vector *per policy*
    policy_returns = {}
    for label, ret in zip(true_labels, obj_feats_per_traj):
        if label not in policy_returns:
            policy_returns[label] = ret
    # Sort by key to get a stable list
    obj_feats_per_policy = [policy_returns[i] for i in sorted(policy_returns.keys())]

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

    full_dataset, _, _, _, loader, _, _, _ = datasets_preparation_ret(
        all_states, all_actions, all_masks, all_labels,
        train_size=len(all_states), val_size=0, test_size=0,
        loader_batch=args.batch_size, val_bptt=1, test_bptt=1, seed=args.seed,
        returns=obj_feats_per_traj # Use returns_per_traj
    )

    # --- Model, Losses, and Optimizer ---
    input_coord_dims = trajectories[-1].obs.shape[1]
    print("Input coordinate dimensions:", input_coord_dims)
    print("Input action dimensions:", trajectories[-1].acts.shape[1])
    num_actions = trajectories[0].acts.shape[1] if trajectories[0].acts.ndim > 1 else 1
    
    encoder = BehaviorEncoderCLSattnSATyped(
        input_channels=input_coord_dims, cnn_output_dim=args.emb_dim,steps=max_len, max_len=max_len,
        nhead=args.nheads, d_hid=args.d_hid, emb_dim=args.emb_dim,
        num_actions=num_actions, nlayers=args.nlayers, dropout=args.dropout,
        input_coord_dims=input_coord_dims, gaussian_m_state=gaussian_m_state, gaussian_m_action=gaussian_m_action,
        gaussian_sigma_state=gaussian_sigma_state, gaussian_sigma_action=gaussian_sigma_action,
    ).to(device)

    decoder = TrajectoryDecoder(args.emb_dim, input_coord_dims, num_actions, max_len).to(device) # Removed spec_norm
    info_loss_fn = InstanceLoss(args.temperature, device=device)
    dim_loss_fn = DeepInfoMaxLoss(args.emb_dim).to(device)
    vc_loss_fn = VarianceCovarianceLoss(std_coeff=25.0, cov_coeff=1.0).to(device)

    params = list(encoder.parameters()) + list(decoder.parameters()) + list(dim_loss_fn.parameters())
    optim = th.optim.AdamW(params, lr=args.lr)

    # --- Training or Loading ---
    base_name = generate_experiment_name(args, env_code)
    model_name = f"{base_name}_seed{args.seed}.pt"
    model_path = os.path.join(args.model_dir, model_name)
    
    print(encoder.model_type,"parameters ->",sum(p.numel() for p in encoder.parameters() if p.requires_grad)/1e6,"M")
    print(f"Model path: {model_path}")

    if args.train:
        print(f"Starting training for {args.epochs} epochs...")
        pbar = tqdm(range(args.epochs))
        for epoch in pbar:
            loss = train_epoch(
                encoder, decoder, loader, optim, device, info_loss_fn, dim_loss_fn, vc_loss_fn,
                args.recon_weight, args.info_weight, args.dim_weight,
                args.segment_weight, env_id
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

    norms = np.linalg.norm(embeddings, axis=1)
    avg_norm = norms.mean()
    min_norm = norms.min()
    max_norm = norms.max()
    print(f"\n--- NORMALIZATION CHECK ---")
    print(f"Average Norm: {avg_norm:.4f} (Should be ~1.0)")
    print(f"Min Norm:     {min_norm:.4f}")
    print(f"Max Norm:     {max_norm:.4f}")
    print(f"---------------------------\n")

    unique = np.unique(policies)
    # per-policy mean and adjacent distances
    policy_means = np.array([embeddings[policies==pid].mean(0) for pid in unique])
    adj_dists = np.linalg.norm(policy_means[1:] - policy_means[:-1], axis=1)
    print("Adjacent mean embedding distances:", adj_dists)

    if args.visualize:
        print("Visualizing raw trajectory embeddings (pre-aggregation)...")
        visualize_trajectory_embeddings(embeddings, policies, args.emb_dim)
        if args.viz_policy_ids is not None:
            print(f"Visualizing raw trajectory embeddings (pre-aggregation) highlighting policy {args.viz_policy_ids}...")
            visualize_trajectory_embeddings(embeddings, policies, args.emb_dim, highlight_pids=args.viz_policy_ids)

    # --- Setup paths for outputs ---
    image_dir = f"./images/no_topo/{env_id}/"
    os.makedirs(image_dir, exist_ok=True)
    
    # --- Mean Aggregation & Visualization ---
    print("Aggregating (by MEAN) and visualizing policy embeddings...")
    mean_policy_latents = get_mean_policy_embeddings(embeddings, policies)
    
    viz_path_mean = os.path.join(image_dir, f"mean_{base_name}_seed{args.seed}.png")
    if not args.with_lines:
        visualize_policy_embeddings(
            mean_policy_latents,
            args.emb_dim,
            title="Aggregated Policy Embeddings (Mean)",
            save_path=viz_path_mean if args.save_data else None
        )
    else:
        visualize_policy_embeddings_with_lines(
            mean_policy_latents,
            args.emb_dim,
            title="Aggregated Policy Embeddings (Mean) with Lines",
            save_path=viz_path_mean if args.save_data else None
        )


    # -----------------------------------------------------------------
    # --- Train Policy-Level Set Encoder (Self-Supervised) ---
    # -----------------------------------------------------------------
    if args.use_set_encoder:
        print("\n" + "-"*30)
        print("--- Initializing Policy-Level Set Encoder ---")
        
        # 1. Group data
        policy_data = {}
        unique_pids = np.unique(policies)

        for pid in unique_pids:
            p_embeddings = embeddings[policies == pid]
            policy_data[pid] = {
                "embeddings": th.tensor(p_embeddings, dtype=th.float32).to(device)
            }
        
        # 2. Define models
        set_encoder = PolicySetEncoder(
            emb_dim=args.emb_dim, 
            hidden_dim=args.set_d_hid,
            n_layers=args.set_n_layers, 
            n_heads=max(1, args.set_n_heads) # ensure n_heads > 0
        ).to(device)
        
        # Define NEW loss functions for this model
        info_loss_fn_set = InstanceLoss(args.temperature, device=device)
        dim_loss_fn_set = DeepInfoMaxLoss(args.emb_dim).to(device)
        
        set_params = list(set_encoder.parameters()) + list(dim_loss_fn_set.parameters())
        set_optim = th.optim.AdamW(set_params, lr=args.set_lr) 
        
        # --- NEW: Set Encoder Pathing ---
        set_base_name = generate_experiment_name(args, env_code).replace(
            args.model_prefix, f"{args.model_prefix}_set"
        )
        set_model_name = f"{set_base_name}_seed{args.seed}.pt"
        set_model_path = os.path.join(args.model_dir, set_model_name)
        print(f"Set Encoder Model path: {set_model_path}")

        # --- Set Encoder Training/Loading ---
        if args.train_set:
            print(f"Starting set encoder training for {args.set_epochs} epochs...")
            pbar = tqdm(range(args.set_epochs), desc="Training SetEncoder")
            
            for epoch in pbar:
                total_loss, dim_loss, info_loss = set_train_epoch(
                    set_encoder, policy_data, set_optim,
                    info_loss_fn_set, dim_loss_fn_set,
                    args.set_info_weight, args.set_dim_weight
                )
                
                if epoch % 20 == 0 or epoch == args.set_epochs - 1:
                    pbar.set_description(f"SetEncoder Ep {epoch} | Total Loss: {total_loss:.4f} | DIM: {dim_loss:.4f} | InfoNCE: {info_loss:.4f}")

            print("--- Policy-Level Set Encoder Training Complete ---")
            
            # Save the checkpoint
            os.makedirs(args.model_dir, exist_ok=True)
            th.save({
                "set_encoder": set_encoder.state_dict(), 
                "dim_disc_set": dim_loss_fn_set.state_dict()
            }, set_model_path)
            print(f"Saved set encoder checkpoint to {set_model_path}")

        else:
            # Load the checkpoint
            print("Loading pre-trained set encoder...")
            if not os.path.exists(set_model_path):
                raise FileNotFoundError(f"Set encoder checkpoint not found at {set_model_path}. Please train first with --train_set.")
            
            ckpt_set = th.load(set_model_path, map_location=device)
            set_encoder.load_state_dict(ckpt_set["set_encoder"])
            dim_loss_fn_set.load_state_dict(ckpt_set["dim_disc_set"])
            print(f"Loaded set encoder checkpoint from {set_model_path}")

        # 4. Get all *learned* embeddings
        set_encoder.eval()
        # learned_policy_latents is already initialized at the top
        with th.no_grad():
            for pid in unique_pids:
                if pid not in policy_data or policy_data[pid]["embeddings"].shape[0] < 1:
                    continue
                emb_set = policy_data[pid]["embeddings"]
                # Get the CLS token (first element of tuple)
                learned_emb = set_encoder(emb_set)[0].squeeze().cpu().numpy()
                learned_policy_latents[int(pid)] = learned_emb
        
        # --- Visualize Learned Embeddings ---
        print("Visualizing LEARNED policy embeddings...")
        viz_path_learned = os.path.join(image_dir, f"learned_{base_name}_seed{args.seed}.png")
        visualize_policy_embeddings(
            learned_policy_latents,
            args.emb_dim,
            title="Aggregated Policy Embeddings (Learned)",
            save_path=viz_path_learned if args.save_data else None
        )


    # -----------------------------------------------------------------
    # --- Save both aggregated embedding files ---
    # -----------------------------------------------------------------
    
    if args.save_data:
        # Save the MEAN embeddings
        json_mean_path = os.path.join(
            embeddings_folder_path, f"mean_{base_name}_seed{args.seed}.json"
        )
        save_policy_latents_to_json(
            mean_policy_latents, 
            trajectories, 
            true_labels, 
            obj_feats_per_traj, 
            json_mean_path
        )
        
        # Save the LEARNED embeddings (if they were computed)
        if learned_policy_latents: # Check if dictionary is not empty
            json_learned_path = os.path.join(
                embeddings_folder_path, f"learned_{base_name}_seed{args.seed}.json"
            )
            save_policy_latents_to_json(
                learned_policy_latents, 
                trajectories, 
                true_labels, 
                obj_feats_per_traj, 
                json_learned_path
            )

    # -----------------------------------------------------------------
    # --- Compare Mean vs. Learned Embeddings ---
    # -----------------------------------------------------------------
    if args.use_set_encoder:
        print("\n" + "-"*30)
        print("--- Comparing Mean vs. Learned Embedding Spaces ---")

        # 1. Get embedding matrices in the same policy order
        pids_ordered = sorted(learned_policy_latents.keys())
        
        if not pids_ordered:
            print("No learned policy latents found. Skipping comparison.")
            return

        mean_emb_matrix = np.array([mean_policy_latents[pid] for pid in pids_ordered])
        learned_emb_matrix = np.array([learned_policy_latents[pid] for pid in pids_ordered])
        
        if len(pids_ordered) < 2:
            print("Not enough policies to compare embedding spaces. Skipping.")
            return

        # 2. Compute pairwise distance matrices
        D_mean = pairwise_distances(mean_emb_matrix, metric='euclidean')
        D_learned = pairwise_distances(learned_emb_matrix, metric='euclidean')

        # 3. Calculate Trustworthiness & Continuity
        # k_neighbors should be small, e.g., 5% of N, but at least 2
        n_policies = len(pids_ordered)
        k_neighbors = max(2, min(7, n_policies // 5)) # e.g., k=7
        
        if n_policies <= k_neighbors:
            print(f"Warning: k_neighbors={k_neighbors} is >= num_policies={n_policies}. Skipping comparison.")
        else:
            # T: How many of k-neighbors from MEAN space are preserved in LEARNED space
            trust = trustworthiness(
                D_mean, D_learned, n_neighbors=k_neighbors, metric='precomputed'
            )
            
            # C: How many of k-neighbors from LEARNED space are preserved in MEAN space
            cont = trustworthiness(
                D_learned, D_mean, n_neighbors=k_neighbors, metric='precomputed'
            )
            
            print(f"Comparison with k={k_neighbors} neighbors:")
            print(f"  Trustworthiness (Mean -> Learned): {trust:.4f}")
            print(f"  Continuity    (Learned -> Mean): {cont:.4f}")
        
        print("--- Comparison Complete ---")


if __name__ == "__main__":
    main()