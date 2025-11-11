import argparse
import json
import os
import pickle
import random
import warnings

from imitation.data.types import Trajectory  # To structure trajectory data #type: ignore[import]
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch as th
import umap  # type: ignore[import]

warnings.filterwarnings("ignore")


from morl_behavior_objective.methods.behaviorencoder import *
from morl_behavior_objective.methods.behaviorencoder_utils import *

## PARSER


def setup_parser():
    parser = argparse.ArgumentParser(description="Model-Free Multi Intention Maximum Likelihood IRL: Experiments Runner")

    arg_env = parser.add_argument_group("Environment Selection")
    arg_env.add_argument(
        "-T2D", "--Traj2d", help="Apply the selected algorithm to the Traj2d environment", action="store_true"
    )
    arg_env.add_argument(
        "-Rv4", "--Reacherv4", help="Apply the selected algorithm to the Reacher-v4 environment", action="store_true"
    )
    arg_env.add_argument(
        "-Pv4", "--Pusherv4", help="Apply the selected algorithm to the Pusher-v4 environment", action="store_true"
    )
    arg_env.add_argument(
        "-DST",
        "--DeepSeaTreasure",
        help="Apply the selected algorithm to the DeepSeaTreasure environment",
        action="store_true",
    )
    arg_env.add_argument(
        "-DSTC",
        "--DeepSeaTreasureConcave",
        help="Apply the selected algorithm to the DeepSeaTreasureConcave environment",
        action="store_true",
    )
    arg_env.add_argument(
        "-DSTS",
        "--DeepSeaTreasureSmooth",
        help="Apply the selected algorithm to the DeepSeaTreasureSmooth environment",
        action="store_true",
    )

    arg_alg = parser.add_argument_group("Unseen Split Selection")
    arg_alg.add_argument(
        "-split",
        "--use_seen_unseen_split",
        action="store_true",
        help="Use seen-unseen split for training. If not set, train on the entire dataset.",
    )
    arg_alg.add_argument(
        "-nUnModes",
        "--num_unseen_modes",
        type=int,
        default=1,
        help="Number of unseen modes in the unseen split for the dataset.",
    )

    arg_hyp = parser.add_argument_group("Settings")
    arg_hyp.add_argument("-nT", "--num_trajs", type=int, default=100, help="int: Number of expert trajectories to generate")
    arg_hyp.add_argument("-seed", "--seed", type=int, default=0, help="int: Random seed for reproducibility")
    arg_hyp.add_argument(
        "--ratio",
        type=int,
        default=1,
        help="Ratio for splitting trajectories between modes. 1 uniform 3 first gets most last gets least, etc.",
    )
    arg_hyp.add_argument(
        "--embedding_strategy",
        type=str,
        default="cls_only",
        choices=["cls_only", "hybrid", "goal_oriented"],
        help="Strategy for creating the final trajectory embedding for clustering.",
    )
    arg_hyp.add_argument("--diagnostics", action="store_true", help="Run diagnostic checks on the trajectory embeddings.")

    arg_vis = parser.add_argument_group("Visualization")
    arg_vis.add_argument(
        "-vS",
        "--visualize_scalers",
        help="Visualize the effect of different scalers on the state space",
        action="store_true",
    )
    arg_vis.add_argument("-vO", "--visualize_original", help="Visualize the original state space", action="store_true")
    arg_vis.add_argument("-vC", "--visualize_clusters", help="Visualize the clusters", action="store_true")
    arg_vis.add_argument("-r", "--render", help="Render the environment", action="store_true", default=False)
    return parser


## MAIN


def main():
    parser = setup_parser()
    args = parser.parse_args()
    if args.Traj2d or args.Reacherv4 or args.Pusherv4:
        unseen_modes = args.num_unseen_modes
        K = 6 - unseen_modes if args.use_seen_unseen_split else 6
        K_known = 6
    elif args.DeepSeaTreasure:
        K = 6
    elif args.DeepSeaTreasureConcave or args.DeepSeaTreasureSmooth:
        K = 10
    else:
        raise ValueError(
            "No available environment selected. Please select either --Two Lakes Fishing (-F), --Traj2d(-T2D), --Reacher-v4(-Rv4) or --Pusher-v4(-Pv4)."
        )

    unseen_modes = args.num_unseen_modes
    visualize_original = args.visualize_original
    visualize_scalers = args.visualize_scalers
    normalize_input = True
    saving = False
    num_trajs = args.num_trajs

    diagnostics = args.diagnostics

    embedding_strategy_code = (
        "CLS"
        if args.embedding_strategy == "cls_only"
        else "HYB"
        if args.embedding_strategy == "hybrid"
        else "GO"
        if args.embedding_strategy == "goal_oriented"
        else "UNK"
    )
    training_code = "SPLIT" if args.use_seen_unseen_split else "FULL"

    env_name = (
        "Traj2d"
        if args.Traj2d
        else "Reacher-v4"
        if args.Reacherv4
        else "Pusher-v4"
        if args.Pusherv4
        else "DeepSeaTreasure"
        if args.DeepSeaTreasure
        else "DeepSeaTreasureConcave"
        if args.DeepSeaTreasureConcave
        else "DeepSeaTreasureSmooth"
        if args.DeepSeaTreasureSmooth
        else "Unknown"
    )
    env_id = "Traj2d" if args.Traj2d else "Reacher-v4" if args.Reacherv4 else "Pusher-v4" if args.Pusherv4 else "UnknownEnv"
    env_code = (
        "T2D"
        if args.Traj2d
        else "Rv4"
        if args.Reacherv4
        else "Pv4"
        if args.Pusherv4
        else "DST"
        if args.DeepSeaTreasure
        else "DSTC"
        if args.DeepSeaTreasureConcave
        else "DSTS"
        if args.DeepSeaTreasureSmooth
        else "UNK"
    )

    print(f"*** CoMIIRL approach on {env_name} ***")

    # Set the random seed for reproducibility
    SEEDS = [0, 1, 2, 3, 4]
    SEED = args.seed
    ratio = args.ratio
    th.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # Set the device to GPU if available
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    HIGH_CONTRAST_PREDICTED_COLORS = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d2",
        "#c7c7c7",
        "#dbdb8d",
        "#9edae5",
    ]

    trajectory_manager = {}
    # Initialize optional objective features container used by diagnostics
    obj_feats_from_json = None
    single_trajectories_manager = {}

    if args.Traj2d or args.Reacherv4 or args.Pusherv4:
        trajectories_directory_path = "morl_behavior_objective/old_expert_trajectories/"
        name_env = (
            "2D-Trajectory"
            if args.Traj2d
            else "Reacher-v4"
            if args.Reacherv4
            else "Pusher-v4"
            if args.Pusherv4
            else "Unknown"
        )
        file_path = f"{trajectories_directory_path}expert_imitation_trajectories_{name_env}_mode_0.pkl"
        file_path_withrew = file_path.replace(".pkl", "_withrew.pkl")
        with open(file_path, "rb") as f:
            demos_0 = pickle.load(f)
            labels_0 = np.array([10] * len(demos_0))
            print(f"Loaded {len(demos_0)} expert trajectories for mode 0")
        with open(file_path_withrew, "rb") as f:
            demos_0_withrew = pickle.load(f)
        file_path = f"{trajectories_directory_path}expert_imitation_trajectories_{name_env}_mode_1.pkl"
        file_path_withrew = file_path.replace(".pkl", "_withrew.pkl")
        with open(file_path, "rb") as f:
            demos_1 = pickle.load(f)
            labels_1 = np.array([11] * len(demos_1))
            print(f"Loaded {len(demos_1)} expert trajectories for mode 1")
        with open(file_path_withrew, "rb") as f:
            demos_1_withrew = pickle.load(f)
        file_path = f"{trajectories_directory_path}expert_imitation_trajectories_{name_env}_mode_2.pkl"
        file_path_withrew = file_path.replace(".pkl", "_withrew.pkl")
        with open(file_path, "rb") as f:
            demos_2 = pickle.load(f)
            labels_2 = np.array([12] * len(demos_2))
            print(f"Loaded {len(demos_2)} expert trajectories for mode 2")
        with open(file_path_withrew, "rb") as f:
            demos_2_withrew = pickle.load(f)
        file_path = f"{trajectories_directory_path}expert_imitation_trajectories_{name_env}_mode_3.pkl"
        file_path_withrew = file_path.replace(".pkl", "_withrew.pkl")
        with open(file_path, "rb") as f:
            demos_3 = pickle.load(f)
            labels_3 = np.array([13] * len(demos_3))
            print(f"Loaded {len(demos_3)} expert trajectories for mode 3")
        with open(file_path_withrew, "rb") as f:
            demos_3_withrew = pickle.load(f)
        file_path = f"{trajectories_directory_path}expert_imitation_trajectories_{name_env}_mode_4.pkl"
        file_path_withrew = file_path.replace(".pkl", "_withrew.pkl")
        with open(file_path, "rb") as f:
            demos_4 = pickle.load(f)
            labels_4 = np.array([14] * len(demos_4))
            print(f"Loaded {len(demos_4)} expert trajectories for mode 4")
        with open(file_path_withrew, "rb") as f:
            demos_4_withrew = pickle.load(f)
        file_path = f"{trajectories_directory_path}expert_imitation_trajectories_{name_env}_mode_5.pkl"
        file_path_withrew = file_path.replace(".pkl", "_withrew.pkl")
        with open(file_path, "rb") as f:
            demos_5 = pickle.load(f)
            labels_5 = np.array([15] * len(demos_5))
            print(f"Loaded {len(demos_5)} expert trajectories for mode 5")
        with open(file_path_withrew, "rb") as f:
            demos_5_withrew = pickle.load(f)
        K_to_split = K + unseen_modes if args.use_seen_unseen_split else K
        counts = compute_mode_counts(num_trajs, K_to_split, ratio)
        print(f"Splitting {num_trajs} trajectories into {K_to_split} modes with ratio {ratio}: {counts}")
        if args.use_seen_unseen_split:
            if args.Traj2d:
                print(f"Processing Traj2d trajectories with {unseen_modes} unseen modes. . . ")
                if unseen_modes == 1:
                    trajectories = (
                        demos_0[: counts[0]]
                        + demos_1[: counts[1]]
                        + demos_2[: counts[2]]
                        + demos_3[: counts[3]]
                        + demos_4[: counts[4]]
                    )
                    trajectories_withrew = (
                        demos_0_withrew[: counts[0]]
                        + demos_1_withrew[: counts[1]]
                        + demos_2_withrew[: counts[2]]
                        + demos_3_withrew[: counts[3]]
                        + demos_4_withrew[: counts[4]]
                    )
                    true_labels = np.concatenate(
                        (
                            labels_0[: counts[0]],
                            labels_1[: counts[1]],
                            labels_2[: counts[2]],
                            labels_3[: counts[3]],
                            labels_4[: counts[4]],
                        )
                    )
                elif unseen_modes == 2:
                    trajectories = demos_0[: counts[0]] + demos_1[: counts[1]] + demos_3[: counts[3]] + demos_4[: counts[4]]
                    trajectories_withrew = (
                        demos_0_withrew[: counts[0]]
                        + demos_1_withrew[: counts[1]]
                        + demos_3_withrew[: counts[3]]
                        + demos_4_withrew[: counts[4]]
                    )
                    true_labels = np.concatenate(
                        (labels_0[: counts[0]], labels_1[: counts[1]], labels_3[: counts[3]], labels_4[: counts[4]])
                    )
                elif unseen_modes == 3:
                    trajectories = demos_0[: counts[0]] + demos_3[: counts[3]] + demos_4[: counts[4]]
                    trajectories_withrew = (
                        demos_0_withrew[: counts[0]] + demos_3_withrew[: counts[3]] + demos_4_withrew[: counts[4]]
                    )
                    true_labels = np.concatenate(
                        (labels_0[: counts[0]], labels_3[: counts[3]], labels_4[: counts[4]])
                    )  # , labels_3[:counts[3]]))
                else:
                    raise ValueError("For Traj2d, when using unseen split, unseen_modes must be 1, 2, or 3.")
                unseen_trajectories_for_online = (
                    demos_0[counts[0] : counts[0] + counts[0]]
                    + demos_1[counts[1] : counts[1] + counts[1]]
                    + demos_2[counts[2] : counts[2] + counts[2]]
                    + demos_3[counts[3] : counts[3] + counts[3]]
                    + demos_4[counts[4] : counts[4] + counts[4]]
                    + demos_5[counts[5] : counts[5] + counts[5]]
                )
                trajectories_withrew_for_online = (
                    demos_0_withrew[counts[0] : counts[0] + counts[0]]
                    + demos_1_withrew[counts[1] : counts[1] + counts[1]]
                    + demos_2_withrew[counts[2] : counts[2] + counts[2]]
                    + demos_3_withrew[counts[3] : counts[3] + counts[3]]
                    + demos_4_withrew[counts[4] : counts[4] + counts[4]]
                    + demos_5_withrew[counts[5] : counts[5] + counts[5]]
                )
                true_labels_online = np.concatenate(
                    (
                        labels_0[counts[0] : counts[0] + counts[0]],
                        labels_1[counts[1] : counts[1] + counts[1]],
                        labels_2[counts[2] : counts[2] + counts[2]],
                        labels_3[counts[3] : counts[3] + counts[3]],
                        labels_4[counts[4] : counts[4] + counts[4]],
                        labels_5[counts[5] : counts[5] + counts[5]],
                    )
                )
            elif args.Reacherv4:
                print(f"Processing Reacher-v4 environment with {unseen_modes} unseen modes...")
                if unseen_modes == 1:
                    trajectories = (
                        demos_0[: counts[0]]
                        + demos_1[: counts[1]]
                        + demos_2[: counts[2]]
                        + demos_3[: counts[3]]
                        + demos_4[: counts[4]]
                    )
                    trajectories_withrew = (
                        demos_0_withrew[: counts[0]]
                        + demos_1_withrew[: counts[1]]
                        + demos_2_withrew[: counts[2]]
                        + demos_3_withrew[: counts[3]]
                        + demos_4_withrew[: counts[4]]
                    )
                    true_labels = np.concatenate(
                        (
                            labels_0[: counts[0]],
                            labels_1[: counts[1]],
                            labels_2[: counts[2]],
                            labels_3[: counts[3]],
                            labels_4[: counts[4]],
                        )
                    )
                elif unseen_modes == 2:
                    trajectories = demos_0[: counts[0]] + demos_1[: counts[1]] + demos_3[: counts[3]] + demos_4[: counts[4]]
                    trajectories_withrew = (
                        demos_0_withrew[: counts[0]]
                        + demos_1_withrew[: counts[1]]
                        + demos_3_withrew[: counts[3]]
                        + demos_4_withrew[: counts[4]]
                    )
                    true_labels = np.concatenate(
                        (labels_0[: counts[0]], labels_1[: counts[1]], labels_3[: counts[3]], labels_4[: counts[4]])
                    )
                elif unseen_modes == 3:
                    trajectories = demos_0[: counts[0]] + demos_2[: counts[2]] + demos_4[: counts[4]]
                    trajectories_withrew = (
                        demos_0_withrew[: counts[0]] + demos_2_withrew[: counts[2]] + demos_4_withrew[: counts[4]]
                    )
                    true_labels = np.concatenate(
                        (labels_0[: counts[0]], labels_2[: counts[2]], labels_4[: counts[4]])
                    )  # , labels_3[:counts[3]]))
                else:
                    raise ValueError("For Reacher-v4, when using unseen split, unseen_modes must be 1, 2, or 3.")
                unseen_trajectories_for_online = (
                    demos_0[counts[0] : counts[0] + counts[0]]
                    + demos_1[counts[1] : counts[1] + counts[1]]
                    + demos_2[counts[2] : counts[2] + counts[2]]
                    + demos_3[counts[3] : counts[3] + counts[3]]
                    + demos_4[counts[4] : counts[4] + counts[4]]
                    + demos_5[counts[5] : counts[5] + counts[5]]
                )
                trajectories_withrew_for_online = (
                    demos_0_withrew[counts[0] : counts[0] + counts[0]]
                    + demos_1_withrew[counts[1] : counts[1] + counts[1]]
                    + demos_2_withrew[counts[2] : counts[2] + counts[2]]
                    + demos_3_withrew[counts[3] : counts[3] + counts[3]]
                    + demos_4_withrew[counts[4] : counts[4] + counts[4]]
                    + demos_5_withrew[counts[5] : counts[5] + counts[5]]
                )
                true_labels_online = np.concatenate(
                    (
                        labels_0[counts[0] : counts[0] + counts[0]],
                        labels_1[counts[1] : counts[1] + counts[1]],
                        labels_2[counts[2] : counts[2] + counts[2]],
                        labels_3[counts[3] : counts[3] + counts[3]],
                        labels_4[counts[4] : counts[4] + counts[4]],
                        labels_5[counts[5] : counts[5] + counts[5]],
                    )
                )
            elif args.Pusherv4:
                print(f"Processing Pusher-v4 environment with {unseen_modes} unseen modes...")
                if unseen_modes == 1:
                    trajectories = (
                        demos_0[: counts[0]]
                        + demos_1[: counts[1]]
                        + demos_2[: counts[2]]
                        + demos_3[: counts[3]]
                        + demos_4[: counts[4]]
                    )
                    trajectories_withrew = (
                        demos_0_withrew[: counts[0]]
                        + demos_1_withrew[: counts[1]]
                        + demos_2_withrew[: counts[2]]
                        + demos_3_withrew[: counts[3]]
                        + demos_4_withrew[: counts[4]]
                    )
                    true_labels = np.concatenate(
                        (
                            labels_0[: counts[0]],
                            labels_1[: counts[1]],
                            labels_2[: counts[2]],
                            labels_3[: counts[3]],
                            labels_4[: counts[4]],
                        )
                    )
                elif unseen_modes == 2:
                    trajectories = demos_0[: counts[0]] + demos_1[: counts[1]] + demos_3[: counts[3]] + demos_4[: counts[4]]
                    trajectories_withrew = (
                        demos_0_withrew[: counts[0]]
                        + demos_1_withrew[: counts[1]]
                        + demos_3_withrew[: counts[3]]
                        + demos_4_withrew[: counts[4]]
                    )
                    true_labels = np.concatenate(
                        (labels_0[: counts[0]], labels_1[: counts[1]], labels_3[: counts[3]], labels_4[: counts[4]])
                    )
                elif unseen_modes == 3:
                    trajectories = demos_0[: counts[0]] + demos_2[: counts[2]] + demos_4[: counts[4]]
                    trajectories_withrew = (
                        demos_0_withrew[: counts[0]] + demos_2_withrew[: counts[2]] + demos_4_withrew[: counts[4]]
                    )
                    true_labels = np.concatenate(
                        (labels_0[: counts[0]], labels_2[: counts[2]], labels_4[: counts[4]])
                    )  # , labels_3[:counts[3]]))
                else:
                    raise ValueError("For Pusher-v4, when using unseen split, unseen_modes must be 1, 2, or 3.")

                unseen_trajectories_for_online = (
                    demos_0[counts[0] : counts[0] + counts[0]]
                    + demos_1[counts[1] : counts[1] + counts[1]]
                    + demos_2[counts[2] : counts[2] + counts[2]]
                    + demos_3[counts[3] : counts[3] + counts[3]]
                    + demos_4[counts[4] : counts[4] + counts[4]]
                    + demos_5[counts[5] : counts[5] + counts[5]]
                )
                trajectories_withrew_for_online = (
                    demos_0_withrew[counts[0] : counts[0] + counts[0]]
                    + demos_1_withrew[counts[1] : counts[1] + counts[1]]
                    + demos_2_withrew[counts[2] : counts[2] + counts[2]]
                    + demos_3_withrew[counts[3] : counts[3] + counts[3]]
                    + demos_4_withrew[counts[4] : counts[4] + counts[4]]
                    + demos_5_withrew[counts[5] : counts[5] + counts[5]]
                )
                true_labels_online = np.concatenate(
                    (
                        labels_0[counts[0] : counts[0] + counts[0]],
                        labels_1[counts[1] : counts[1] + counts[1]],
                        labels_2[counts[2] : counts[2] + counts[2]],
                        labels_3[counts[3] : counts[3] + counts[3]],
                        labels_4[counts[4] : counts[4] + counts[4]],
                        labels_5[counts[5] : counts[5] + counts[5]],
                    )
                )
            else:
                raise ValueError("Unsupported environment. Please select either Traj2d, Reacher-v4, or Pusher-v4.")
        else:
            trajectories = (
                demos_0[: counts[0]]
                + demos_1[: counts[1]]
                + demos_2[: counts[2]]
                + demos_3[: counts[3]]
                + demos_4[: counts[4]]
                + demos_5[: counts[5]]
            )
            trajectories_withrew = (
                demos_0_withrew[: counts[0]]
                + demos_1_withrew[: counts[1]]
                + demos_2_withrew[: counts[2]]
                + demos_3_withrew[: counts[3]]
                + demos_4_withrew[: counts[4]]
                + demos_5_withrew[: counts[5]]
            )
            true_labels = np.concatenate(
                (
                    labels_0[: counts[0]],
                    labels_1[: counts[1]],
                    labels_2[: counts[2]],
                    labels_3[: counts[3]],
                    labels_4[: counts[4]],
                    labels_5[: counts[5]],
                )
            )

            unseen_trajectories_for_online = (
                demos_0[counts[0] : counts[0] + counts[0]]
                + demos_1[counts[1] : counts[1] + counts[1]]
                + demos_2[counts[2] : counts[2] + counts[2]]
                + demos_3[counts[3] : counts[3] + counts[3]]
                + demos_4[counts[4] : counts[4] + counts[4]]
                + demos_5[counts[5] : counts[5] + counts[5]]
            )
            trajectories_withrew_for_online = (
                demos_0_withrew[counts[0] : counts[0] + counts[0]]
                + demos_1_withrew[counts[1] : counts[1] + counts[1]]
                + demos_2_withrew[counts[2] : counts[2] + counts[2]]
                + demos_3_withrew[counts[3] : counts[3] + counts[3]]
                + demos_4_withrew[counts[4] : counts[4] + counts[4]]
                + demos_5_withrew[counts[5] : counts[5] + counts[5]]
            )
            true_labels_online = np.concatenate(
                (
                    labels_0[counts[0] : counts[0] + counts[0]],
                    labels_1[counts[1] : counts[1] + counts[1]],
                    labels_2[counts[2] : counts[2] + counts[2]],
                    labels_3[counts[3] : counts[3] + counts[3]],
                    labels_4[counts[4] : counts[4] + counts[4]],
                    labels_5[counts[5] : counts[5] + counts[5]],
                )
            )

        print(f"\n--- Calculating Original Expert Reward Statistics for {name_env} ---")
        if args.Reacherv4 or args.Pusherv4:
            mean_expert_reward, std_expert_reward = calculate_original_expert_reward_stats(trajectories_withrew)
        # elif args.Traj2d:
        #     import my_envs.traj2d_gymnasium as traj2d_mod
        #     env = traj2d_mod.Traj(mode_idx=0)
        #     mean_expert_reward, std_expert_reward = calculate_expert_reward(demos_0[:counts[0]], env, mode_idx=0, env_name=env_name)

        print(f"Original Expert Reward (from stored .rews): Mean={mean_expert_reward:.4f} ± Std={std_expert_reward:.4f}")

        # true_labels = np.concatenate((labels_0[:counts[0]], labels_1[:counts[1]], labels_2[:counts[2]]))#, labels_3[:counts[3]]))
        # true_labels = np.concatenate((labels_0[:counts[0]], labels_3[:counts[3]], labels_4[:counts[4]]))#, labels_3[:counts[3]]))
        # true_labels_online = np.concatenate((labels_3[:counts[3]], labels_4[:counts[4]], labels_5[:counts[5]])) if args.Reacherv4 else np.concatenate((labels_2[:counts[2]],labels_3[:counts[3]],))
        # true_labels_online = np.concatenate((labels_1[:counts[1]], labels_2[:counts[2]], labels_3[:counts[3]], labels_5[:counts[5]]))
        # true_labels = np.concatenate((true_labels, labels_4[:counts[4]], labels_5[:counts[5]])) if args.Reacherv4 else true_labels
        input_coord_dims = demos_0[0].obs[0].shape[0]
        env_num_step = 1  # unused in SA case
        num_actions = demos_0[0].acts[0].shape[0]
        num_trajs = len(trajectories)
        print(f"Generated {len(trajectories)} expert trajectories for {name_env} with {K} modes.")

    elif args.DeepSeaTreasure:
        name_env = "left_right_dst"
        trajectories_directory_path = f"trajectories/{name_env}/"
        num_policies = 6
        env_id = "left-right-dst-v0"

        trajectories = []
        true_labels = []
        obj_feats_list = []
        single_trajectories = []
        true_labels_single = []
        for i in range(num_policies):
            single_trajectory = None
            file_path = os.path.join(trajectories_directory_path, f"{name_env}_{i}.json")
            with open(file_path) as f:
                data = json.load(f)
            ret_vec = data.get("return", None)
            for states, actions in data["trajectories"]:
                # The trajectories from JSON are lists of lists, convert to numpy arrays
                obs_list = list(states)
                obs_list.append(states[-1])  # Repeat the last state
                obs = np.array(obs_list, dtype=np.float32)
                acts = np.array(actions, dtype=np.float32)

                # Create a Trajectory object. infos and terminal are set to defaults.
                traj = Trajectory(obs=obs, acts=acts, infos=None, terminal=True)
                if single_trajectory is None:
                    single_trajectory = traj
                trajectories.append(traj)
                true_labels.append(i)
                if ret_vec is not None:
                    obj_feats_list.append(np.asarray(ret_vec, dtype=np.float64))
                else:
                    obj_feats_list.append(None)
            single_trajectories_manager[i] = {"trajectory": single_trajectory, "return": ret_vec}
            single_trajectories.append(single_trajectory)
            true_labels_single.append(i)

        true_labels = np.array(true_labels)
        true_labels_single = np.unique(np.array(true_labels_single))
        if len(obj_feats_list) == len(trajectories) and all(x is not None for x in obj_feats_list):
            obj_feats_from_json = np.vstack(obj_feats_list).astype(np.float64)
        else:
            obj_feats_from_json = None
        # It will pad trajectories since they have different lengths and prepare them for the model.
        all_states, all_actions, all_masks, all_labels, max_len = prepare_sa_trajectories(env_id, trajectories, true_labels)
        all_states_single, all_actions_single, all_masks_single, all_labels_single, max_len_single = (
            prepare_sa_trajectories(env_id, single_trajectories, true_labels_single)
        )

        input_coord_dims = trajectories[0].obs.shape[1] if len(trajectories) > 0 and trajectories[0].obs.ndim > 1 else 1
        num_actions = trajectories[0].acts.shape[1] if len(trajectories) > 0 and trajectories[0].acts.ndim > 1 else 1
        num_trajs = len(trajectories)
        K = len(np.unique(true_labels))
        print(f"Loaded {num_trajs} expert trajectories for {name_env} with {K} modes.")

    elif args.DeepSeaTreasureConcave or args.DeepSeaTreasureSmooth:
        name_env = "dst_concave" if args.DeepSeaTreasureConcave else "smooth"
        trajectories_directory_path = f"trajectories/{name_env}/"
        num_policies = K
        env_id = "deep-sea-treasure-v0" if args.DeepSeaTreasureConcave else "dst-smooth-v0"

        trajectories = []
        true_labels = []
        obj_feats_list = []
        single_trajectories = []
        true_labels_single = []
        for i in range(num_policies):
            single_trajectory = None
            file_path = os.path.join(
                trajectories_directory_path, f"dst_{i}.json" if args.DeepSeaTreasureConcave else f"smooth_{i}.json"
            )
            with open(file_path) as f:
                data = json.load(f)
            ret_vec = data.get("return", None)
            for states, actions in data["trajectories"]:
                # The trajectories from JSON are lists of lists, convert to numpy arrays
                obs_list = list(states)
                obs_list.append(states[-1])  # Repeat the last state
                obs = np.array(obs_list, dtype=np.float32)
                acts = np.array(actions, dtype=np.float32)

                # Create a Trajectory object. infos and terminal are set to defaults.
                traj = Trajectory(obs=obs, acts=acts, infos=None, terminal=True)
                if single_trajectory is None:
                    single_trajectory = traj
                trajectories.append(traj)
                true_labels.append(i)
                if ret_vec is not None:
                    obj_feats_list.append(np.asarray(ret_vec, dtype=np.float64))
                else:
                    obj_feats_list.append(None)
            single_trajectories_manager[i] = {"trajectory": single_trajectory, "return": ret_vec}
            single_trajectories.append(single_trajectory)
            true_labels_single.append(i)

        true_labels = np.array(true_labels)
        true_labels_single = np.unique(np.array(true_labels_single))
        if len(obj_feats_list) == len(trajectories) and all(x is not None for x in obj_feats_list):
            obj_feats_from_json = np.vstack(obj_feats_list).astype(np.float64)
        else:
            obj_feats_from_json = None

        # It will pad trajectories since they have different lengths and prepare them for the model.
        all_states, all_actions, all_masks, all_labels, max_len = prepare_sa_trajectories(env_id, trajectories, true_labels)
        all_states_single, all_actions_single, all_masks_single, all_labels_single, max_len_single = (
            prepare_sa_trajectories(env_id, single_trajectories, true_labels_single)
        )

        input_coord_dims = trajectories[0].obs.shape[1] if len(trajectories) > 0 and trajectories[0].obs.ndim > 1 else 1
        num_actions = trajectories[0].acts.shape[1] if len(trajectories) > 0 and trajectories[0].acts.ndim > 1 else 1
        num_trajs = len(trajectories)
        K = len(np.unique(true_labels))
        print(f"Loaded {num_trajs} expert trajectories for {name_env} with {K} modes.")

    else:
        raise ValueError(
            "No available environment selected. Please select either --Two Lakes Fishing (-F), --Traj2d(-T2D), --Reacher-v4(-Rv4) or --Pusher-v4(-Pv4) or --Humanoidv4(-Hv4) or --Walker2dv4(-Wv4)"
        )

    if not args.DeepSeaTreasure and not args.DeepSeaTreasureConcave and not args.DeepSeaTreasureSmooth:
        print("--- Preparing State-Action Tensors")
        all_states, all_actions, all_masks, all_labels, max_len = prepare_sa_trajectories(env_id, trajectories, true_labels)
        all_states_online, all_actions_online, all_masks_online, all_labels_online, max_len_online = (
            prepare_sa_trajectories(env_id, unseen_trajectories_for_online, true_labels_online)
        )

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
            plt.scatter(
                umap_proj[idx, 0], umap_proj[idx, 1], c=label_colors.get(lbl, "gray"), label=f"Mode {lbl - 10}", alpha=0.7
            )
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
            min_dist=0.5,
        )

    # state_scaler = "robust"
    state_scaler = "quantile_normal"
    # action_scaler = "robust"
    action_scaler = "quantile_normal"
    simple_scaler_fit = "seen"

    if normalize_input:
        if is_coord_states(all_states[0]) and state_scaler != "none":
            ss = build_scaler(state_scaler)
            print(f"[Simple] Fitting state scaler='{state_scaler}' on {simple_scaler_fit}.")
            ss = fit_on_flat(
                all_states,
                ss,
                fit_mode=simple_scaler_fit,
                list_of_td_online=all_states_online if simple_scaler_fit == "both" else None,
            )
            all_states = apply_per_traj(all_states, ss)
            print("[Simple] States scaled.")

        # Actions: only if continuous [T,A]
        if is_cont_actions(all_actions) and action_scaler != "none":
            sa = build_scaler(action_scaler)
            print(f"[Simple] Fitting action scaler='{action_scaler}' on {simple_scaler_fit}.")
            sa = fit_on_flat(
                all_actions,
                sa,
                fit_mode=simple_scaler_fit,
                list_of_td_online=all_actions_online if simple_scaler_fit == "both" else None,
            )
            all_actions = apply_per_traj(all_actions, sa)
            print("[Simple] Actions scaled.")
        try:
            if "all_states_single" in locals() and all_states_single is not None:
                if is_coord_states(all_states_single[0]) and state_scaler != "none":
                    ss_single = build_scaler(state_scaler)
                    print(f"[Single] Fitting state scaler='{state_scaler}' for single trajectories on {simple_scaler_fit}.")
                    ss_single = fit_on_flat(
                        all_states_single,
                        ss_single,
                        fit_mode=simple_scaler_fit,
                        list_of_td_online=all_states_online
                        if simple_scaler_fit == "both" and "all_states_online" in locals()
                        else None,
                    )
                    all_states_single = apply_per_traj(all_states_single, ss_single)
                    print("[Single] States scaled.")

            if "all_actions_single" in locals() and all_actions_single is not None:
                if is_cont_actions(all_actions_single) and action_scaler != "none":
                    sa_single = build_scaler(action_scaler)
                    print(
                        f"[Single] Fitting action scaler='{action_scaler}' for single trajectories on {simple_scaler_fit}."
                    )
                    sa_single = fit_on_flat(
                        all_actions_single,
                        sa_single,
                        fit_mode=simple_scaler_fit,
                        list_of_td_online=all_actions_online
                        if simple_scaler_fit == "both" and "all_actions_online" in locals()
                        else None,
                    )
                    all_actions_single = apply_per_traj(all_actions_single, sa_single)
                    print("[Single] Actions scaled.")
        except Exception as e:
            print("Warning: failed to normalize single trajectories:", e)

    print("--- Populating Trajectory Manager ---")
    for i in range(len(trajectories)):
        trajectory_manager[i] = {
            "id": i,
            "original_trajectory": trajectories[i],
            "prepared_states": all_states[i],
            "prepared_actions": all_actions[i],
            "prepared_masks": all_masks[i],
            "real_cluster_label": all_labels[i],
        }
    print("Len trajectory manager:", len(trajectory_manager))

    obs_shape = trajectories[0].obs[0].shape
    print(f"Generated {len(trajectories)} expert trajectories.")
    transformer_folder = "./morl_behavior_objective/methods/transformer_folder"
    model_folder = transformer_folder + f"/models/scaler_{state_scaler}/{env_code}/ntrj_{num_trajs}"
    csv_folder = f"./csvs/{env_id}_CoMIIRL_results"
    csv_file_path = os.path.join(csv_folder, f"{env_code}_ntrj_{num_trajs}_ratio_{ratio}.csv")
    os.makedirs(model_folder, exist_ok=True)
    if saving:
        os.makedirs(csv_folder, exist_ok=True)
    image_dir = f"./images/{env_id}/"
    os.makedirs(image_dir, exist_ok=True)

    results = []

    seq_max_len = max_len
    train_size = int(num_trajs * 0.9)
    val_size = int(num_trajs * 0.0)
    test_size = int(len(trajectories) - train_size - val_size)

    ### transformer hyperparameters

    tr_lr = (
        0.0001
        if args.Traj2d
        else 0.0001
        if args.Reacherv4
        else 0.0001
        if args.Pusherv4
        else 0.0001
        if args.DeepSeaTreasure
        else 0.0001
        if args.DeepSeaTreasureConcave
        else 0.0001
    )
    input_channels = obs_shape[0]

    emb_dim = (
        12
        if args.Traj2d
        else 32
        if args.Reacherv4
        else 32
        if args.Pusherv4
        else 3
        if args.DeepSeaTreasure
        else 3
        if args.DeepSeaTreasureConcave
        else 3
    )
    cnn_output_dim = emb_dim

    num_heads = (
        4
        if args.Traj2d
        else 4
        if args.Reacherv4
        else 4
        if args.Pusherv4
        else 3
        if args.DeepSeaTreasure
        else 3
        if args.DeepSeaTreasureConcave
        else 3
    )
    nlayers = (
        2
        if args.Traj2d
        else 2
        if args.Reacherv4
        else 2
        if args.Pusherv4
        else 1
        if args.DeepSeaTreasure
        else 1
        if args.DeepSeaTreasureConcave
        else 1
    )
    d_hid = (
        1024
        if args.Traj2d
        else 1024
        if args.Reacherv4
        else 1024
        if args.Pusherv4
        else 32
        if args.DeepSeaTreasure
        else 32
        if args.DeepSeaTreasureConcave
        else 32
    )
    loader_batch = 64 if args.Traj2d else 64 if args.Reacherv4 else 32 if args.Pusherv4 else 32
    val_bptt = 8
    test_bptt = 1
    dropout = 0.1 if args.Traj2d else 0.1 if args.Reacherv4 else 0.1 if args.Pusherv4 else 0.1

    # Fourier feature encoding, - gaussian mapping values
    gaussian_m_state = (
        64 if args.Traj2d else 512 if args.Reacherv4 else 1024 if args.Pusherv4 else 64
    )  # bigger m -> better kernel approximation and cross-dim mixing
    gaussian_m_action = 32 if args.Traj2d else 256 if args.Reacherv4 else 512 if args.Pusherv4 else 32
    gaussian_sigma_state = (
        10 if args.Traj2d else 5 if args.Reacherv4 else 5 if args.Pusherv4 else 10
    )  # smaller -> high frequency, larger -> smoother features
    gaussian_sigma_action = (
        10 if args.Traj2d else 5 if args.Reacherv4 else 5 if args.Pusherv4 else 10
    )  # smaller -> high frequency, larger -> smoother features

    training_beta = 0.5 if env_id == "Reacherv4" else 0.5 if args.DeepSeaTreasure else 0.5  # beta for contrastive
    training_gamma = 1.0 if env_id == "Reacherv4" else 1.0 if args.DeepSeaTreasure else 1.0  # gamma for infomax

    bptt = loader_batch

    config_name = f"emb_dim_{emb_dim}_nh_{num_heads}_nl_{nlayers}_hd_{d_hid}_max_len_{seq_max_len}_seed_{SEED}_ratio_{ratio}_strategy_{embedding_strategy_code}_training_{training_code}"
    config_name = config_name + f"_modes_{unseen_modes}" if args.use_seen_unseen_split else config_name

    print("--- Creating Dataloaders for State-Action Trajectories ---")
    (
        full_dataset,
        train_dataset,
        val_dataset,
        test_dataset,
        total_dataloader,
        train_dataloader,
        val_dataloader,
        test_dataloader,
    ) = datasets_preparation_sa(
        all_states,
        all_actions,
        all_masks,
        true_labels,
        train_size,
        val_size,
        test_size,
        loader_batch,
        val_bptt,
        test_bptt,
        SEED,
    )
    (
        full_dataset_single,
        train_dataset_single,
        val_dataset_single,
        test_dataset_single,
        total_dataloader_single,
        train_dataloader_single,
        val_dataloader_single,
        test_dataloader_single,
    ) = datasets_preparation_sa(
        all_states_single, all_actions_single, all_masks_single, true_labels_single, num_policies, 0, 0, 1, 1, 1, SEED
    )

    print(f"*-*-*-*-*-*-*-*-* Datasets created from {len(full_dataset)} trajectories")
    # Create the model using the factory function for state-action models
    behaviorencoder = create_model_BECwASATyped(
        input_channels=input_channels,
        cnn_output_dim=cnn_output_dim,
        steps=seq_max_len,  # Use the max_len from the data preparation step
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
        gaussian_sigma_action=gaussian_sigma_action,
    )

    behaviorencoder.to(device)
    transformer_total_params = sum(p.numel() for p in behaviorencoder.parameters() if p.requires_grad)
    print(behaviorencoder.model_type, "parameters ->", transformer_total_params / 1e6, "M")

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
            beta=training_beta,
            gamma=training_gamma,
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

    trajectory_manager, concatenations_train, indices_train = inference_on_dataloader_cdec_sa(
        behaviorencoder, cluster_centroids, train_dataloader, trajectory_manager, device
    )

    print("Inference on test dataloader...")
    trajectory_manager, concatenations_test, indices_test = inference_on_dataloader_cdec_sa(
        behaviorencoder, cluster_centroids, test_dataloader, trajectory_manager, device
    )

    print("Inference on single dataset")
    single_trajectories_manager, concatenations_single, indices_single = inference_on_dataloader_cdec_sa(
        behaviorencoder, cluster_centroids, total_dataloader_single, single_trajectories_manager, device
    )

    # --- Process Embeddings for Visualization ---
    tj_embeddings_train_true_labels = [trajectory_manager[i]["real_cluster_label"].item() for i in indices_train]
    tj_embeddings_test_true_labels = [trajectory_manager[i]["real_cluster_label"].item() for i in indices_test]

    tj_embeddings_train_pred_labels = [trajectory_manager[i]["predicted_cluster_label"] for i in indices_train]
    tj_embeddings_test_pred_labels = [trajectory_manager[i]["predicted_cluster_label"] for i in indices_test]

    tj_concatenations_train = concatenations_train.squeeze(1).cpu().numpy()
    tj_concatenations_test = concatenations_test.squeeze(1).cpu().numpy()
    tj_concatenations_single = concatenations_single.squeeze(1).cpu().numpy()

    tj_concatenations_evaluation = np.vstack([tj_concatenations_test])
    tj_embeddings_evaluation_true_labels = np.hstack([tj_embeddings_test_true_labels])
    tj_embeddings_evaluations_pred_labels = np.hstack([tj_embeddings_test_pred_labels])
    tj_concatenations_single = np.vstack([tj_concatenations_single])

    tj_concatenations_seen = np.vstack([tj_concatenations_train, tj_concatenations_evaluation])
    tj_embeddings_seen_true_labels = np.hstack([tj_embeddings_train_true_labels, tj_embeddings_evaluation_true_labels])
    tj_embeddings_seen_pred_labels = np.hstack([tj_embeddings_train_pred_labels, tj_embeddings_evaluations_pred_labels])
    tj_concatenations_seen_true_labels = tj_embeddings_seen_true_labels

    # --- Visualization and Metrics ---
    if args.Traj2d or args.Reacherv4 or args.Pusherv4:
        color_label_mapping = {
            "tab:blue": "Mode 0",
            "tab:red": "Mode 1",
            "tab:green": "Mode 2",
            "tab:purple": "Mode 3",
            "tab:brown": "Mode 4",
            "tab:orange": "Mode 5",
            "y": "???",
        }
    elif args.DeepSeaTreasure:
        policy_names = ["7_left", "6_right", "25_left", "24_right", "120_left", "124_right"]
        label_colors = {0: "lightblue", 1: "lightcoral", 2: "dodgerblue", 3: "red", 4: "darkblue", 5: "darkred"}
        label_map = {i: name for i, name in enumerate(policy_names)}
    elif args.DeepSeaTreasureConcave or args.DeepSeaTreasureSmooth:
        num_policies = K
        # Create a color gradient from dark blue to dark red for 10 policies
        colors = sns.blend_palette(["darkblue", "darkred"], n_colors=num_policies)
        label_colors = {i: colors[i] for i in range(num_policies)}
        label_map = {i: f"Mode {i}" for i in range(num_policies)}
    else:
        unique_true_labels = np.unique(tj_embeddings_seen_true_labels)
        label_colors = {0: "lightblue", 1: "lightcoral", 2: "dodgerblue", 3: "red", 4: "darkblue", 5: "darkred"}
        label_map = {label: f"Mode {label}" for label in unique_true_labels}

    out = []

    def _to_list(x):
        if x is None:
            return None
        # torch Tensor
        if isinstance(x, th.Tensor):
            return x.detach().cpu().numpy().tolist()
        # numpy array
        try:
            if isinstance(x, np.ndarray):
                return x.tolist()
        except Exception:
            pass
        # objects with tolist (e.g., lists, nested lists)
        try:
            if hasattr(x, "tolist"):
                return x.tolist()
        except Exception:
            pass
        # fallback to string
        return str(x)

    for i in range(len(single_trajectories_manager)):
        traj = single_trajectories_manager[i].get("trajectory", None)
        entry = {
            "States": _to_list(traj.obs) if traj is not None else None,
            "Actions": _to_list(traj.acts) if traj is not None else None,
            "Return": _to_list(single_trajectories_manager[i].get("return", None)),
            "T_Embedding": _to_list(single_trajectories_manager[i].get("cls_emb", None)),
        }
        out.append(entry)

    os.makedirs(trajectories_directory_path, exist_ok=True)
    out_path = os.path.join(trajectories_directory_path, f"{name_env}_{emb_dim}D_embeddings.json")
    with open(out_path, "w") as jf:
        json.dump(out, jf, indent=2, ensure_ascii=False)
    print(f"Saved single trajectories JSON to {out_path}")

    single_cls_embs = np.array(
        [
            single_trajectories_manager[i].get("cls_emb", np.zeros((emb_dim,)))
            for i in range(len(single_trajectories_manager))
        ]
    )
    single_cls_embs_true_labels = np.array(
        [single_trajectories_manager[i].get("return", None) for i in range(len(single_trajectories_manager))]
    )
    single_cls_embs_true_labels = np.arange(len(single_cls_embs))
    # for i in range(len(single_cls_embs)):
    #     print("Label:", single_cls_embs_true_labels[i], "Embedding:", single_cls_embs[i], "...")

    # Plot single_cls_embs: raw 3D if emb_dim==3, else UMAP to 3D
    if emb_dim == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            single_cls_embs[:, 0],
            single_cls_embs[:, 1],
            single_cls_embs[:, 2],
            c=[label_colors.get(lbl, "gray") for lbl in single_cls_embs_true_labels],
            s=50,
            alpha=0.8,
        )

        ax.set_title(f"Single CLS Embeddings ({name_env})")
        ax.set_xlabel("Dim-1")
        ax.set_ylabel("Dim-2")
        ax.set_zlabel("Dim-3")
        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, f"single_cls_embeddings_{name_env}.png"))
        plt.show()
    else:
        reducer = umap.UMAP(random_state=SEED, n_neighbors=10, min_dist=0.1, n_components=3)
        umap_proj_single = reducer.fit_transform(single_cls_embs)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            umap_proj_single[:, 0],
            umap_proj_single[:, 1],
            umap_proj_single[:, 2],
            c=[label_colors.get(lbl, "gray") for lbl in single_cls_embs_true_labels],
            s=50,
            alpha=0.8,
        )

        ax.set_title(f"UMAP Projection of Single CLS Embeddings ({name_env})")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_zlabel("UMAP-3")
        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, f"UMAP_{name_env}.png"))
        plt.show()


if __name__ == "__main__":
    main()
