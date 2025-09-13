import numpy as np
import math
import random
import json
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
from sklearn.manifold import TSNE
from sklearn.manifold import trustworthiness
from scipy.stats import spearmanr
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

from morl_behavior_objective.methods.behaviorencoder import *
from morl_behavior_objective.methods.behaviorencoder_utils import *
from matplotlib.lines import Line2D



## PARSER

def setup_parser():
    parser = argparse.ArgumentParser(description='Model-Free Multi Intention Maximum Likelihood IRL: Experiments Runner')
    
    arg_env = parser.add_argument_group('Environment Selection')
    arg_env.add_argument("-T2D","--Traj2d", help="Apply the selected algorithm to the Traj2d environment",action="store_true")
    arg_env.add_argument("-Rv4","--Reacherv4", help="Apply the selected algorithm to the Reacher-v4 environment",action="store_true")
    arg_env.add_argument("-Pv4","--Pusherv4", help="Apply the selected algorithm to the Pusher-v4 environment",action="store_true")
    arg_env.add_argument("-DST","--DeepSeaTreasure", help="Apply the selected algorithm to the DeepSeaTreasure environment",action="store_true")
    arg_env.add_argument("-DSTC","--DeepSeaTreasureConcave", help="Apply the selected algorithm to the DeepSeaTreasureConcave environment",action="store_true")

    arg_alg = parser.add_argument_group('Unseen Split Selection')
    arg_alg.add_argument("-split","--use_seen_unseen_split", action="store_true", 
                    help="Use seen-unseen split for training. If not set, train on the entire dataset.")
    arg_alg.add_argument("-nUnModes","--num_unseen_modes", type=int, default=1, help="Number of unseen modes in the unseen split for the dataset.")

    arg_hyp = parser.add_argument_group('Settings') 
    arg_hyp.add_argument('-nT','--num_trajs', type=int,default=100,help='int: Number of expert trajectories to generate')
    arg_hyp.add_argument('-seed','--seed', type=int,default=0,help='int: Random seed for reproducibility') 
    arg_hyp.add_argument('--ratio', type=int, default=1,help="Ratio for splitting trajectories between modes. 1 uniform 3 first gets most last gets least, etc.")
    arg_hyp.add_argument('--embedding_strategy', type=str, default='cls_only', choices=['cls_only', 'hybrid','goal_oriented'], help='Strategy for creating the final trajectory embedding for clustering.')
    arg_hyp.add_argument('--diagnostics', action="store_true", help="Run diagnostic checks on the trajectory embeddings.")

    arg_vis = parser.add_argument_group('Visualization')
    arg_vis.add_argument('-vS','--visualize_scalers', help='Visualize the effect of different scalers on the state space',action="store_true")
    arg_vis.add_argument('-vO','--visualize_original', help='Visualize the original state space',action="store_true")
    arg_vis.add_argument('-vC','--visualize_clusters', help='Visualize the clusters',action="store_true")
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
    elif args.Reacherv4 or args.Pusherv4:
        unseen_modes = args.num_unseen_modes
        K = 6-unseen_modes if args.use_seen_unseen_split else 6
        K_known = 6
    elif args.DeepSeaTreasure:
        K = 6
    elif args.DeepSeaTreasureConcave:
        K = 10
    else:
        raise ValueError("No available environment selected. Please select either --Two Lakes Fishing (-F), --Traj2d(-T2D), --Reacher-v4(-Rv4) or --Pusher-v4(-Pv4).")

    unseen_modes = args.num_unseen_modes
    visualize_original = args.visualize_original
    visualize_scalers = args.visualize_scalers
    normalize_input = True
    saving = False
    num_trajs = args.num_trajs

    diagnostics = args.diagnostics

    embedding_strategy_code = "CLS" if args.embedding_strategy == "cls_only" else "HYB" if args.embedding_strategy == "hybrid" else "GO" if args.embedding_strategy == "goal_oriented" else "UNK"
    training_code = "SPLIT" if args.use_seen_unseen_split else "FULL"

    env_name =  "Traj2d" if args.Traj2d else "Reacher-v4" if args.Reacherv4 else "Pusher-v4" if args.Pusherv4 else "DeepSeaTreasure" if args.DeepSeaTreasure else "DeepSeaTreasureConcave" if args.DeepSeaTreasureConcave else "Unknown"
    env_id =  "Traj2d" if args.Traj2d else "Reacher-v4" if args.Reacherv4 else "Pusher-v4" if args.Pusherv4 else "UnknownEnv"
    env_code = "T2D" if args.Traj2d else "Rv4" if args.Reacherv4 else "Pv4" if args.Pusherv4 else "DST" if args.DeepSeaTreasure else "DSTC" if args.DeepSeaTreasureConcave else "UNK"

    print(f"*** CoMIIRL approach on {env_name} ***")

    # Set the random seed for reproducibility
    SEEDS = [0,1,2,3,4]
    SEED = args.seed
    ratio = args.ratio
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

    trajectory_manager = {}
    # Initialize optional objective features container used by diagnostics
    obj_feats_from_json = None

    if args.Traj2d or args.Reacherv4 or args.Pusherv4:
        trajectories_directory_path = "morl_behavior_objective/old_expert_trajectories/"
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
        env_num_step = 1 #unused in SA case
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
        for i in range(num_policies):
            file_path = os.path.join(trajectories_directory_path, f"{name_env}_{i}.json")
            with open(file_path, 'r') as f:
                data = json.load(f)
            ret_vec = data.get('return', None)
            for states, actions in data['trajectories']:
                # The trajectories from JSON are lists of lists, convert to numpy arrays
                obs_list = list(states)
                obs_list.append(states[-1]) # Repeat the last state
                obs = np.array(obs_list, dtype=np.float32)
                acts = np.array(actions, dtype=np.float32)
                
                # Create a Trajectory object. infos and terminal are set to defaults.
                trajectories.append(Trajectory(obs=obs, acts=acts, infos=None, terminal=True))
                true_labels.append(i)
                if ret_vec is not None:
                    obj_feats_list.append(np.asarray(ret_vec, dtype=np.float64))
                else:
                    obj_feats_list.append(None)

        true_labels = np.array(true_labels)
        if len(obj_feats_list) == len(trajectories) and all(x is not None for x in obj_feats_list):
            obj_feats_from_json = np.vstack(obj_feats_list).astype(np.float64)
        else:
            obj_feats_from_json = None
        # It will pad trajectories since they have different lengths and prepare them for the model.
        all_states, all_actions, all_masks, all_labels, max_len = prepare_sa_trajectories(env_id, trajectories, true_labels)

        input_coord_dims = trajectories[0].obs.shape[1] if len(trajectories) > 0 and trajectories[0].obs.ndim > 1 else 1
        num_actions = trajectories[0].acts.shape[1] if len(trajectories) > 0 and trajectories[0].acts.ndim > 1 else 1
        num_trajs = len(trajectories)
        K = len(np.unique(true_labels))
        print(f"Loaded {num_trajs} expert trajectories for {name_env} with {K} modes.")

    elif args.DeepSeaTreasureConcave:
        name_env = "dst_concave"
        trajectories_directory_path = f"trajectories/{name_env}/"
        num_policies = K
        env_id = "deep-sea-treasure-v0"

        trajectories = []
        true_labels = []
        obj_feats_list = []
        for i in range(num_policies):
            file_path = os.path.join(trajectories_directory_path, f"dst_{i}.json")
            with open(file_path, 'r') as f:
                data = json.load(f)
            ret_vec = data.get('return', None)
            for states, actions in data['trajectories']:
                # The trajectories from JSON are lists of lists, convert to numpy arrays
                obs_list = list(states)
                obs_list.append(states[-1]) # Repeat the last state
                obs = np.array(obs_list, dtype=np.float32)
                acts = np.array(actions, dtype=np.float32)
                
                # Create a Trajectory object. infos and terminal are set to defaults.
                trajectories.append(Trajectory(obs=obs, acts=acts, infos=None, terminal=True))
                true_labels.append(i)
                if ret_vec is not None:
                    obj_feats_list.append(np.asarray(ret_vec, dtype=np.float64))
                else:
                    obj_feats_list.append(None)

        true_labels = np.array(true_labels)
        if len(obj_feats_list) == len(trajectories) and all(x is not None for x in obj_feats_list):
            obj_feats_from_json = np.vstack(obj_feats_list).astype(np.float64)
        else:
            obj_feats_from_json = None

        # It will pad trajectories since they have different lengths and prepare them for the model.
        all_states, all_actions, all_masks, all_labels, max_len = prepare_sa_trajectories(env_id, trajectories, true_labels)

        input_coord_dims = trajectories[0].obs.shape[1] if len(trajectories) > 0 and trajectories[0].obs.ndim > 1 else 1
        num_actions = trajectories[0].acts.shape[1] if len(trajectories) > 0 and trajectories[0].acts.ndim > 1 else 1
        num_trajs = len(trajectories)
        K = len(np.unique(true_labels))
        print(f"Loaded {num_trajs} expert trajectories for {name_env} with {K} modes.")

    else:
        raise ValueError("No available environment selected. Please select either --Two Lakes Fishing (-F), --Traj2d(-T2D), --Reacher-v4(-Rv4) or --Pusher-v4(-Pv4) or --Humanoidv4(-Hv4) or --Walker2dv4(-Wv4)")

    if not args.DeepSeaTreasure and not args.DeepSeaTreasureConcave:
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
        if is_coord_states(all_states[0]) and state_scaler != 'none':
            ss = build_scaler(state_scaler)
            print(f"[Simple] Fitting state scaler='{state_scaler}' on {simple_scaler_fit}.")
            ss = fit_on_flat(all_states, ss, fit_mode=simple_scaler_fit,
                            list_of_td_online=all_states_online if simple_scaler_fit=='both' else None)
            all_states = apply_per_traj(all_states, ss)
            print("[Simple] States scaled.")

        # Actions: only if continuous [T,A]
        if is_cont_actions(all_actions) and action_scaler != 'none':
            sa = build_scaler(action_scaler)
            print(f"[Simple] Fitting action scaler='{action_scaler}' on {simple_scaler_fit}.")
            sa = fit_on_flat(all_actions, sa, fit_mode=simple_scaler_fit,
                            list_of_td_online=all_actions_online if simple_scaler_fit=='both' else None)
            all_actions = apply_per_traj(all_actions, sa)
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

    obs_shape = trajectories[0].obs[0].shape
    print(f"Generated {len(trajectories)} expert trajectories.")
    transformer_folder = "./morl_behavior_objective/methods/transformer_folder"
    model_folder = transformer_folder+f"/models/scaler_{state_scaler}/{env_code}/ntrj_{num_trajs}"
    csv_folder = f"./csvs/{env_id}_CoMIIRL_results"
    csv_file_path = os.path.join(csv_folder, f"{env_code}_ntrj_{num_trajs}_ratio_{ratio}.csv")
    os.makedirs(model_folder, exist_ok=True)
    if saving:
        os.makedirs(csv_folder, exist_ok=True)
    image_dir = f"./images/{env_id}/"
    os.makedirs(image_dir, exist_ok=True)

    results = []

    seq_max_len = max_len
    train_size = int(num_trajs*0.9)
    val_size = int(num_trajs*0.0)
    test_size = int(len(trajectories) - train_size - val_size)
    
    ### transformer hyperparameters

    tr_lr = 0.0001 if args.Traj2d else 0.0001 if args.Reacherv4 else 0.0001 if args.Pusherv4 else 0.0001 if args.DeepSeaTreasure else 0.0001 if args.DeepSeaTreasureConcave else 0.0001
    input_channels = obs_shape[0]

    emb_dim =  12 if args.Traj2d else 32 if args.Reacherv4 else 32 if args.Pusherv4 else 3 if args.DeepSeaTreasure else 3 if args.DeepSeaTreasureConcave else 4
    cnn_output_dim = emb_dim

    num_heads = 4 if args.Traj2d else 4 if args.Reacherv4 else 4 if args.Pusherv4 else 3 if args.DeepSeaTreasure else 3 if args.DeepSeaTreasureConcave else 4
    nlayers = 2 if args.Traj2d else 2 if args.Reacherv4 else 2 if args.Pusherv4 else 1 if args.DeepSeaTreasure else 1 if args.DeepSeaTreasureConcave else 2
    d_hid = 1024 if args.Traj2d else 1024 if args.Reacherv4 else 1024 if args.Pusherv4 else 32 if args.DeepSeaTreasure else 32 if args.DeepSeaTreasureConcave else 1024
    loader_batch = 64 if args.Traj2d else 64 if args.Reacherv4 else 32 if args.Pusherv4 else 32
    val_bptt = 8
    test_bptt = 1
    dropout = 0.1 if args.Traj2d else 0.1 if args.Reacherv4 else 0.1 if args.Pusherv4 else 0.1

    #Fourier feature encoding, - gaussian mapping values
    gaussian_m_state = 64 if args.Traj2d else 512 if args.Reacherv4 else 1024 if args.Pusherv4 else 64 #bigger m -> better kernel approximation and cross-dim mixing
    gaussian_m_action = 32 if args.Traj2d else 256 if args.Reacherv4 else 512 if args.Pusherv4 else 32
    gaussian_sigma_state = 10 if args.Traj2d else 5 if args.Reacherv4 else 5 if args.Pusherv4 else 10 #smaller -> high frequency, larger -> smoother features
    gaussian_sigma_action = 10 if args.Traj2d else 5 if args.Reacherv4 else 5 if args.Pusherv4 else 10 #smaller -> high frequency, larger -> smoother features

    training_beta = 0.5 if env_id == "Reacherv4"  else 0.5 if args.DeepSeaTreasure else 0.5 #beta for contrastive
    training_gamma =  1.0 if env_id == "Reacherv4" else 1.0 if args.DeepSeaTreasure else 1.0 #gamma for infomax

    bptt = loader_batch

    config_name = f"emb_dim_{emb_dim}_nh_{num_heads}_nl_{nlayers}_hd_{d_hid}_max_len_{seq_max_len}_seed_{SEED}_ratio_{ratio}_strategy_{embedding_strategy_code}_training_{training_code}"
    config_name = config_name + f"_modes_{unseen_modes}" if args.use_seen_unseen_split else config_name


    print("--- Creating Dataloaders for State-Action Trajectories ---")
    full_dataset, train_dataset, val_dataset, test_dataset, total_dataloader, train_dataloader, val_dataloader, test_dataloader = datasets_preparation_sa(all_states,all_actions,all_masks,true_labels,train_size,val_size,test_size,loader_batch,val_bptt,test_bptt,SEED)

    print(f"*-*-*-*-*-*-*-*-* Datasets created from {len(full_dataset)} trajectories")
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
            beta = training_beta,
            gamma = training_gamma,
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

    trajectory_manager, concatenations_train, indices_train = \
        inference_on_dataloader_cdec_sa(behaviorencoder, cluster_centroids, train_dataloader, trajectory_manager, device)

    print("Inference on test dataloader...")
    trajectory_manager, concatenations_test, indices_test = \
        inference_on_dataloader_cdec_sa(behaviorencoder, cluster_centroids, test_dataloader, trajectory_manager, device)


    # --- Process Embeddings for Visualization ---
    tj_embeddings_train_true_labels = [trajectory_manager[i]['real_cluster_label'].item() for i in indices_train]
    tj_embeddings_test_true_labels = [trajectory_manager[i]['real_cluster_label'].item() for i in indices_test]


    tj_embeddings_train_pred_labels = [trajectory_manager[i]['predicted_cluster_label'] for i in indices_train]
    tj_embeddings_test_pred_labels = [trajectory_manager[i]['predicted_cluster_label'] for i in indices_test]

    tj_concatenations_train = concatenations_train.squeeze(1).cpu().numpy()
    tj_concatenations_test = concatenations_test.squeeze(1).cpu().numpy()

    tj_concatenations_evaluation = np.vstack([tj_concatenations_test])
    tj_embeddings_evaluation_true_labels = np.hstack([tj_embeddings_test_true_labels])
    tj_embeddings_evaluations_pred_labels = np.hstack([tj_embeddings_test_pred_labels])

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
                "y": "???"
            }
    elif args.DeepSeaTreasure:
        policy_names = ["7_left", "6_right", "25_left", "24_right", "120_left", "124_right"]
        label_colors = {
            0: "lightblue", 1: "lightcoral",
            2: "dodgerblue", 3: "red",
            4: "darkblue", 5: "darkred"
        }
        label_map = {i: name for i, name in enumerate(policy_names)}
    elif args.DeepSeaTreasureConcave:
        num_policies = K
        # Create a color gradient from light blue to dark red for 10 policies
        colors = sns.blend_palette(["lightblue", "darkred"], n_colors=num_policies)
        label_colors = {i: colors[i] for i in range(num_policies)}
        label_map = {i: f"Mode {i}" for i in range(num_policies)}
    else:
        label_colors = {
            0: "lightblue", 1: "lightcoral",
            2: "dodgerblue", 3: "red",
            4: "darkblue", 5: "darkred"
        }
        label_map = {label: f'Mode {label}' for label in unique_true_labels}
        
    reducer = umap.UMAP(
            random_state=SEED,
            n_neighbors=100,
            min_dist=0.99,
            n_components=2 if emb_dim != 3 else 3,
            metric='cosine',
        )

    umap_combined = reducer.fit_transform(tj_concatenations_seen)

    # If embeddings are 3D, show a raw 3D scatter of the learned embeddings (no reduction)
    if args.visualize_clusters and emb_dim == 3:
        print("\n--- Visualizing RAW 3D Embeddings (no reduction) ---")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        unique_true_labels = np.unique(tj_embeddings_seen_true_labels)
        for label in unique_true_labels:
            mask = tj_embeddings_seen_true_labels == label
            pts = tj_concatenations_seen[mask]
            if pts.size == 0:
                continue
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=label_colors.get(label, 'gray'),
                       label=label_map.get(label, f'Mode {label}'),
                       alpha=0.7, s=50)
        ax.set_title(f'Raw 3D Embeddings with True Labels ({env_id})')
        ax.set_xlabel('Emb-1')
        ax.set_ylabel('Emb-2')
        ax.set_zlabel('Emb-3')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.tight_layout(rect=[0, 0, 0.85, 1])
        image_filename = os.path.join(image_dir, f"3d_space_{name_env}_emb_dim{emb_dim}.pdf")
        plt.savefig(image_filename, bbox_inches='tight')
        print(f"Saved raw 3D embedding plot to {image_filename}")
        plt.show()

    # --- Simple UMAP-TSNE Visualization with True Labels ---
    else:
        print("\n--- Visualizing UMAP with True Labels ---")
        fig = plt.figure(figsize=(10, 8))
        
        # Define colors for true labels
        unique_true_labels = np.unique(tj_embeddings_seen_true_labels)
        
        ax = fig.add_subplot(111)
        for label in unique_true_labels:
            mask = tj_embeddings_seen_true_labels == label
            ax.scatter(umap_combined[mask, 0], umap_combined[mask, 1],
                        c=label_colors.get(label, 'gray'),
                        label=label_map.get(label, f'Mode {label}'),
                        alpha=0.7, s=50)
        ax.set_title(f'2D UMAP of Embeddings with True Labels ({env_id})')
        ax.set_xlabel('UMAP-1')
        ax.set_ylabel('UMAP-2')

        ax.legend()
        plt.tight_layout()
        
        image_filename = os.path.join(image_dir, f"space_{name_env}_emb_dim{emb_dim}.pdf")
        
        plt.savefig(image_filename, bbox_inches='tight')
        print(f"Saved UMAP plot to {image_filename}")

        plt.show()

        # --- TSNE Visualization (same data as UMAP) ---
        print("\n--- Visualizing t-SNE of Embeddings with True Labels ---")
        tsne = TSNE(n_components=2, random_state=SEED, init='pca', metric='cosine')
        tsne_proj = tsne.fit_transform(tj_concatenations_seen)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        unique_true_labels = np.unique(tj_embeddings_seen_true_labels)
        for label in unique_true_labels:
            mask = tj_embeddings_seen_true_labels == label
            pts = tsne_proj[mask]
            if pts.size == 0:
                continue
            ax.scatter(pts[:, 0], pts[:, 1], c=label_colors.get(label, 'gray'), label=label_map.get(label, f'Mode {label}'), alpha=0.7, s=50)

        ax.set_title(f't-SNE of Embeddings with True Labels ({env_id})')
        ax.set_xlabel('tSNE-1')
        ax.set_ylabel('tSNE-2')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.tight_layout(rect=[0, 0, 0.85, 1])
        image_filename = os.path.join(image_dir, f"tsne_space_{name_env}_emb_dim{emb_dim}.pdf")
        plt.savefig(image_filename, bbox_inches='tight')
        print(f"Saved t-SNE plot to {image_filename}")

        plt.show()


    if diagnostics:
        print('\n--- Running embedding / objective-space diagnostics ---')
        try:
            E = tj_concatenations_seen.astype(np.float64)
        except Exception:
            E = np.array(tj_concatenations_seen, dtype=np.float64)

        
        Obj_feats = None

        if obj_feats_from_json is not None and len(obj_feats_from_json) == E.shape[0]:
            Obj_feats = np.array(obj_feats_from_json, dtype=np.float64)
            print('Objective features: loaded per-trajectory returns from JSON.')
        else:
            try:
                pair_positions = {0: 0.00, 1: 0.05, 2: 2.00, 3: 2.05, 4: 4.00, 5: 4.05}
                Obj_feats = np.array([pair_positions[int(lbl)] for lbl in tj_embeddings_seen_true_labels]).reshape(-1, 1)
                print('Objective features: built pseudo-objective from known left-right pairs (pairs are close).')
            except Exception:
                # Last resort: use flattened observed states if available
                try:
                    Obj_feats = traj_flat_seen.astype(np.float64)
                    print('Objective features: using flattened state vectors as fallback.')
                except Exception:
                    Obj_feats = np.array(tj_embeddings_seen_true_labels).reshape(-1, 1)
                    print('Objective features: fallback to true labels (proxy).')

        # PCA + participation ratio
        pca = PCA(n_components=min(E.shape[1], E.shape[0]-1))
        pca.fit(E)
        explained = pca.explained_variance_ratio_
        cum = explained.cumsum()
        eigvals = pca.explained_variance_
        participation_ratio = (eigvals.sum()**2) / (eigvals**2).sum() if eigvals.size > 0 else float('nan')

        # Intrinsic dim (MLE) — add tiny jitter and use adaptive k to avoid NaNs from duplicates
        try:
            E_jitter = E + 1e-9 * np.random.randn(*E.shape)
            k_mle = max(3, min(10, E.shape[0] // 4))
            id_mle = mle_intrinsic_dim(E_jitter, k=k_mle)
        except Exception:
            id_mle = float('nan'); k_mle = None

        # Clustering / separability
        unique_true = np.unique(tj_embeddings_seen_true_labels)
        K_est = len(unique_true)
        try:
            km = KMeans(n_clusters=max(1, K_est), random_state=0).fit(E)
            ari = adjusted_rand_score(tj_embeddings_seen_true_labels, km.labels_)
            nmi = normalized_mutual_info_score(tj_embeddings_seen_true_labels, km.labels_)
        except Exception:
            ari = float('nan'); nmi = float('nan')
        try:
            sil_true = silhouette_score(E, tj_embeddings_seen_true_labels) if len(unique_true) > 1 else float('nan')
        except Exception:
            sil_true = float('nan')

        # Trustworthiness (local neighborhood preservation) between objective and embedding spaces
        try:
            tw = trustworthiness(Obj_feats, E, n_neighbors=5)
        except Exception:
            tw = float('nan')

        # Pairwise distance correlations
        D_emb = pairwise_distances(E, metric='euclidean')
        D_obj = pairwise_distances(Obj_feats, metric='euclidean')

        # Optional: standardize Obj_feats before computing distances (uncomment if desired)
        # _sc = StandardScaler()
        # Obj_feats_std = _sc.fit_transform(Obj_feats)
        # D_obj = pairwise_distances(Obj_feats_std, metric='euclidean')

        mask = np.triu_indices_from(D_emb, k=1)
        try:
            rho, pval = spearmanr(D_emb[mask], D_obj[mask])
        except Exception:
            rho, pval = float('nan'), float('nan')

        # --- Within-pair vs Between-pair distances: only meaningful for left-right DST ---
        if args.DeepSeaTreasure:
            try:
                pairs = {(0, 1), (2, 3), (4, 5)}
                within = []
                between = []
                n = D_emb.shape[0]
                for i in range(n):
                    for j in range(i+1, n):
                        a = int(tj_embeddings_seen_true_labels[i])
                        b = int(tj_embeddings_seen_true_labels[j])
                        if (a, b) in pairs or (b, a) in pairs:
                            within.append(D_emb[i, j])
                        else:
                            between.append(D_emb[i, j])
                mean_within = float(np.mean(within)) if len(within) > 0 else float('nan')
                mean_between = float(np.mean(between)) if len(between) > 0 else float('nan')
                # boxplot
                try:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.boxplot([within, between], labels=['within-pair', 'between-pair'])
                    ax.set_ylabel('Embedding distance')
                    ax.set_title(f'Within vs Between pair distances (emb_dim={E.shape[1]})')
                    img_path = os.path.join(image_dir, f'within_between_embdist_embdim{E.shape[1]}.pdf')
                    fig.tight_layout()
                    plt.savefig(img_path, bbox_inches='tight')
                    plt.close(fig)
                except Exception:
                    img_path = None
            except Exception:
                mean_within = float('nan')
                mean_between = float('nan')
                img_path = None
        else:
            mean_within = float('nan')
            mean_between = float('nan')
            img_path = None

        # Mantel-like permutation test (two-sided) + optional histogram

        try:
            mantel_r, mantel_p, mantel_perm_rs, mantel_hist_path = mantel_permutation_test_two_sided(D_emb, D_obj, perms=1000, seed=SEED,image_dir=image_dir, E=E)
        except Exception:
            mantel_r, mantel_p, mantel_perm_rs, mantel_hist_path = float('nan'), float('nan'), [], None

        # kNN overlap (global)
        try:
            mean_ov, quartiles = knn_overlap(Obj_feats, E, k=5)
        except Exception:
            mean_ov, quartiles = float('nan'), [float('nan')]*3

    # Pareto-restricted kNN overlap: nearest neighbors in Obj_feats vs E (local test)
        try:
            k = min(5, E.shape[0]-1)
            nn_obj = np.argsort(D_obj, axis=1)[:, 1:k+1]
            nn_emb = np.argsort(D_emb, axis=1)[:, 1:k+1]
            overlaps = [len(set(nn_obj[i]) & set(nn_emb[i])) / float(k) for i in range(E.shape[0])]
            pareto_knn_mean = float(np.mean(overlaps))
        except Exception:
            pareto_knn_mean = float('nan')

        # Lipschitz histogram diagnostics (local sensitivity both directions)
        try:
            obj_for_lip = Obj_feats if Obj_feats.ndim == 2 else Obj_feats.reshape(-1,1)
            k_lip = min(5, obj_for_lip.shape[0]-1) if obj_for_lip.shape[0] > 1 else 1
            eps = 1e-12
            min_sep = 1e-6  # New: avoid dividing by near-zero distances
            if k_lip >= 1:
                # objective space neighbors
                nn_obj = NearestNeighbors(n_neighbors=k_lip+1).fit(obj_for_lip)
                dist_obj_nn, idx_obj_nn = nn_obj.kneighbors(obj_for_lip)
                dist_obj_nn = dist_obj_nn[:,1:]
                idx_obj_nn = idx_obj_nn[:,1:]
                emb_d_objnbr = np.linalg.norm(E[:,None,:]-E[idx_obj_nn], axis=2)
                mask_obj = dist_obj_nn > min_sep
                ratios_obj2emb = np.full_like(emb_d_objnbr, np.nan, dtype=np.float64)
                ratios_obj2emb[mask_obj] = emb_d_objnbr[mask_obj]/(dist_obj_nn[mask_obj]+eps)
                local_med_obj2emb = np.nanmedian(ratios_obj2emb, axis=1)
                local_max_obj2emb = np.nanmax(ratios_obj2emb, axis=1)
                # embedding space neighbors
                nn_emb = NearestNeighbors(n_neighbors=k_lip+1).fit(E)
                dist_emb_nn, idx_emb_nn = nn_emb.kneighbors(E)
                dist_emb_nn = dist_emb_nn[:,1:]
                idx_emb_nn = idx_emb_nn[:,1:]
                obj_d_embnbr = np.linalg.norm(obj_for_lip[:,None,:]-obj_for_lip[idx_emb_nn], axis=2)
                mask_emb = dist_emb_nn > min_sep
                ratios_emb2obj = np.full_like(obj_d_embnbr, np.nan, dtype=np.float64)
                ratios_emb2obj[mask_emb] = obj_d_embnbr[mask_emb]/(dist_emb_nn[mask_emb]+eps)
                local_med_emb2obj = np.nanmedian(ratios_emb2obj, axis=1)
                local_max_emb2obj = np.nanmax(ratios_emb2obj, axis=1)
                # hist helper
                def save_hist(data, name):
                    try:
                        d = data[np.isfinite(data)]
                        plt.figure(figsize=(5,3.2))
                        sns.histplot(d, bins=60, stat='density', color='C0', alpha=0.8)
                        if d.size and (np.max(d) / max(np.median(d), 1e-9) > 1e3):
                            plt.xscale('log')
                        plt.xlabel(name); plt.ylabel('Density')
                        outp = os.path.join(image_dir, f'{name}_embdim{E.shape[1]}.pdf')
                        plt.tight_layout(); plt.savefig(outp); plt.close()
                        return outp
                    except Exception:
                        return None
                lip_obj2emb_med_pdf = save_hist(local_med_obj2emb, 'lip_local_med_obj2emb')
                lip_obj2emb_max_pdf = save_hist(local_max_obj2emb, 'lip_local_max_obj2emb')
                lip_emb2obj_med_pdf = save_hist(local_med_emb2obj, 'lip_local_med_emb2obj')
                lip_emb2obj_max_pdf = save_hist(local_max_emb2obj, 'lip_local_max_emb2obj')
                # summaries
                def s(a):
                    d = a[np.isfinite(a)]
                    return dict(
                        median=float(np.median(d)) if d.size else float('nan'),
                        p90=float(np.percentile(d,90)) if d.size else float('nan'),
                        frac_gt10=float(np.mean(d>10)) if d.size else float('nan'),
                        max=float(np.max(d)) if d.size else float('nan')
                    )
                lip_summary = {
                    'obj2emb_med': s(local_med_obj2emb),
                    'obj2emb_max': s(local_max_obj2emb),
                    'emb2obj_med': s(local_med_emb2obj),
                    'emb2obj_max': s(local_max_emb2obj),
                    'obj2emb_med_pdf': lip_obj2emb_med_pdf,
                    'obj2emb_max_pdf': lip_obj2emb_max_pdf,
                    'emb2obj_med_pdf': lip_emb2obj_med_pdf,
                    'emb2obj_max_pdf': lip_emb2obj_max_pdf,
                    'k_lip': k_lip
                }
            else:
                lip_summary = {}
        except Exception:
            lip_summary = {}
        
        # --- Centroid-level diagnostics (policy-level) to mitigate duplicate objectives ---
        try:
            labels_arr = np.asarray(tj_embeddings_seen_true_labels)
            uniq = np.unique(labels_arr)
            # compute centroids in embedding and objective spaces
            E_centroids = []
            O_centroids = []
            counts_centroids = []
            for lab in uniq:
                idx = np.where(labels_arr == lab)[0]
                if idx.size == 0:
                    continue
                E_centroids.append(E[idx].mean(axis=0))
                O_centroids.append(Obj_feats[idx].mean(axis=0))
                counts_centroids.append(idx.size)
            E_c = np.vstack(E_centroids)
            O_c = np.vstack(O_centroids)
            # PCA/participation ratio on centroids (policy-level)
            try:
                pca_c = PCA(n_components=min(E_c.shape[1], max(1, E_c.shape[0]-1)))
                pca_c.fit(E_c)
                explained_c = pca_c.explained_variance_ratio_
                cum_c = explained_c.cumsum()
                eigvals_c = pca_c.explained_variance_
                participation_ratio_c = (eigvals_c.sum()**2) / (eigvals_c**2).sum() if eigvals_c.size > 0 else float('nan')
            except Exception:
                cum_c = np.array([]); participation_ratio_c = float('nan')
            # Intrinsic dimension (MLE) on centroids with tiny jitter and small k
            try:
                E_c_jitter = E_c + 1e-9 * np.random.randn(*E_c.shape)
                k_mle_c = max(2, min(5, E_c.shape[0] - 1))
                id_mle_c = mle_intrinsic_dim(E_c_jitter, k=k_mle_c)
            except Exception:
                id_mle_c = float('nan'); k_mle_c = None
            D_emb_c = pairwise_distances(E_c, metric='euclidean')
            D_obj_c = pairwise_distances(O_c, metric='euclidean')
            mask_c = np.triu_indices_from(D_emb_c, k=1)
            try:
                rho_c, pval_c = spearmanr(D_emb_c[mask_c], D_obj_c[mask_c])
            except Exception:
                rho_c, pval_c = float('nan'), float('nan')
            # two-sided Mantel on centroids, save histogram under a subdir
            image_dir_centroids = os.path.join(image_dir, 'centroids')
            os.makedirs(image_dir_centroids, exist_ok=True)
            try:
                mantel_r_c, mantel_p_c, _, mantel_hist_path_c = mantel_permutation_test_two_sided(D_emb_c, D_obj_c, perms=1000, seed=SEED, image_dir=image_dir_centroids, E=E_c)
            except Exception:
                mantel_r_c, mantel_p_c, mantel_hist_path_c = float('nan'), float('nan'), None
            # Trustworthiness and kNN overlap at centroid level (objective -> embedding)
            try:
                tw_c = trustworthiness(O_c, E_c, n_neighbors=min(5, max(1, E_c.shape[0]-1)))
            except Exception:
                tw_c = float('nan')
            try:
                mean_ov_c, quartiles_c = knn_overlap(O_c, E_c, k=min(5, max(1, E_c.shape[0]-1)))
            except Exception:
                mean_ov_c, quartiles_c = float('nan'), [float('nan')]*3
            # Pareto-like kNN overlap at centroid level
            try:
                k_cnn = min(5, max(1, E_c.shape[0]-1))
                nn_obj_cnn = np.argsort(D_obj_c, axis=1)[:, 1:k_cnn+1]
                nn_emb_cnn = np.argsort(D_emb_c, axis=1)[:, 1:k_cnn+1]
                overlaps_c = [len(set(nn_obj_cnn[i]) & set(nn_emb_cnn[i])) / float(k_cnn) for i in range(E_c.shape[0])]
                pareto_knn_mean_c = float(np.mean(overlaps_c)) if len(overlaps_c) else float('nan')
            except Exception:
                pareto_knn_mean_c = float('nan')
            # Within-vs-between (policy-level) only for left-right DST
            if args.DeepSeaTreasure:
                try:
                    pairs = {(0, 1), (2, 3), (4, 5)}
                    # map centroid row index -> original label
                    label_by_row = {ri: int(lab) for ri, lab in enumerate(uniq)}
                    within_c, between_c = [], []
                    n_c = D_emb_c.shape[0]
                    for i in range(n_c):
                        for j in range(i+1, n_c):
                            a = label_by_row[i]; b = label_by_row[j]
                            if (a, b) in pairs or (b, a) in pairs:
                                within_c.append(D_emb_c[i, j])
                            else:
                                between_c.append(D_emb_c[i, j])
                    mean_within_c = float(np.mean(within_c)) if len(within_c) > 0 else float('nan')
                    mean_between_c = float(np.mean(between_c)) if len(between_c) > 0 else float('nan')
                except Exception:
                    mean_within_c, mean_between_c = float('nan'), float('nan')
            else:
                mean_within_c, mean_between_c = float('nan'), float('nan')
            # Lipschitz at centroid level (use small k)
            try:
                k_c = min(3, E_c.shape[0]-1) if E_c.shape[0] > 1 else 1
                eps = 1e-12
                min_sep = 1e-6
                if k_c >= 1 and E_c.shape[0] > 1:
                    nn_obj_c = NearestNeighbors(n_neighbors=k_c+1).fit(O_c)
                    dist_obj_nn_c, idx_obj_nn_c = nn_obj_c.kneighbors(O_c)
                    dist_obj_nn_c = dist_obj_nn_c[:,1:]
                    idx_obj_nn_c = idx_obj_nn_c[:,1:]
                    emb_d_objnbr_c = np.linalg.norm(E_c[:,None,:]-E_c[idx_obj_nn_c], axis=2)
                    mask_obj_c = dist_obj_nn_c > min_sep
                    ratios_o2e_c = np.full_like(emb_d_objnbr_c, np.nan, dtype=np.float64)
                    ratios_o2e_c[mask_obj_c] = emb_d_objnbr_c[mask_obj_c]/(dist_obj_nn_c[mask_obj_c]+eps)
                    med_o2e_c = np.nanmedian(ratios_o2e_c, axis=1)
                    nn_emb_c = NearestNeighbors(n_neighbors=k_c+1).fit(E_c)
                    dist_emb_nn_c, idx_emb_nn_c = nn_emb_c.kneighbors(E_c)
                    dist_emb_nn_c = dist_emb_nn_c[:,1:]
                    idx_emb_nn_c = idx_emb_nn_c[:,1:]
                    obj_d_embnbr_c = np.linalg.norm(O_c[:,None,:]-O_c[idx_emb_nn_c], axis=2)
                    mask_emb_c = dist_emb_nn_c > min_sep
                    ratios_e2o_c = np.full_like(obj_d_embnbr_c, np.nan, dtype=np.float64)
                    ratios_e2o_c[mask_emb_c] = obj_d_embnbr_c[mask_emb_c]/(dist_emb_nn_c[mask_emb_c]+eps)
                    med_e2o_c = np.nanmedian(ratios_e2o_c, axis=1)
                    # save hists
                    def save_hist_c(data, name):
                        try:
                            d = data[np.isfinite(data)]
                            plt.figure(figsize=(5,3.2))
                            sns.histplot(d, bins=40, stat='density', color='C1', alpha=0.8)
                            if d.size and (np.max(d) / max(np.median(d), 1e-9) > 1e3):
                                plt.xscale('log')
                            outp = os.path.join(image_dir_centroids, f'{name}_embdim{E.shape[1]}.pdf')
                            plt.tight_layout(); plt.savefig(outp); plt.close()
                            return outp
                        except Exception:
                            return None
                    lip_c_o2e_pdf = save_hist_c(med_o2e_c, 'centroids_lip_local_med_obj2emb')
                    lip_c_e2o_pdf = save_hist_c(med_e2o_c, 'centroids_lip_local_med_emb2obj')
                    lip_centroid = {
                        'obj2emb_med': {
                            'median': float(np.nanmedian(med_o2e_c)) if np.isfinite(med_o2e_c).any() else float('nan'),
                            'p90': float(np.nanpercentile(med_o2e_c,90)) if np.isfinite(med_o2e_c).any() else float('nan'),
                        },
                        'emb2obj_med': {
                            'median': float(np.nanmedian(med_e2o_c)) if np.isfinite(med_e2o_c).any() else float('nan'),
                            'p90': float(np.nanpercentile(med_e2o_c,90)) if np.isfinite(med_e2o_c).any() else float('nan'),
                        },
                        'obj2emb_med_pdf': lip_c_o2e_pdf,
                        'emb2obj_med_pdf': lip_c_e2o_pdf,
                        'k_lip_c': k_c,
                        'counts': counts_centroids,
                    }
                else:
                    lip_centroid = {}
            except Exception:
                lip_centroid = {}
        except Exception:
            rho_c, pval_c, mantel_r_c, mantel_p_c, mantel_hist_path_c, lip_centroid = float('nan'), float('nan'), float('nan'), float('nan'), None, {}

        # Print compact report
        print(f'\nEmbedding diagnostics summary for emb_dim {E.shape[1]}:')
        print('  PCA cumulative explained:', cum[:8])
        print(f'  Participation ratio (effective dim) = {participation_ratio:.3f}')
        print(f'  MLE intrinsic dim (k={k_mle}) = {id_mle:.3f}')
        print(f'  Clustering (KMeans vs true): ARI={ari:.3f}, NMI={nmi:.3f}, Silhouette(true)={sil_true:.3f}')
        print(f'  Trustworthiness (obj -> emb) = {tw:.3f}')
        print(f'  Spearman(D_emb, D_obj) = {rho:.3f} (p~{pval})')
        print(f'  Mantel (two-sided) spearman r={mantel_r:.3f}, p={mantel_p:.3f}')
        print(f'  kNN overlap (k=5) mean={mean_ov:.3f}, quartiles={quartiles}; pareto-like kNN mean={pareto_knn_mean:.3f}')
        print(f'  Within/Between means: within={mean_within:.4f}, between={mean_between:.4f}')
        if lip_summary:
            print('  Lipschitz obj->emb median p90 frac>10:', lip_summary['obj2emb_med']['median'], lip_summary['obj2emb_med']['p90'], lip_summary['obj2emb_med']['frac_gt10'])
            print('  Lipschitz emb->obj median p90 frac>10:', lip_summary['emb2obj_med']['median'], lip_summary['emb2obj_med']['p90'], lip_summary['emb2obj_med']['frac_gt10'])
        # Centroid-level (policy-level, deduped) report
        print('  [Policy-level deduped] PCA cumulative explained:', cum_c[:8] if 'cum_c' in locals() else [])
        print(f'  [Policy-level deduped] Participation ratio = {participation_ratio_c:.3f}')
        print(f'  [Policy-level deduped] MLE intrinsic dim (k={k_mle_c}) = {id_mle_c:.3f}')
        print(f'  [Policy-level deduped] Trustworthiness (obj -> emb) = {tw_c:.3f}')
        print(f'  [Policy-level deduped] Spearman(D_emb, D_obj) = {rho_c:.3f} (p~{pval_c})')
        print(f'  [Policy-level deduped] Mantel r={mantel_r_c:.3f}, p={mantel_p_c:.3f}')
        print(f'  [Policy-level deduped] kNN overlap mean={mean_ov_c:.3f}, quartiles={quartiles_c}; pareto-like kNN mean={pareto_knn_mean_c:.3f}')
        if args.DeepSeaTreasure:
            print(f'  [Policy-level deduped] Within/Between means: within={mean_within_c:.4f}, between={mean_between_c:.4f}')
        if lip_centroid:
            print('  [Policy-level deduped] Lipschitz obj->emb median,p90:', lip_centroid['obj2emb_med']['median'], lip_centroid['obj2emb_med']['p90'])
            print('  [Policy-level deduped] Lipschitz emb->obj median,p90:', lip_centroid['emb2obj_med']['median'], lip_centroid['emb2obj_med']['p90'])

        # Save to results for later CSV if desired
        res_entry = {
            'emb_dim': E.shape[1],
            'participation_ratio': float(participation_ratio),
            'id_mle': float(id_mle),
            'ari': float(ari),
            'nmi': float(nmi),
            'silhouette_true': float(sil_true),
            'trustworthiness': float(tw),
            'spearman_dist': float(rho),
            'spearman_p': float(pval),
            'mantel_r': float(mantel_r),
            'mantel_p': float(mantel_p),
            'mantel_hist_path': mantel_hist_path,
            'knn_overlap_mean': float(mean_ov),
            'pareto_knn_mean': float(pareto_knn_mean),
            'mean_within': mean_within,
            'mean_between': mean_between
        }
        # incorporate lipschitz stats if available
        if lip_summary:
            for kcat, stats_dict in lip_summary.items():
                if isinstance(stats_dict, dict) and 'median' in stats_dict:
                    for sk, val in stats_dict.items():
                        if isinstance(val, (int,float)):
                            res_entry[f'lip_{kcat}_{sk}'] = val
            # add pdf paths
            for key in ['obj2emb_med_pdf','obj2emb_max_pdf','emb2obj_med_pdf','emb2obj_max_pdf','k_lip']:
                if key in lip_summary:
                    res_entry[f'lip_{key}'] = lip_summary[key]
        # add centroid-level metrics
        res_entry.update({
            'centroid_spearman': float(rho_c) if rho_c is not None else float('nan'),
            'centroid_spearman_p': float(pval_c) if pval_c is not None else float('nan'),
            'centroid_mantel_r': float(mantel_r_c) if mantel_r_c is not None else float('nan'),
            'centroid_mantel_p': float(mantel_p_c) if mantel_p_c is not None else float('nan'),
            'centroid_mantel_hist_path': mantel_hist_path_c,
            'centroid_participation_ratio': float(participation_ratio_c) if 'participation_ratio_c' in locals() else float('nan'),
            'centroid_id_mle': float(id_mle_c) if 'id_mle_c' in locals() else float('nan'),
            'centroid_trustworthiness': float(tw_c) if 'tw_c' in locals() else float('nan'),
            'centroid_knn_overlap_mean': float(mean_ov_c) if 'mean_ov_c' in locals() else float('nan'),
            'centroid_pareto_knn_mean': float(pareto_knn_mean_c) if 'pareto_knn_mean_c' in locals() else float('nan'),
            'centroid_within': mean_within_c if 'mean_within_c' in locals() else float('nan'),
            'centroid_between': mean_between_c if 'mean_between_c' in locals() else float('nan'),
        })
        if lip_centroid:
            for subk, stats in lip_centroid.items():
                if isinstance(stats, dict) and 'median' in stats:
                    for sk, val in stats.items():
                        if isinstance(val, (int,float)):
                            res_entry[f'centroid_lip_{subk}_{sk}'] = val
            for key in ['obj2emb_med_pdf','emb2obj_med_pdf','k_lip_c']:
                if key in lip_centroid:
                    res_entry[f'centroid_lip_{key}'] = lip_centroid[key]
        results.append(res_entry)

        if saving:
            final_df = pd.DataFrame(results)
            final_df.to_csv(csv_file_path, index=False)
            print(f"\nAll experiments done. Final results saved to {csv_file_path}")

if __name__ == "__main__":
    main()