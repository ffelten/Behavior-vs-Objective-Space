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

from methods.behaviorencoder import *
from methods.behaviorencoder_utils import *


## MAIN UTILS for normalization and Controller HDBSCAN clustering

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
    arg_env.add_argument("-T2D","--Traj2d", help="Apply the selected algorithm to the Traj2d environment",action="store_true")
    arg_env.add_argument("-Rv4","--Reacherv4", help="Apply the selected algorithm to the Reacher-v4 environment",action="store_true")
    arg_env.add_argument("-Pv4","--Pusherv4", help="Apply the selected algorithm to the Pusher-v4 environment",action="store_true")
    arg_env.add_argument("-Hv4","--Humanoidv4", help="Apply the selected algorithm to the Humanoid-v4 environment",action="store_true")
    arg_env.add_argument("-W2D","--Walker2dv4", help="Apply the selected algorithm to the Walker2d-v4 environment",action="store_true")

    arg_alg = parser.add_argument_group('Unseen Split Selection')
    arg_alg.add_argument("-split","--use_seen_unseen_split", action="store_true", 
                    help="Use seen-unseen split for training. If not set, train on the entire dataset.")
    arg_alg.add_argument("-nUnModes","--num_unseen_modes", type=int, default=1, help="Number of unseen modes in the unseen split for the dataset.")

    arg_irl = parser.add_argument_group('IRL Selection') #not really needed here
    arg_irl.add_argument("-gail","--GAIL", help="Run GAIL IRL",action="store_true")
    arg_irl.add_argument("-sqil","--SQIL", help="Run SQIL IRL",action="store_true")
    arg_irl.add_argument("-airl","--AIRL", help="Run AIRL IRL",action="store_true")

    arg_hyp = parser.add_argument_group('Settings') 
    arg_hyp.add_argument('-nT','--num_trajs', type=int,default=100,help='int: Number of expert trajectories to generate')
    arg_hyp.add_argument('-seed','--seed', type=int,default=0,help='int: Random seed for reproducibility') 
    arg_hyp.add_argument('--ratio', type=int, default=1,help="Ratio for splitting trajectories between modes. 1 uniform 3 first gets most last gets least, etc.")
    arg_hyp.add_argument('--embedding_strategy', type=str, default='cls_only', choices=['cls_only', 'hybrid','goal_oriented'], help='Strategy for creating the final trajectory embedding for clustering.')

    arg_vis = parser.add_argument_group('Visualization')
    arg_vis.add_argument('-vA','--visualize_attention', help='Visualize the interaction and aggregated attention',action="store_true")
    arg_vis.add_argument('-vS','--visualize_scalers', help='Visualize the effect of different scalers on the state space',action="store_true")
    arg_vis.add_argument('-vO','--visualize_original', help='Visualize the original state space',action="store_true")
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
    visualize_original = args.visualize_original
    visualize_scalers = args.visualize_scalers
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

    print("\n--- Generating Visualizations and Metrics ---")
    # This visualization block is the same as the one you use for your other models.
    check_unseen = args.use_seen_unseen_split
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

    if controller:
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

    print("*******************************************************************************************")
    # exit("Exiting after Traj2d clustering and IRL training.")

    if saving:
        final_df = pd.DataFrame(results)
        final_df.to_csv(csv_file_path, index=False)
        print(f"\nAll experiments done. Final results saved to {csv_file_path}")

if __name__ == "__main__":
    main()