from pathlib import Path

import numpy as np

from morl_behavior_objective.utils import get_pareto_front
from morl_behavior_objective.utils import render_policy

RENDER = "cheetah"  # "hopper" #"cheetah"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
seed = 0

# Configuration for environments
environments = {
    "cheetah": {
        "trajectory_dir": f"{PROJECT_ROOT}/trajectories/morld/mo-halfcheetah-v5/seed0",
        "embedding_path": f"PROJECT_ROOT/trajectories/morld/mo-halfcheetah-v5/embeddings/mean_be_MHC_e100_seed{seed}.json",
    },
    "highway": {
        "trajectory_dir": f"{PROJECT_ROOT}/trajectories/morld/mo-highway-fast-v0/seed0",
        "embedding_path": f"{PROJECT_ROOT}/trajectories/morld/mo-highway-fast-v0/embeddings/mean_be_MHW_e100_seed{seed}.json",
    },
    "hopper_2d": {
        "trajectory_dir": f"{PROJECT_ROOT}/trajectories/morld/mo-hopper-2obj-v5/seed0",
        "embedding_path": f"{PROJECT_ROOT}/trajectories/morld/mo-hopper-2obj-v5/embeddings/mean_be_MHo2_e100_seed{seed}.json",
    },
    "hopper_3d": {
        "trajectory_dir": f"{PROJECT_ROOT}/trajectories/morld/mo-hopper-v5/seed0",
        "embedding_path": f"{PROJECT_ROOT}/trajectories/morld/mo-hopper-v5/embeddings/mean_be_MHo_e200_seed{seed}.json",
    },
}

returns_cheetah = get_pareto_front(environments["cheetah"]["trajectory_dir"])

returns_highway = get_pareto_front(environments["highway"]["trajectory_dir"])

returns_hopper_2d = get_pareto_front(environments["hopper_2d"]["trajectory_dir"])

<<<<<<< Updated upstream
=======
returns_hopper = get_pareto_front(environments["hopper_3d"]["trajectory_dir"])
>>>>>>> Stashed changes

####################### CHEETAH ########################

# Cheetah is a two-objective problem, so we use normal lextargsort
# Get policies with the  secondhighest Lipschitz constants

<<<<<<< Updated upstream
if RENDER == "cheetah":
    policies_list = [[45, 46], [22, 23], [56, 57], [7, 8]]
=======
if RENDER=="cheetah":
    policies_list=[[12,13], [63,64]]
>>>>>>> Stashed changes
    for policies in policies_list:
        # see what policies these correspond to in the original pareto front
        org_policy_id = np.lexsort(returns_cheetah.T)[policies]

        render_policy(
            env_id="mo-halfcheetah-v5",
            check_point="MORL_policies/morld_cheetah_v5/seed0.tar",
            policy_id=org_policy_id[0],
            n_episodes=5,
            save_dic=f"videos/half-cheetah/policies_{policies}",
            base_path=PROJECT_ROOT,
        )

        render_policy(
            env_id="mo-halfcheetah-v5",
            check_point="MORL_policies/morld_cheetah_v5/seed0.tar",
            policy_id=org_policy_id[1],
            n_episodes=5,
            save_dic=f"videos/half-cheetah/policies_{policies}",
            base_path=PROJECT_ROOT,
        )


<<<<<<< Updated upstream
####################### HIGHWAY ########################
# Highway has 3 objectives so we have to use a different ordering method
if RENDER == "highway":
    pareto_frontier = returns_highway
=======

####################### HOPPER ########################
#hopper has 3 objectives so we have to use a different ordering method
if RENDER=="hopper":
    pareto_frontier=returns_hopper
>>>>>>> Stashed changes
    pf_points = pareto_frontier.copy()
    n_points = len(pf_points)

    lowest_obj_idx = np.argmin(pf_points[:, 0])
    seq_idx = [lowest_obj_idx]  # start with point with lowest obj value in first dimension
    remaining_idx = list(range(1, n_points))

    while remaining_idx:
        last = pf_points[seq_idx[-1]]
        remaining = pf_points[remaining_idx]
        distances = np.linalg.norm(remaining - last, axis=1)
        nearest_idx = remaining_idx[np.argmin(distances)]
        seq_idx.append(nearest_idx)
        remaining_idx.remove(nearest_idx)

<<<<<<< Updated upstream
    policies_list = [[2, 3], [3, 4]]
=======


    policies_list=[[50,51],[52,53],[116,117],[83,84]]
>>>>>>> Stashed changes
    for policies in policies_list:
        # see what policies these correspond to in the original pareto front
        org_policy_id = np.array(seq_idx)[policies]

        render_policy(
            env_id="mo-hopper-v5",
            check_point="MORL_policies/mo-hopper-v5/seed0.tar",
            policy_id=org_policy_id[0],
            n_episodes=5,
            save_dic=f"videos/hopper/policies_{policies}",
            base_path=PROJECT_ROOT,
        )

        render_policy(
            env_id="mo-hopper-v5",
            check_point="MORL_policies/mo-hopper-v5/seed0.tar",
            policy_id=org_policy_id[1],
            n_episodes=5,
            save_dic=f"videos/hopper/policies_{policies}",
            base_path=PROJECT_ROOT,
        )
