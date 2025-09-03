'''Roll out GPI (discrete or continuous; auto-detected) or MORLD agents and save (state, action) pairs per episode.'''

import os
import json
import pickle
import numpy as np
import mo_gymnasium as mo_gym
import torch as th  # for checkpoint inspection/manual load

from gymnasium.wrappers import FlattenObservation

from morl_baselines.common.weights import equally_spaced_weights
from morl_baselines.common.pareto import filter_pareto_dominated
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPIPD, GPILS
from morl_baselines.multi_policy.morld.morld import MORLD
from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import (
    GPILSContinuousAction,
    GPIPDContinuousAction,
)

# ===================== CONFIG =====================
MODE = "morld"  # "gpi" or "morld"

# Common
ENV_ID = "mo-halfcheetah-v4"   # works for both discrete/continuous, we detect action space at runtime
GAMMA = 0.99
SEED = 0
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# GPI settings
GPI_CHECKPOINT = os.path.join(SCRIPT_DIR, "..", "trained_policies", "conditional_policy", "GPI_cheetah.tar")
NUM_SAMPLE_WEIGHTS = 100
NUM_EVAL_EPISODES_FOR_FRONT = 10
EPISODES_PER_WEIGHT = 50
GPI_OUTPUT_DIR = os.path.join("trajectories", "gpi", ENV_ID)
SAVE_GPI_PICKLES = True  # save whole_front, pareto_front, pareto_weights

# MORLD settings
MORLD_CHECKPOINT = os.path.join(SCRIPT_DIR, "..", "trained_policies", "multi_policy", "morld_cheetah.tar")
LOAD_MORLD_REPLAY = False
EPISODES_PER_POLICY = 50
MORLD_OUTPUT_DIR = os.path.join("trajectories", "morld", ENV_ID)
# =======================================================

# -------------------- JSON helper (NEW) --------------------
def make_json_safe(obj):
    """
    Recursively convert objects to JSON-serializable types:
    - np.ndarray -> list
    - np.* scalars -> native Python scalars
    - tuples/sets -> lists
    - bytes -> utf-8 string (with replacement)
    - dict keys -> strings
    Also replaces NaN/Inf with None to keep strict JSON valid.
    """
    # numpy arrays
    if isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist())
    # numpy scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        if not np.isfinite(x):
            return None
        return x
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    # containers
    if isinstance(obj, dict):
        return {str(make_json_safe(k)): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]
    # bytes
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    # has tolist (fallback)
    if hasattr(obj, "tolist"):
        try:
            return make_json_safe(obj.tolist())
        except Exception:
            pass
    return obj

# -------------------- helpers --------------------

def _attach_reward_space(env):
    """
    Ensure env.reward_space exists on the top-level wrapper.
    Some library calls access env.reward_space directly (not .unwrapped).
    """
    base = getattr(env, "unwrapped", env)
    if hasattr(base, "reward_space"):
        setattr(env, "reward_space", base.reward_space)
        return env
    cur = getattr(env, "env", None)
    while cur is not None and not hasattr(cur, "reward_space"):
        cur = getattr(cur, "env", None)
    if cur is not None and hasattr(cur, "reward_space"):
        setattr(env, "reward_space", cur.reward_space)
        return env
    raise AttributeError("Could not find reward_space on the underlying environment.")

def _flatten_obs(obs):
    """Return a flat list (JSON-friendly)."""
    if isinstance(obs, np.ndarray):
        return list(np.asarray(obs).ravel().tolist())
    if isinstance(obs, dict):
        items = []
        for k in sorted(obs.keys()):
            v = obs[k]
            if isinstance(v, np.ndarray):
                items.extend(v.ravel().tolist())
            elif isinstance(v, (list, tuple)):
                items.extend(np.asarray(v, dtype=float).ravel().tolist())
            else:
                try:
                    items.append(float(v))
                except Exception:
                    pass
        return items
    if isinstance(obs, (list, tuple)):
        return list(np.asarray(obs, dtype=float).ravel().tolist())
    try:
        return [float(obs)]
    except Exception:
        return []

def _action_to_serializable(action):
    """Convert action to simple JSON types (int/float/list)."""
    if isinstance(action, (int, np.integer)):
        return int(action)
    if isinstance(action, (float, np.floating)):
        return float(action)
    if isinstance(action, np.ndarray):
        if action.ndim == 0:
            return float(action.item())
        return list(action.ravel().tolist())
    if isinstance(action, (list, tuple)):
        return list(np.asarray(action, dtype=float).ravel().tolist())
    return action

def _produce_whole_front(weights, agent, env, num_eval_episodes: int, discounted_vec_return: bool = False):
    """
    Evaluate each weight; return list of objective vectors.
    Assumes agent.policy_eval(env, weights=..., num_episodes=...) returns:
      idx 3 -> vector return, idx 4 -> discounted vector return
    """
    if discounted_vec_return:
        return [agent.policy_eval(env, weights=ew, num_episodes=num_eval_episodes)[4] for ew in weights]
    else:
        return [agent.policy_eval(env, weights=ew, num_episodes=num_eval_episodes)[3] for ew in weights]

def _get_pareto_front(whole_front, weights):
    """Pareto-filter and return (pareto_front, pareto_weights)."""
    pareto_front = list(filter_pareto_dominated(whole_front))
    rows = [[(cur == p).all() for cur in whole_front].index(True) for p in pareto_front]
    pareto_weights = [weights[w] for w in rows]
    return pareto_front, pareto_weights

def _rollout_gpi_with_weight(agent, env, w: np.ndarray, episodes: int, seed: int | None = None):
    """
    Run episodes for GPI (discrete or continuous class) with fixed weight vector w.
    Returns: tuple of (states, actions) where each is a list of lists (episodes x steps)
    """
    if seed is not None:
        env.reset(seed=seed)

    all_states = []
    all_actions = []

    for _ in range(episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        episode_states = []
        episode_actions = []

        while not (terminated or truncated):
            action = agent.eval(obs, w)  # discrete -> int; continuous -> np array
            s_flat = _flatten_obs(obs)
            a_ser = _action_to_serializable(action)

            episode_states.append(s_flat)
            episode_actions.append(a_ser)

            obs, reward, terminated, truncated, info = env.step(action)

        all_states.append(episode_states)
        all_actions.append(episode_actions)

    return all_states, all_actions

def _rollout_morld_policy(policy, env, episodes: int, seed: int | None = None):
    """
    Run episodes for a single MORLD policy (policy.wrapped).
    Returns: list of trajectories; each trajectory is [states, actions],
             where states/actions are lists of per-step lists/floats (JSON-friendly).
    """
    if seed is not None:
        env.reset(seed=seed)

    all_trajectories = []

    for _ in range(episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        episode_states = []
        episode_actions = []

        while not (terminated or truncated):
            try:
                action = policy.wrapped.eval(obs, None)
            except TypeError:
                action = policy.wrapped.eval(obs)
            s_flat = _flatten_obs(obs)
            a_ser = _action_to_serializable(action)

            episode_states.append(s_flat)
            episode_actions.append(a_ser)

            obs, reward, terminated, truncated, info = env.step(action)

        trajectory_i = [episode_states, episode_actions]
        all_trajectories.append(trajectory_i)

    return all_trajectories

# --------- GPI loader that handles discrete/continuous and GPILS/GPIPD ----------

def load_gpi_agent(env, checkpoint_path, **kwargs):
    """
    Auto-select the correct GPI class based on the env's action space:
      - Discrete -> GPILS or GPIPD (if psi nets present)
      - Continuous -> GPILSContinuousAction (or GPIPDContinuousAction if psi nets present)
    """
    params = th.load(checkpoint_path, map_location="cpu", weights_only=False)
    has_psi = any(k.startswith("psi_net_") for k in params.keys())
    has_q = any(k.startswith("q_net_") for k in params.keys())

    continuous = not hasattr(env.action_space, "n")  # Box -> continuous

    if continuous:
        if has_psi:
            agent = GPIPDContinuousAction(env, log=False, project_name="MORL-Baselines", experiment_name="GPIPD (cont.)", **kwargs)
        else:
            agent = GPILSContinuousAction(env, log=False, project_name="MORL-Baselines", experiment_name="GPI-LS (cont.)", **kwargs)
        agent.load(path=checkpoint_path, load_replay_buffer=False)
        return agent

    # discrete
    if has_psi:
        agent = GPIPD(env, log=False, project_name="MORL-Baselines", experiment_name="GPIPD", **kwargs)
        agent.load(path=checkpoint_path, load_replay_buffer=False)
        return agent

    # GPILS (discrete)
    else:
        agent = GPILS(env, log=False, project_name="MORL-Baselines", experiment_name="GPI-LS", **kwargs)
        return agent

# -------------------- runners --------------------

def run_gpi():
    os.makedirs(GPI_OUTPUT_DIR, exist_ok=True)

    env = mo_gym.make(ENV_ID)
    eval_env = mo_gym.make(ENV_ID)

    # Optional: flatten obs for highway-like envs
    if "highway" in ENV_ID:
        env = FlattenObservation(env)
        eval_env = FlattenObservation(eval_env)

    # Expose reward_space at top-level so library code finds it through wrappers
    env = _attach_reward_space(env)
    eval_env = _attach_reward_space(eval_env)

    reward_dim = env.reward_space.shape[0]

    # Loader for discrete/continuous + GPILS/GPIPD
    agent = load_gpi_agent(env, GPI_CHECKPOINT)

    # Sample weights, evaluate, Pareto-filter
    weights = equally_spaced_weights(reward_dim, n=NUM_SAMPLE_WEIGHTS)
    whole_front = _produce_whole_front(weights, agent, eval_env, NUM_EVAL_EPISODES_FOR_FRONT, discounted_vec_return=False)
    pareto_front, pareto_weights = _get_pareto_front(whole_front, weights)

    # pareto stuff
    if SAVE_GPI_PICKLES:
        with open(os.path.join(GPI_OUTPUT_DIR, f'whole_front_{ENV_ID}.pkl'), "wb") as f:
            pickle.dump(whole_front, f)
        with open(os.path.join(GPI_OUTPUT_DIR, f'pareto_front_{ENV_ID}.pkl'), "wb") as f:
            pickle.dump(pareto_front, f)
        with open(os.path.join(GPI_OUTPUT_DIR, f'pareto_weights_{ENV_ID}.pkl'), "wb") as f:
            pickle.dump(pareto_weights, f)

    # Treat each Pareto weight as a "policy" and roll out
    for idx, (w, return_vec) in enumerate(zip(pareto_weights, pareto_front)):
        w = np.asarray(w, dtype=float)
        print(f"[GPI] {ENV_ID}: running {EPISODES_PER_WEIGHT} episodes for Pareto weight #{idx}: {w}")

        states, actions = _rollout_gpi_with_weight(agent, eval_env, w, EPISODES_PER_WEIGHT, seed=SEED)

        # JSON-safe data structure (NEW: no raw ndarrays)
        json_data = {
            "return": return_vec,
            "trajectories": [states, actions],
        }

        out_json = os.path.join(GPI_OUTPUT_DIR, f"policy_w{idx}.json")
        with open(out_json, "w") as f:
            json.dump(make_json_safe(json_data), f, indent=2, allow_nan=False)
        print(f"  -> wrote JSON with return {return_vec} to {out_json}")

def run_morld():
    os.makedirs(MORLD_OUTPUT_DIR, exist_ok=True)

    env = mo_gym.make(ENV_ID)
    eval_env = mo_gym.make(ENV_ID)

    if "highway" in ENV_ID:
        env = FlattenObservation(env)
        eval_env = FlattenObservation(eval_env)

    env = _attach_reward_space(env)
    eval_env = _attach_reward_space(eval_env)

    agent = MORLD(env=eval_env, gamma=GAMMA, log=False, seed=SEED)
    agent.load(MORLD_CHECKPOINT, load_replay_buffer=LOAD_MORLD_REPLAY)

    if not hasattr(agent, "archive") or len(agent.archive.individuals) == 0:
        raise RuntimeError("MORLD pareto archive is empty after load().")

    for pid, pol in enumerate(agent.archive.individuals):
        pweights = getattr(pol, "weights", None)
        print(f"[MORLD] {ENV_ID}: running {EPISODES_PER_POLICY} episodes for policy {pid} with weights {pweights}")

        all_trajectories = _rollout_morld_policy(pol, eval_env, EPISODES_PER_POLICY, seed=SEED)

        # Return vector from archive evaluations (could be ndarray) -> JSON-safe later
        return_vec = agent.archive.evaluations[pid]

        json_data = {
            "return": return_vec,
            "trajectories": all_trajectories,
        }

        out_json = os.path.join(MORLD_OUTPUT_DIR, f"policy_{pid}.json")
        with open(out_json, "w") as f:
            json.dump(make_json_safe(json_data), f, indent=2, allow_nan=False)
        print(f"  -> wrote JSON with return {return_vec} to {out_json}")

# -------------------- main --------------------

if __name__ == "__main__":
    if MODE == "gpi":
        run_gpi()
    elif MODE == "morld":
        run_morld()
    else:
        raise ValueError("MODE must be 'gpi' or 'morld'.")
