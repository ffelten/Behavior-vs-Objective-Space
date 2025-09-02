'''Roll out GPI (discrete or continuous; auto-detected) or MORLD agents and save (state, action) pairs per episode.'''

import os
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
MODE = "gpi"  # "gpi" or "morld"

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
MORLD_CHECKPOINT = os.path.join(SCRIPT_DIR, "..", "trained_policies", "multi_policy", "MORLD_checkpoint.tar")
LOAD_MORLD_REPLAY = False
EPISODES_PER_POLICY = 50
MORLD_OUTPUT_DIR = os.path.join("trajectories", "morld", ENV_ID)
# =======================================================


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
    """Return a tuple (flat) for stable pickling/strings."""
    if isinstance(obs, np.ndarray):
        return tuple(obs.ravel().tolist())
    if isinstance(obs, dict):
        items = []
        for k in sorted(obs.keys()):
            v = obs[k]
            if isinstance(v, np.ndarray):
                items.extend(v.ravel().tolist())
            else:
                try:
                    items.append(float(v))
                except Exception:
                    pass
        return tuple(items)
    if isinstance(obs, (list, tuple)):
        return tuple(np.array(obs, dtype=float).ravel().tolist())
    try:
        return (float(obs),)
    except Exception:
        return tuple()


def _action_to_serializable(action):
    """Convert action to simple serializable types (int/float/tuple)."""
    if isinstance(action, (int, np.integer)):
        return int(action)
    if isinstance(action, (float, np.floating)):
        return float(action)
    if isinstance(action, np.ndarray):
        if action.ndim == 0:
            return float(action.item())
        return tuple(action.ravel().tolist())
    if isinstance(action, (list, tuple)):
        return tuple(np.array(action, dtype=float).ravel().tolist())
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
    Returns: list[ list[(state_tuple, action)] ] — one inner list per episode.
    """
    if seed is not None:
        env.reset(seed=seed)

    episodes_data = []
    for _ in range(episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        step_pairs = []
        while not (terminated or truncated):
            action = agent.eval(obs, w)  # discrete -> int; continuous -> np array
            s_flat = _flatten_obs(obs)
            a_ser = _action_to_serializable(action)
            step_pairs.append((s_flat, a_ser))
            obs, reward, terminated, truncated, info = env.step(action)
        episodes_data.append(step_pairs)
    return episodes_data


def _rollout_morld_policy(policy, env, episodes: int, seed: int | None = None):
    """
    Run episodes for a single MORLD policy (policy.wrapped).
    Returns: list[ list[(state_tuple, action)] ]
    """
    if seed is not None:
        env.reset(seed=seed)

    episodes_data = []
    for _ in range(episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        step_pairs = []
        while not (terminated or truncated):
            try:
                action = policy.wrapped.eval(obs, None)
            except TypeError:
                action = policy.wrapped.eval(obs)
            s_flat = _flatten_obs(obs)
            a_ser = _action_to_serializable(action)
            step_pairs.append((s_flat, a_ser))
            obs, reward, terminated, truncated, info = env.step(action)
        episodes_data.append(step_pairs)
    return episodes_data


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

    # Robust loader for discrete/continuous + GPILS/GPIPD
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

    # Treat each Pareto weight as a “policy” and roll out
    for idx, w in enumerate(pareto_weights):
        w = np.asarray(w, dtype=float)
        print(f"[GPI] {ENV_ID}: running {EPISODES_PER_WEIGHT} episodes for Pareto weight #{idx}: {w}")
        episodes_data = _rollout_gpi_with_weight(agent, eval_env, w, EPISODES_PER_WEIGHT, seed=SEED)
        out_pkl = os.path.join(GPI_OUTPUT_DIR, f"policy_w{idx}.pkl")
        with open(out_pkl, "wb") as f:
            pickle.dump(episodes_data, f)
        print(f"  -> wrote {len(episodes_data)} episodes to {out_pkl}")


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

    if not hasattr(agent, "population") or len(agent.population) == 0:
        raise RuntimeError("MORLD population is empty after load().")

    for pol in agent.population:
        pid = getattr(pol, "id", None)
        pweights = getattr(pol, "weights", None)
        print(f"[MORLD] {ENV_ID}: running {EPISODES_PER_POLICY} episodes for policy {pid} with weights {pweights}")
        episodes_data = _rollout_morld_policy(pol, eval_env, EPISODES_PER_POLICY, seed=SEED)
        out_pkl = os.path.join(MORLD_OUTPUT_DIR, f"policy_{pid}.pkl")
        with open(out_pkl, "wb") as f:
            pickle.dump(episodes_data, f)
        print(f"  -> wrote {len(episodes_data)} episodes to {out_pkl}")


# -------------------- main --------------------

if __name__ == "__main__":
    if MODE == "gpi":
        run_gpi()
    elif MODE == "morld":
        run_morld()
    else:
        raise ValueError("MODE must be 'gpi' or 'morld'.")
