"""Roll out GPI (discrete or continuous; auto-detected) or MORLD agents and save (state, action) pairs per episode."""

import json
import os

from gymnasium.wrappers import FlattenObservation
from gymnasium.wrappers import TimeLimit
import mo_gymnasium as mo_gym
from morl_baselines.multi_policy.morld.morld import MORLD
import numpy as np

# ===================== CONFIG =====================
MODE = "morld"  # "gpi" or "morld"

# Common
ENV_ID = "mo-highway-fast-v0"  # works for both discrete/continuous, we detect action space at runtime
GAMMA = 0.99
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# MORLD settings
LOAD_MORLD_REPLAY = False
EPISODES_PER_POLICY = 50
# =======================================================


# -------------------- JSON helper --------------------
def make_json_safe(obj):
    """Recursively convert objects to JSON-serializable types.

    Conversions:
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
        except Exception:  # noqa: BLE001 -- best-effort serialization; fall back to the raw object
            return obj
    return obj


# --------------------- helpers --------------------


def _rollout_morld_policy(policy, env, episodes: int, seed: int | None = None):
    """Run episodes for a single MORLD policy (policy.wrapped).

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

            episode_states.append(obs)
            episode_actions.append(action)

            obs, _reward, terminated, truncated, _info = env.step(action)

        trajectory_i = [episode_states, episode_actions]
        all_trajectories.append(trajectory_i)

    return all_trajectories


# -------------------- runners --------------------


def run_morld():
    """Train a MORL/D agent on the configured environment and save its policy trajectories."""
    os.makedirs(MORLD_OUTPUT_DIR, exist_ok=True)

    env = mo_gym.make(ENV_ID)
    eval_env = mo_gym.make(ENV_ID)

    if "mo-halfcheetah" in ENV_ID or "mo-hopper" in ENV_ID:
        env = TimeLimit(env, max_episode_steps=100)
        eval_env = TimeLimit(eval_env, max_episode_steps=100)
    if "mo-highway-fast-v0" in ENV_ID:
        env = FlattenObservation(env)
        eval_env = FlattenObservation(eval_env)

    agent = MORLD(env=eval_env, gamma=params["gamma"], log=False, seed=params["seed"], **params["init_hyperparams"])
    agent.load(MORLD_CHECKPOINT, load_replay_buffer=LOAD_MORLD_REPLAY)

    if not hasattr(agent, "archive") or len(agent.archive.individuals) == 0:
        raise RuntimeError("MORLD pareto archive is empty after load().")

    for pid, pol in enumerate(agent.archive.individuals):
        pweights = getattr(pol, "weights", None)
        print(f"[MORLD] {ENV_ID}: running {EPISODES_PER_POLICY} episodes for policy {pid} with weights {pweights}")

        all_trajectories = _rollout_morld_policy(pol, eval_env, EPISODES_PER_POLICY, seed=seed)

        return_vec = agent.archive.evaluations[pid]

        json_data = {
            "return": return_vec,
            "trajectories": all_trajectories,
        }

        out_json = os.path.join(MORLD_OUTPUT_DIR, f"policy_{pid}.json")
        with open(out_json, "w") as f:
            json.dump(make_json_safe(json_data), f, indent=2, allow_nan=False)
        print(f"  -> wrote JSON with return {return_vec} to {out_json}")


def reder_policy(policy_id: int):
    """Renders a single MORLD policy by its ID."""
    env = mo_gym.make(ENV_ID, render_mode="human")
    if "mo-halfcheetah" in ENV_ID or "mo-hopper" in ENV_ID:
        env = TimeLimit(env, max_episode_steps=100)
    if "mo-highway-fast-v0" in ENV_ID:
        env = FlattenObservation(env)

    agent = MORLD(env=env, gamma=params["gamma"], log=False, seed=params["seed"], **params["init_hyperparams"])
    agent.load(MORLD_CHECKPOINT, load_replay_buffer=LOAD_MORLD_REPLAY)

    if not hasattr(agent, "archive") or len(agent.archive.individuals) == 0:
        raise RuntimeError("MORLD pareto archive is empty after load().")

    pol = agent.archive.individuals[policy_id]
    pweights = getattr(pol, "weights", None)
    print(f"[MORLD] {ENV_ID}: rendering policy {policy_id} with weights {pweights}")

    obs, _ = env.reset()
    terminated = truncated = False

    while not (terminated or truncated):
        env.render()
        try:
            action = pol.wrapped.eval(obs, None)
        except TypeError:
            action = pol.wrapped.eval(obs)

        obs, _reward, terminated, truncated, _info = env.step(action)

    env.close()


# -------------------- main --------------------

if __name__ == "__main__":
    if MODE == "morld":
        for seed in range(5):
            MORLD_CHECKPOINT = os.path.join(SCRIPT_DIR, "MORL_policies", f"{ENV_ID}", f"seed{seed}.tar")
            MORLD_OUTPUT_DIR = os.path.join("trajectories", "morld", f"{ENV_ID}", f"seed{seed}")
            params = {
                "algo": "morld",
                "env_id": ENV_ID,  # "mo-halfcheetah-v5", #mo-highway-fast-v0
                "num_timesteps": 1_000_000,
                "gamma": 0.99,
                "ref_point": [-1, -1, -40],  # [-100, -100],
                "seed": seed,
                "wandb_entity": "florian-felten",
                "init_hyperparams": {
                    "scalarization_method": "ws",
                    "evaluation_mode": "ser",
                    "policy_name": "MOSACDiscrete",  # "MOSAC", "MOSACDiscrete"
                    "shared_buffer": False,
                    "weight_adaptation_method": None,
                    "exchange_every": 10_000,
                },
                "train_hyperparams": {},
                "save_dic": "weights",
            }
            run_morld()
    else:
        raise ValueError("MODE must be 'morld'.")
