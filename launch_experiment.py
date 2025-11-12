import argparse
import os

from gymnasium.wrappers import FlattenObservation
from gymnasium.wrappers import TimeLimit
import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers import MORecordEpisodeStatistics
from morl_baselines.common.evaluation import seed_everything
from morl_baselines.multi_policy.morld.morld import MORLD
import numpy as np

params = {
    "algo": "morld",
    "env_id": "mo-hopper-2obj-v5",  # "mo-highway-fast-v0",  # "mo-halfcheetah-v5",
    "num_timesteps": 1_000_000,
    "gamma": 0.99,
    "ref_point": [-100, -100],  # [-1, -1, -40],  # [-100, -100],
    "seed": 0,
    "wandb_entity": "florian-felten",
    "init_hyperparams": {
        "scalarization_method": "ws",
        "evaluation_mode": "ser",
        "policy_name": "MOSAC",  # "MOSACDiscrete",  # "MOSAC",
        "shared_buffer": False,
        "weight_adaptation_method": None,
        "exchange_every": 10_000,
    },
    "train_hyperparams": {},
    "save_dic": "weights",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch MORL experiment")
    parser.add_argument("--seed", type=int, required=True, help="Random seed (default: 0)")
    args = parser.parse_args()
    # Update seed from command line argument
    params["seed"] = args.seed

    print("Launching experiment with parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    seed_everything(params["seed"])

    # ---- ENVIRONMENT ----
    if "mo-halfcheetah" in params["env_id"].lower() or "mo-hopper" in params["env_id"].lower():
        env = mo_gym.make(params["env_id"])
        env = TimeLimit(env, max_episode_steps=100)
        eval_env = mo_gym.make(
            params["env_id"],
            render_mode=None,
        )
        eval_env = TimeLimit(eval_env, max_episode_steps=100)
    elif "highway" in params["env_id"].lower():
        env = mo_gym.make(params["env_id"])
        eval_env = mo_gym.make(params["env_id"], render_mode=None)
        env = FlattenObservation(env)
        eval_env = FlattenObservation(eval_env)
    else:
        raise ValueError(
            f"Environment {params['env_id']} not supported, only 'mo-halfcheetah', 'mo-hopper' and 'highway' are supported"
        )
    env = MORecordEpisodeStatistics(env, gamma=params["gamma"])

    print(f"Instantiating {params['algo']} on {params['env_id']}")

    algo = MORLD(
        env=env,
        gamma=params["gamma"],
        log=True,
        seed=params["seed"],
        wandb_entity=params["wandb_entity"],
        **params["init_hyperparams"],
    )

    print(algo.get_config())

    print("Training starts... Let's roll!")
    algo.train(
        total_timesteps=params["num_timesteps"],
        eval_env=eval_env,
        ref_point=np.array(params["ref_point"]),
        known_pareto_front=None,  # You can add lookup logic here if needed
        **params["train_hyperparams"],
    )
    # create directory if it doesn't exist
    save_dir = str(params["save_dic"]) + f"-{params['env_id']}" + f"-{params['seed']}/"
    os.makedirs(save_dir, exist_ok=True)
    algo.save(save_dir=save_dir, save_replay_buffer=False)
