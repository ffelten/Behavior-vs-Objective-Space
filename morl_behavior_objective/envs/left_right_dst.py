"""This script plots the Pareto frontier for the Left-Right DST environment."""

import json

import matplotlib.pyplot as plt
import mo_gymnasium as mo_gym
import numpy as np
import numpy.typing as npt

import morl_behavior_objective  # noqa: F401

LEFT = 2
RIGHT = 3
DOWN = 1
UP = 0

OPTIMAL_POLICIES_LEFT_RIGHT = {
    "7_left": 4 * [LEFT] + 5 * [DOWN],  # 9 moves
    "6_right": 3 * [RIGHT] + 5 * [DOWN],  # 8 moves
    "25_left": 7 * [LEFT] + 8 * [DOWN],  # 15 moves
    "24_right": 6 * [RIGHT] + 8 * [DOWN],  # 14 moves
    "120_left": 9 * [LEFT] + 9 * [DOWN],  # 18 moves
    "124_right": 9 * [RIGHT] + 10 * [DOWN],  # 19 moves
}


OPTIMAL_POLICIES_LOUVRE_DST = {
    "7_left": 4 * [LEFT] + 5 * [UP],  # 9 moves
    "6_right": 3 * [RIGHT] + 5 * [DOWN],  # 8 moves
    "25_left": 7 * [LEFT] + 8 * [UP],  # 15 moves
    "24_right": 6 * [RIGHT] + 8 * [DOWN],  # 14 moves
    "120_left": 9 * [LEFT] + 9 * [UP],  # 18 moves
    "124_right": 9 * [RIGHT] + 10 * [DOWN],  # 19 moves
}

OPTIMAL_POLICIES_DST_CONCAVE = {
    "1": 1 * [DOWN],  # 1 move
    "2": 1 * [RIGHT] + 2 * [DOWN],  # 3 moves
    "3": 2 * [RIGHT] + 3 * [DOWN],  # 5 moves
    "5": 3 * [RIGHT] + 4 * [DOWN],  # 7 moves
    "8": 4 * [RIGHT] + 4 * [DOWN],  # 8 moves
    "16": 5 * [RIGHT] + 4 * [DOWN],  # 9 moves
    "24": 6 * [RIGHT] + 7 * [DOWN],  # 13 moves
    "50": 7 * [RIGHT] + 7 * [DOWN],  # 14 moves
    "74": 8 * [RIGHT] + 9 * [DOWN],  # 17 moves
    "124": 9 * [RIGHT] + 10 * [DOWN],  # 19 moves
}

OPTIMAL_POLICIES_SMOOTH = {
    "1": 1 * [DOWN],  # 1 move
    "3": 1 * [RIGHT] + 2 * [DOWN],  # 3 moves
    "5": 2 * [RIGHT] + 3 * [DOWN],  # 5 moves
    "7": 3 * [RIGHT] + 4 * [DOWN],  # 7 moves
    "9": 4 * [RIGHT] + 5 * [DOWN],  # 9 moves
    "11": 5 * [RIGHT] + 6 * [DOWN],  # 11 moves
    "13": 6 * [RIGHT] + 7 * [DOWN],  # 13 moves
    "15": 7 * [RIGHT] + 8 * [DOWN],  # 15 moves
    "17": 8 * [RIGHT] + 9 * [DOWN],  # 17 moves
    "19": 9 * [RIGHT] + 10 * [DOWN],  # 19 moves
}

EPISODES_PER_POLICY = 50


# env = mo_gym.make("left-right-dst-v0", render_mode=None)
# env = mo_gym.make("deep-sea-treasure-louvre-dst-v0", render_mode="human")
env = mo_gym.make("deep-sea-treasure-smooth-v0", render_mode="human")
# env = mo_gym.make("deep-sea-treasure-v0", render_mode="human", dst_map=CONCAVE_MAP)

policy_disc_returns: dict[str, npt.NDArray] = {}

for policy_name, policy in OPTIMAL_POLICIES_SMOOTH.items():
    env.reset()
    done = False
    disc_return = np.array([0.0, 0.0])
    discount = 1.0
    import time

    time.sleep(10)
    i = 0
    print("Executing policy: ", policy_name)
    while not done:
        action = policy[i]
        print(f" Action: {action}")
        i += 1
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        disc_return += discount * reward
        discount *= 1.0

    policy_disc_returns[policy_name] = disc_return

print(policy_disc_returns)


def collect_trajectory(policy: list[int]) -> tuple[list, list, npt.NDArray]:
    env.reset()
    done = False
    disc_return = np.array([0.0, 0.0])
    discount = 1.0
    i = 0
    states = []
    actions = []
    while not done:
        action = policy[i]
        i += 1
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        disc_return += discount * reward
        discount *= 1.0
        states.append(obs.tolist())
        actions.append([action])  # convert action to vector
    return states, actions, disc_return


for i in range(len(OPTIMAL_POLICIES_LOUVRE_DST)):
    POLICY_ID = i
    policy = list(OPTIMAL_POLICIES_LOUVRE_DST.values())[POLICY_ID]
    states, actions, disc_return = collect_trajectory(policy)
    with open(f"trajectories/louvres_dst/policy_{POLICY_ID}.json", "w") as f:
        json.dump(
            {"return": disc_return.tolist(), "trajectories": [[states, actions]] * EPISODES_PER_POLICY},
            f,
            indent=2,
            allow_nan=False,
        )
        f.close()


# Extract x and y coordinates from policy_disc_returns
x_coords = [returns[0] for returns in policy_disc_returns.values()]
y_coords = [returns[1] for returns in policy_disc_returns.values()]
policy_names = list(policy_disc_returns.keys())

# Create scatter plot with blue for left policies and red for right policies
plt.figure(figsize=(10, 8))
scatter_plots = []

for name, returns in policy_disc_returns.items():
    color = "blue" if "left" in name else "red"
    scatter = plt.scatter(returns[1], returns[0], s=100, alpha=0.7, color=color, label=name)
    scatter_plots.append(scatter)

    plt.annotate(name, (returns[0], returns[1]), xytext=(5, 5), textcoords="offset points", fontsize=10, fontweight="bold")

plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.xlabel("Time")
plt.ylabel("Treasure Value")
plt.title("Pareto Frontier")
plt.grid(visible=True, alpha=0.3)

# Adjust layout to prevent legend cutoff
plt.tight_layout()

plt.show()
