import json
import os
from time import sleep

import matplotlib.pyplot as plt
import mo_gymnasium as mo_gym
import numpy as np
import numpy.typing as npt

env = mo_gym.make("resource-gathering-v0")

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

OPTIMAL_POLICIES = {
    "diamond_home": [RIGHT] * 2 + [UP] * 3 + [DOWN] * 3 + [LEFT] * 2,
    "both_resources_through_both_Es": [RIGHT] * 2 + [UP] * 4 + [LEFT] * 2 + [DOWN] * 4,
    "gold_through_E1_both_ways": [UP] * 4 + [DOWN] * 4,
    "both_resources_dodging_E1_through_E2": [RIGHT] * 2 + [UP] * 4 + [LEFT] * 3 + [DOWN] * 4 + [RIGHT],
    "gold_dodging_all_Es_in_12_steps": [LEFT] + [UP] * 4 + [RIGHT] + [LEFT] + [DOWN] * 4 + [RIGHT],
    "gold_through_E1_only_once": [UP] * 4 + [LEFT] + [DOWN] * 4 + [RIGHT],
    # "both_resources_dodging_both_Es_in_12_steps": [RIGHT] * 3
    # + [UP] * 3
    # + [LEFT]
    # + [DOWN]
    # + [LEFT] * 2
    # + [UP] * 2
    # + [RIGHT]
    # + [LEFT]
    # + [DOWN] * 4
    # + [RIGHT],
}


def collect_trajectory(policy: list[int]) -> tuple[list, list, npt.NDArray]:
    env.reset()
    done = False
    disc_return = np.array([0.0, 0.0, 0.0])
    discount = 1.0
    gamma = 0.9
    i = 0
    states = []
    actions = []
    while not done:
        action = policy[i]
        i += 1
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        disc_return += discount * reward
        discount *= gamma
        states.append(obs.tolist())
        actions.append([action])  # convert action to vector
    return states, actions, disc_return


pareto_frontier = {policy_name: np.zeros(3) for policy_name in OPTIMAL_POLICIES}

for policy_name, policy in OPTIMAL_POLICIES.items():
    print("Executing policy: ", policy_name)
    avg_disc_return = np.zeros(3)
    repeats = 100
    states = []
    actions = []
    for _ in range(repeats):
        states_i, actions_i, disc_return = collect_trajectory(policy)
        states.append(states_i)
        actions.append(actions_i)
        avg_disc_return += disc_return / repeats
    pareto_frontier[policy_name] = avg_disc_return
    print(f"Policy: {policy_name}, Average Return: {avg_disc_return}")

    os.makedirs("trajectories/resource_gathering", exist_ok=True)
    with open(f"trajectories/resource_gathering/{policy_name}.json", "w") as f:
        json.dump(
            {
                "return": avg_disc_return.tolist(),
                "trajectories": [[states_i, actions_i] for states_i, actions_i in zip(states, actions, strict=False)],
            },
            f,
            indent=2,
            allow_nan=False,
        )


# # plot the pareto frontier
# pareto_frontier = np.array([pareto_frontier[policy_name] for policy_name in OPTIMAL_POLICIES])
# plt.figure()
# ax = plt.axes(projection="3d")
# # add policy names as text labels
# for i, policy_name in enumerate(OPTIMAL_POLICIES):
#     ax.text(pareto_frontier[i, 0], pareto_frontier[i, 1], pareto_frontier[i, 2], policy_name)
# ax.scatter(pareto_frontier[:, 0], pareto_frontier[:, 1], pareto_frontier[:, 2])
# ax.set_xlabel("Objective 1")
# ax.set_ylabel("Objective 2")
# ax.set_zlabel("Objective 3")
# ax.set_title("Pareto Frontier")
# plt.show()
