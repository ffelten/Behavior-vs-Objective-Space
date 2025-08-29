import matplotlib.pyplot as plt
import mo_gymnasium as mo_gym
import numpy as np
import numpy.typing as npt
import morl_behavior_objective


LEFT = 2
RIGHT = 3
DOWN = 1

OPTIMAL_POLICIES = {
    "7_left": 4 * [LEFT] + 5 * [DOWN],  # 9 moves
    "6_right": 3 * [RIGHT] + 5 * [DOWN],  # 8 moves
    "25_left": 7 * [LEFT] + 8 * [DOWN],  # 15 moves
    "24_right": 6 * [RIGHT] + 8 * [DOWN],  # 14 moves
    "120_left": 9 * [LEFT] + 9 * [DOWN],  # 18 moves
    "124_right": 9 * [RIGHT] + 10 * [DOWN],  # 19 moves
}


env = mo_gym.make("left-right-dst-v0", render_mode="human")

policy_disc_returns: dict[str, npt.NDArray] = {}

for policy_name, policy in OPTIMAL_POLICIES.items():
    env.reset()
    done = False
    disc_return = np.array([0.0, 0.0])
    discount = 1.0
    i = 0
    print("Executing policy: ", policy_name)
    while not done:
        action = policy[i]
        i += 1
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        disc_return += discount * reward
        discount *= 1.0

    policy_disc_returns[policy_name] = disc_return

print(policy_disc_returns)

# scatter plot for each policy
plt.figure(figsize=(10, 8))

# Extract x and y coordinates from policy_disc_returns
x_coords = [returns[0] for returns in policy_disc_returns.values()]
y_coords = [returns[1] for returns in policy_disc_returns.values()]
policy_names = list(policy_disc_returns.keys())

# Create scatter plot with blue for left policies and red for right policies
scatter_plots = []

for name, returns in policy_disc_returns.items():
    # Determine color based on policy name
    if "left" in name:
        color = "blue"
    elif "right" in name:
        color = "red"
    scatter = plt.scatter(returns[0], returns[1], s=100, alpha=0.7, color=color, label=name)
    scatter_plots.append(scatter)

    # Add labels for each point
    plt.annotate(name, (returns[0], returns[1]), xytext=(5, 5), textcoords="offset points", fontsize=10, fontweight="bold")

# Add legend
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

# Labels and title
plt.xlabel("Time")
plt.ylabel("Treasure Value")
plt.title("Pareto Frontier")
plt.grid(True, alpha=0.3)

# Adjust layout to prevent legend cutoff
plt.tight_layout()

plt.show()
