"""This script renders and saves optimal policies for the Left-Right DST environment as MP4 videos."""

import os

import imageio
import mo_gymnasium as mo_gym
import numpy as np

import morl_behavior_objective  # noqa: F401  # registers custom DST environments via side effect

# Action definitions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# As in Vamplew et al. (2018):
CONCAVE_MAP = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-10, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-10, -10, 3.0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-10, -10, -10, 5.0, 8.0, 16.0, 0, 0, 0, 0, 0],
        [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
        [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
        [-10, -10, -10, -10, -10, -10, 24.0, 50.0, 0, 0, 0],
        [-10, -10, -10, -10, -10, -10, -10, -10, 0, 0, 0],
        [-10, -10, -10, -10, -10, -10, -10, -10, 74.0, 0, 0],
        [-10, -10, -10, -10, -10, -10, -10, -10, -10, 124.0, 0],
    ]
)

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

# env = mo_gym.make("left-right-dst-v0", render_mode=None)
# env = mo_gym.make("deep-sea-treasure-louvre-dst-v0", render_mode="human")
# env = mo_gym.make("deep-sea-treasure-smooth-v0", render_mode="human")

# Settings
OUTPUT_DIR = "videos/smooth_dst"
os.makedirs(OUTPUT_DIR, exist_ok=True)
ENV_ID = "deep-sea-treasure-smooth-v0"
FPS = 4  # Frames per second


# Function to render and save a policy as MP4
def render_policy_to_video(policy_name: str, policy: list[int], env_id: str, save_path: str, fps: int = 4):
    """Render a fixed action sequence in the environment and save the frames as an MP4."""
    env = mo_gym.make(
        env_id,
        render_mode="rgb_array",  # ,dst_map=CONCAVE_MAP
    )
    _obs, _ = env.reset()
    done = False
    i = 0
    frames = []

    while not done and i < len(policy):
        frame = env.render()
        frames.append(frame)

        action = policy[i]
        i += 1
        _obs, _reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated

    env.close()

    imageio.mimsave(save_path, frames, fps=fps, codec="libx264")
    print(f"Saved: {save_path}")


# Render and save each policy
for policy_name, policy in OPTIMAL_POLICIES_SMOOTH.items():
    filename = f"{policy_name}.mp4"
    filepath = os.path.join(OUTPUT_DIR, filename)
    print(f"Rendering policy: {policy_name}")
    render_policy_to_video(policy_name, policy, ENV_ID, filepath, fps=FPS)
