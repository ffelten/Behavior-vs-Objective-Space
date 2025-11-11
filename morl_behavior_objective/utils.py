import json
import os

from gymnasium.wrappers import FlattenObservation
from gymnasium.wrappers import TimeLimit
import imageio
import mo_gymnasium as mo_gym
from morl_baselines.multi_policy.morld.morld import MORLD
import numpy as np
import numpy.typing as npt

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


def create_ground_truth_dst(trajectory_path: str) -> npt.NDArray:
    """Create ground truth deep sea treasure embeddings from trajectories.

    Args:
        trajectory_path (str): Path to the trajectory data.

    Returns:
        npt.NDArray: Ground truth embedding array [n_right - n_left, n_down].
    """
    with open(trajectory_path) as f:
        policy_i = json.load(f)

    # get actions from the first trajectory (all trajectories are the same)
    trajectory = policy_i["trajectories"][1]
    # flatten the 2D list of actions
    actions = [action[0] for action in trajectory[1]]

    # count the number of left, right, down actions
    n_left = actions.count(2)
    n_right = actions.count(3)
    n_down = actions.count(1)

    emb_gt = np.array([n_right - n_left, n_down])

    return emb_gt


def load_ground_truth_data(
    env_name: str, num_files: int, base_path: str = "../../trajectories"
) -> tuple[npt.NDArray, npt.NDArray]:
    """Load ground truth embeddings and returns for a dataset.

    Args:
        env_name: Name of the environment folder (e.g., 'left_right_dst', 'dst_concave')
        num_files: Number of trajectory files to load
        base_path: Base path to trajectories folder (default: '../../trajectories' for notebooks in analysis/)

    Returns:
        tuple: (embeddings array, returns array)
    """
    all_embeddings = []
    all_returns = []

    for i in range(num_files):
        path = f"{base_path}/{env_name}/policy_{i}.json"
        emb_gt = create_ground_truth_dst(path)
        returns = get_returns(path)
        all_embeddings.append(emb_gt)
        all_returns.append(returns)

    return np.array(all_embeddings), np.array(all_returns)


def load_transformer_embeddings(embedding_path: str) -> tuple[npt.NDArray, npt.NDArray]:
    """Load transformer embeddings from a JSON file.

    Args:
        embedding_path: Path to the embeddings JSON file

    Returns:
        tuple: (embeddings array, returns array)
    """
    with open(embedding_path) as f:
        embedding = json.load(f)

    all_embeddings = []
    all_returns = []

    for policy in embedding:
        returns = np.array(policy["Return"])
        all_returns.append(returns)
        t_emb = np.array(policy["T_Embedding"])
        all_embeddings.append(t_emb)

    return np.array(all_embeddings), np.array(all_returns)


def get_returns(trajectory_path: str) -> npt.NDArray:
    """Extract objectives from a trajectory file.

    Args:
        trajectory_path (str): Path to the trajectory data.
    """
    with open(trajectory_path) as f:
        policy_i = json.load(f)

    # get the returns for the policy
    returns = np.array(policy_i["return"])

    return returns


def get_returns_json(trajectory_path: str) -> npt.NDArray:
    """Extract objectives from a trajectory file in JSON format.

    Args:
        trajectory_path (str): Path to the trajectory data.
    """
    with open(trajectory_path) as f:
        policy_i = json.load(f)

    # get the returns for the policy
    returns = np.array(policy_i["return"])

    return returns


def get_pareto_front(directory_path: str, pareto_size: int) -> npt.NDArray:
    """Extract Pareto front objectives from a trajectory file.

    Args:
        trajectory_path (str): Path to the trajectory data.
    """
    import os

    # get list with names of all files in the directory
    pareto_front = []
    for i in range(pareto_size):
        returns_i = get_returns_json(os.path.join(directory_path, f"policy_{i}.json"))
        pareto_front.append(returns_i)
    pareto_front = np.array(pareto_front)
    return pareto_front


def visualize_pareto_front(pareto_front: npt.NDArray) -> None:
    """Visualize the Pareto front.

    Args:
        pareto_front (npt.NDArray): Array of objectives in the Pareto front.
    """
    import matplotlib.pyplot as plt

    plt.figure()
    if pareto_front.shape[1] == 2:
        plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c="blue")
        plt.xlabel("Objective 1")
        plt.ylabel("Objective 2")
        plt.title("Pareto Front")
        plt.grid()
        plt.show()
    elif pareto_front.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], c="blue")
        ax.set_xlabel("Objective 1")
        ax.set_ylabel("Objective 2")
        ax.set_zlabel("Objective 3")
        ax.set_title("Pareto Front")
        plt.show()
    else:
        print("Visualization only supported for 2D or 3D objectives.")


# ...existing code...
def visualize_pareto_front(
    pareto_front: npt.NDArray, use_plotly: bool = True, save_html: str | None = None, show: bool = True
):
    """Visualize the Pareto front using Plotly (preferred) with Matplotlib fallback.

    Args:
        pareto_front: (n_points, d) array of objectives (d==2 or d==3).
        use_plotly: prefer Plotly if available.
        save_html: optional path to save interactive html (Plotly).
        show: whether to show the figure (fig.show() for Plotly / plt.show() for Matplotlib).

    Returns:
        fig object (plotly.graph_objects.Figure or matplotlib.Figure)
    """
    if pareto_front is None or len(pareto_front) == 0:
        raise ValueError("pareto_front is empty")

    pf = np.asarray(pareto_front)
    if pf.ndim != 2 or pf.shape[1] not in (2, 3):
        raise ValueError("Visualization only supported for 2D or 3D objectives (shape (n,2) or (n,3))")

    n = pf.shape[0]
    indices = np.arange(n)

    if use_plotly:
        try:
            import plotly.express as px

            if pf.shape[1] == 2:
                x = pf[:, 0]
                y = pf[:, 1]
                hover = [f"idx: {i}<br>({x[i]:.3f}, {y[i]:.3f})" for i in indices]
                fig = px.scatter(
                    x=x,
                    y=y,
                    text=[str(i) for i in indices],
                    labels={"x": "Objective 1", "y": "Objective 2"},
                    title="Pareto Front (2D)",
                )
                fig.update_traces(
                    marker=dict(size=7), hovertemplate="%{customdata}", customdata=hover, textposition="top center"
                )

            else:
                x = pf[:, 0]
                y = pf[:, 1]
                z = pf[:, 2]
                hover = [f"idx: {i}<br>({x[i]:.3f}, {y[i]:.3f}, {z[i]:.3f})" for i in indices]
                fig = px.scatter_3d(
                    x=x,
                    y=y,
                    z=z,
                    text=[str(i) for i in indices],
                    labels={"x": "Objective 1", "y": "Objective 2", "z": "Objective 3"},
                    title="Pareto Front (3D)",
                )
                fig.update_traces(marker=dict(size=4), hovertemplate="%{customdata}", customdata=hover)

            if save_html:
                try:
                    fig.write_html(save_html)
                except Exception as e:
                    print(f"Warning: failed to write html {save_html}: {e}")

            if show:
                fig.show()
            return fig

        except Exception:
            # Plotly not available or failed — fall back to matplotlib
            pass

    # Matplotlib fallback
    import matplotlib.pyplot as plt

    fig = plt.figure()
    if pf.shape[1] == 2:
        ax = fig.add_subplot(111)
        ax.scatter(pf[:, 0], pf[:, 1], c="blue")
        for i, (x, y) in enumerate(pf):
            ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(5, 3), fontsize=8)
        ax.set_xlabel("Objective 1")
        ax.set_ylabel("Objective 2")
        ax.set_title("Pareto Front")
        ax.grid(True)
    else:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], c="blue")
        for i, (x, y, z) in enumerate(pf):
            ax.text(x, y, z, str(i), fontsize=8)
        ax.set_xlabel("Objective 1")
        ax.set_ylabel("Objective 2")
        ax.set_zlabel("Objective 3")
        ax.set_title("Pareto Front")

    if show:
        plt.show()
    return fig


def render_policy(
    env_id: str, check_point: str, policy_id: int, n_episodes: int, save_dic=None, fps=30, base_path: str = "."
) -> None:
    """Renders a single MORLD policy by its ID.

    Args:
        env_id: Environment ID
        check_point: Path to checkpoint file (relative to base_path)
        policy_id: Policy ID to render
        n_episodes: Number of episodes to render
        save_dic: Directory to save videos (optional)
        fps: Frames per second for video
        base_path: Base path for checkpoint file (default: current directory)
    """
    # Construct full checkpoint path
    full_check_point = os.path.join(base_path, check_point) if not os.path.isabs(check_point) else check_point

    if save_dic is not None:
        env = mo_gym.make(env_id, render_mode="rgb_array")
        os.makedirs(save_dic, exist_ok=True)
    else:
        # check if directory exists otherwise create it
        env = mo_gym.make(env_id, render_mode="human")
    if "mo-halfcheetah" in env_id.lower() or "mo-hopper" in env_id.lower():
        env = TimeLimit(env, max_episode_steps=100)
    elif "highway" in env_id.lower():
        env = FlattenObservation(env)
    else:
        raise ValueError(
            f"Environment {env_id} not supported, only 'mo-halfcheetah', 'mo-hopper' and 'highway' are supported"
        )
    agent = MORLD(env=env, log=False, seed=params["seed"], **params["init_hyperparams"])

    agent.load(full_check_point, load_replay_buffer=False)

    if not hasattr(agent, "archive") or len(agent.archive.individuals) == 0:
        raise RuntimeError("MORLD pareto archive is empty after load().")

    pol = agent.archive.individuals[policy_id]
    pweights = getattr(pol, "weights", None)
    print(f"[MORLD] {env_id}: rendering policy {policy_id} with weights {pweights}")
    for episode in range(n_episodes):
        frames = []
        obs, _ = env.reset()
        terminated = truncated = False

        while not (terminated or truncated):
            if save_dic is not None:
                frame = env.render()
                frames.append(frame)
            else:
                env.render()
            try:
                action = pol.wrapped.eval(obs, None)
            except TypeError:
                action = pol.wrapped.eval(obs)

            obs, reward, terminated, truncated, info = env.step(action)

        if save_dic is not None:
            imageio.mimsave(os.path.join(save_dic, f"{env_id}_policy_{policy_id}_ep_{episode}.gif"), frames, fps=fps)
    env.close()


if __name__ == "__main__":
    # pf= get_pareto_front("trajectories/morld/mo-highway-fast-v0_100steps_test", pareto_size=41)
    # print(pf)
    # visualize_pareto_front(pf, save_html="pareto_front.html")
    render_policy(
        env_id="mo-halfcheetah-v5",
        check_point="MORL_policies/morld_cheetah_v5/seed0.tar",
        policy_id=50,
        n_episodes=5,
        save_dic="videos",
    )
