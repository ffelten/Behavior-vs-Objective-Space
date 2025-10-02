import json
import numpy as np
import numpy.typing as npt




def create_ground_truth_dst(trajectory_path: str) -> None:
    """Create ground truth deep sea treasure embeddings from trajectories.

    Args:
        trajectory_path (str): Path to the trajectory data.
    """
    with open(trajectory_path, "r") as f:
        policy_i = json.load(f)
    
    #get actions from the first trajectory (all trajectories are the same)
    trajectory = policy_i['trajectories'][1]
    #flatten the 2D list of actions
    actions = [action[0] for action in trajectory[1]]

    #count the number of left, right, down actions
    n_left = actions.count(2)
    n_right = actions.count(3)
    n_down = actions.count(1)

    emb_gt = np.array([n_right - n_left, n_down])

    return emb_gt

def get_returns(trajectory_path: str) -> npt.NDArray:
    """Extract objectives from a trajectory file.

    Args:
        trajectory_path (str): Path to the trajectory data.
    """
    with open(trajectory_path, "r") as f:
        policy_i = json.load(f)
    
    #get the returns for the policy
    returns = np.array(policy_i['return'])

    return returns

