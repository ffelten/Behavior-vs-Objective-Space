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

def get_returns_json(trajectory_path: str) -> npt.NDArray:
    """Extract objectives from a trajectory file in JSON format.

    Args:
        trajectory_path (str): Path to the trajectory data.
    """
    with open(trajectory_path, "r") as f:
        policy_i = json.load(f)
    
    #get the returns for the policy
    returns = np.array(policy_i['return'])

    return returns

def get_pareto_front(directory_path: str) -> npt.NDArray:
    """Extract Pareto front objectives from a trajectory file.

    Args:
        trajectory_path (str): Path to the trajectory data.
    """
    #get list with names of all files in the directory
    import os
    file_names = os.listdir(directory_path)
    pareto_front = []
    for file_name in file_names:
        returns_i = get_returns_json(os.path.join(directory_path, file_name))
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
        plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c='blue')
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title('Pareto Front')
        plt.grid()
        plt.show()
    elif pareto_front.shape[1] == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], c='blue')
        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.set_zlabel('Objective 3')
        ax.set_title('Pareto Front')
        plt.show()
    else:
        print("Visualization only supported for 2D or 3D objectives.")





if __name__ == "__main__":
    pf= get_pareto_front("trajectories/morld/mo-halfcheetah-v4_100steps")
    print(pf)
    visualize_pareto_front(pf)