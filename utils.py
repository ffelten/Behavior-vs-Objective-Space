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

def get_pareto_front(directory_path: str,  pareto_size: int) -> npt.NDArray:
    """Extract Pareto front objectives from a trajectory file.

    Args:
        trajectory_path (str): Path to the trajectory data.
    """
    import os
    #get list with names of all files in the directory
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

# ...existing code...
def visualize_pareto_front(pareto_front: npt.NDArray, use_plotly: bool = True, save_html: str | None = None, show: bool = True):
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
            import plotly.io as pio

            if pf.shape[1] == 2:
                x = pf[:, 0]
                y = pf[:, 1]
                hover = [f"idx: {i}<br>({x[i]:.3f}, {y[i]:.3f})" for i in indices]
                fig = px.scatter(x=x, y=y, text=[str(i) for i in indices],
                                 labels={'x': 'Objective 1', 'y': 'Objective 2'},
                                 title='Pareto Front (2D)')
                fig.update_traces(marker=dict(size=7), hovertemplate='%{customdata}', customdata=hover, textposition='top center')

            else:
                x = pf[:, 0]
                y = pf[:, 1]
                z = pf[:, 2]
                hover = [f"idx: {i}<br>({x[i]:.3f}, {y[i]:.3f}, {z[i]:.3f})" for i in indices]
                fig = px.scatter_3d(x=x, y=y, z=z, text=[str(i) for i in indices],
                                    labels={'x': 'Objective 1', 'y': 'Objective 2', 'z': 'Objective 3'},
                                    title='Pareto Front (3D)')
                fig.update_traces(marker=dict(size=4), hovertemplate='%{customdata}', customdata=hover)

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
        ax.scatter(pf[:, 0], pf[:, 1], c='blue')
        for i, (x, y) in enumerate(pf):
            ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(5, 3), fontsize=8)
        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.set_title('Pareto Front')
        ax.grid(True)
    else:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], c='blue')
        for i, (x, y, z) in enumerate(pf):
            ax.text(x, y, z, str(i), fontsize=8)
        ax.set_xlabel('Objective 1'); ax.set_ylabel('Objective 2'); ax.set_zlabel('Objective 3')
        ax.set_title('Pareto Front')

    if show:
        plt.show()
    return fig
#



if __name__ == "__main__":
    pf= get_pareto_front("trajectories/morld/mo-highway-fast-v0_100steps_test", pareto_size=41)
    print(pf)
    visualize_pareto_front(pf, save_html="pareto_front.html")