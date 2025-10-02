'''Script to compute metrics for evaluating the quality of embeddings using well-established libraries.'''

import numpy as np
import torch
from utils import create_ground_truth_dst, get_returns
from reptrix import alpha, rankme, lidar
from zadu import zadu
import json
import ast
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def compute_reptrix_metrics(embeddings):
    '''Compute RepTrix metrics for the given embeddings.'''

    embeddings_tensor = torch.from_numpy(embeddings).float()
    
    metrics = {}
    
    metrics['alpha'] = alpha.get_alpha(embeddings_tensor) #OPTIMUM=1
    print(f"Alpha computed successfully: {metrics['alpha']}")

    metrics['rankme'] = rankme.get_rankme(embeddings_tensor) #OPTIMUM>1 (the higher the better)
    print(f"RankMe computed successfully: {metrics['rankme']}")
        
    return metrics
    

def compute_zadu_metrics(raw_data, embeddings, k=2):
    '''Compute ZADU metrics for the given raw data and embeddings.'''
    
    print(f"Computing ZADU metrics for raw_data shape: {raw_data.shape}, embeddings shape: {embeddings.shape}")
    
    spec = [{ #Trustworthiness & Continuity #OPTIMUM=1
        "id"    : "tnc",
        "params": { "k": k, "return_local": True }
    }, { #Mean Relative Rank Errors #OPTIMUM=0
        "id"    : "mrre",
        "params": { "k": k, "return_local": True }
    }
    
    ]

    zadu_instance = zadu.ZADU(spec, raw_data)
    result = zadu_instance.measure(embeddings)
    print(f"ZADU metrics computed successfully")
    return result
        

if __name__ == "__main__":

    #########True embeddings###############
    all_gt_emb = []
    all_returns = []
    for i in range(6):
        path = f"trajectories/left_right_dst/left_right_dst_{i}.json"
        emb_gt = create_ground_truth_dst(path)
        returns = get_returns(path)
        print(f"Policy {i} returns: {returns}")
        print(f"Policy {i} ground truth embedding: {emb_gt}")
        all_gt_emb.append(emb_gt)
        all_returns.append(returns)

    all_gt_emb = np.array(all_gt_emb)
    all_returns = np.array(all_returns)
    
    print(f"Data loaded successfully:")
    print(f"  all_gt_emb shape: {all_gt_emb.shape}")
    print(f"  all_returns shape: {all_returns.shape}")
    
    #  ZADU metrics
    print("\n--- Computing ZADU Metrics for Left Right GTE vs Objective ---")
    zadu_metrics = compute_zadu_metrics(all_returns, all_gt_emb)
    print("Left-right metrics:", zadu_metrics)

    all_gt_emb_concave = []
    all_returns_concave = []
    for i in range(10):
        path=f"trajectories/dst_concave/dst_{i}.json"
        emb_gt = create_ground_truth_dst(path)
        returns = get_returns(path)
        print(f"Policy {i} returns: {returns}")
        print(f"Policy {i} ground truth embedding: {emb_gt}")
        all_gt_emb_concave.append(emb_gt)
        all_returns_concave.append(returns)

    all_gt_emb_concave = np.array(all_gt_emb_concave)
    all_returns_concave = np.array(all_returns_concave)
    print("\n--- Computing ZADU Metrics for Concave GTE vs Objective ---")
    zadu_metrics_concave = compute_zadu_metrics(all_returns_concave, all_gt_emb_concave)
    print(f"Concave Metrics: {zadu_metrics_concave}")


    #########Transformer embeddings###############
    dim=["3D"]
    for dim_i in dim:
        all_t_emb = []
        all_returns = []
        path=f'trajectories/dst_concave/dst_concave_{dim_i}_embeddings.json'
        with open(path, "r") as f:
            embedding = json.load(f)
        all_returns = []
        for policy in embedding:
            returns = np.array(ast.literal_eval(policy['Return']))
            all_returns.append(returns)
            t_emb = np.array(policy['T_Embedding'])
            all_t_emb.append(t_emb)
        all_t_emb_concave = np.array(all_t_emb)
        all_returns = np.array(all_returns)
        print(f"Transformer embeddings and returns loaded successfully:")
        print(f"  all_t_emb shape: {all_t_emb_concave.shape}")
        print(f"  all_returns shape: {all_returns.shape}")
        print(f"\n--- Computing ZADU Metrics for Concave TE vs Objective {dim_i} ---")

        zadu_metrics_t = compute_zadu_metrics(all_returns, all_t_emb_concave)
        print(f"Transformer Metrics for {dim_i} and concave: {zadu_metrics_t}")


    for dim_i in dim:
        all_t_emb = []
        all_returns = []
        path=f'trajectories/left_right_dst/left_right_dst_{dim_i}_embeddings.json'
        with open(path, "r") as f:
            embedding = json.load(f)
        all_returns = []
        for policy in embedding:
            returns = np.array(ast.literal_eval(policy['Return']))
            all_returns.append(returns)
            t_emb = np.array(policy['T_Embedding'])
            all_t_emb.append(t_emb)
        all_t_emb = np.array(all_t_emb)
        all_returns = np.array(all_returns)
        print(f"Transformer embeddings and returns loaded successfully:")
        print(f"  all_t_emb shape: {all_t_emb.shape}")
        print(f"  all_returns shape: {all_returns.shape}")
        zadu_metrics_t = compute_zadu_metrics(all_returns, all_t_emb)
        print(f"\n--- Computing ZADU Metrics for Left Right TE vs Objective {dim_i} ---")

        print(f"Transformer Metrics for {dim_i} and left_right: {zadu_metrics_t}")


    #########FINAL check - GTE vs TE##################
    zadu_metrics_concave = compute_zadu_metrics(all_gt_emb_concave, all_t_emb_concave)
    print(f"\n--- Computing ZADU Metrics for Concave TE vs GTE {dim_i} ---")

    print(f"GTE vs TE Metrics for concave: {zadu_metrics_concave}")
    zadu_metrics_lr = compute_zadu_metrics(all_gt_emb, all_t_emb)
    print(f"\n--- Computing ZADU Metrics for Left Right TE vs GTE {dim_i} ---")


    print('--------------------------------------------')
    #conduct PCA on embeddings to put them in 2D
    pca = PCA(n_components=2)
    all_t_emb_concave_2d = pca.fit_transform(all_t_emb_concave)
    all_t_emb_2d = pca.fit_transform(all_t_emb)
    zadu_metrics_concave_2d = compute_zadu_metrics(all_gt_emb_concave, all_t_emb_concave_2d)
    print(f"GTE vs TE Metrics for concave 2D: {zadu_metrics_concave_2d}")
    zadu_metrics_ls_2d = compute_zadu_metrics(all_gt_emb, all_t_emb_2d)
    print(f"GTE vs TE Metrics for left_right 2D: {zadu_metrics_ls_2d}")



    print(f"GTE vs TE Metrics for left_right: {zadu_metrics_lr}")


    print("\n Left right")
    print(f"\n All returns: {all_returns} \n All GTE: {all_gt_emb} \n All TE {dim_i}: {all_t_emb} ")

    print("\n Concave ")
    print(f"\n All returns: {all_returns_concave} \n All GTE: {all_gt_emb_concave} \n All TE {dim_i}: {all_t_emb_concave} ")

    ########Standardizing returns and gt embeddings###############
    scaler = StandardScaler()
    #returns
    all_returns_std = scaler.fit_transform(all_returns)
    all_returns_concave_std = scaler.fit_transform(all_returns_concave)
    print(f"\n Standardized returns: {all_returns_std}")
    print(f"\n Standardized concave returns: {all_returns_concave_std}")
    #gt embeddings
    all_gt_emb_std = scaler.fit_transform(all_gt_emb)
    all_gt_emb_concave_std = scaler.fit_transform(all_gt_emb_concave)
    print(f"\n Standardized GTE: {all_gt_emb_std}")
    print(f"\n Standardized concave GTE: {all_gt_emb_concave_std}")

    print("__________STANDARDIZED METRICS__________")
    zadu_metrics_r_gt_std = compute_zadu_metrics(all_returns_std, all_gt_emb_std)
    zadu_metrics_r_gt_concave_std = compute_zadu_metrics(all_returns_concave_std, all_gt_emb_concave_std)
    zadu_metrics_r_t_std = compute_zadu_metrics(all_returns_std, all_t_emb)
    zadu_metrics_r_t_concave_std = compute_zadu_metrics(all_returns_concave_std, all_t_emb_concave)
    zadu_metrics_gt_t_std = compute_zadu_metrics(all_gt_emb_std, all_t_emb)
    zadu_metrics_gt_t_concave_std = compute_zadu_metrics(all_gt_emb_concave_std, all_t_emb_concave)
    print(f"Standardized Returns vs Standardized GTE Metrics for left_right: {zadu_metrics_r_gt_std}")
    print(f"Standardized Returns vs Standardized GTE Metrics for concave: {zadu_metrics_r_gt_concave_std}")
    print(f"Standardized Returns vs TE Metrics for left_right: {zadu_metrics_r_t_std}")
    print(f"Standardized Returns vs TE Metrics for concave: {zadu_metrics_r_t_concave_std}")
    print(f"Standardized GTE vs TE Metrics for left_right: {zadu_metrics_gt_t_std}")
    print(f"Standardized GTE vs TE Metrics for concave: {zadu_metrics_gt_t_concave_std}")

    #do the same analusis but on normalized data (0-1)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    #returns
    all_returns_norm = scaler.fit_transform(all_returns)
    all_returns_concave_norm = scaler.fit_transform(all_returns_concave)
    print(f"\n Normalized returns: {all_returns_norm}")
    print(f"\n Normalized concave returns: {all_returns_concave_norm}")
    #gt embeddings
    all_gt_emb_norm = scaler.fit_transform(all_gt_emb)
    all_gt_emb_concave_norm = scaler.fit_transform(all_gt_emb_concave)
    print(f"\n Normalized GTE: {all_gt_emb_norm}")
    print(f"\n Normalized concave GTE: {all_gt_emb_concave_norm}")  
    #t embeddings
    all_t_emb_norm = scaler.fit_transform(all_t_emb)
    all_t_emb_concave_norm = scaler.fit_transform(all_t_emb_concave)
    print(f"\n Normalized TE: {all_t_emb_norm}")
    print(f"\n Normalized concave TE: {all_t_emb_concave_norm}")    

    #metrics
    print("__________NORMALIZED METRICS__________")
    zadu_metrics_r_gt_norm = compute_zadu_metrics(all_returns_norm, all_gt_emb_norm)
    zadu_metrics_r_gt_concave_norm = compute_zadu_metrics(all_returns_concave_norm, all_gt_emb_concave_norm)
    zadu_metrics_r_t_norm = compute_zadu_metrics(all_returns_norm, all_t_emb_norm)
    zadu_metrics_r_t_concave_norm = compute_zadu_metrics(all_returns_concave_norm, all_t_emb_concave_norm)
    zadu_metrics_gt_t_norm = compute_zadu_metrics(all_gt_emb_norm, all_t_emb_norm)
    zadu_metrics_gt_t_concave_norm = compute_zadu_metrics(all_gt_emb_concave_norm, all_t_emb_concave_norm)
    print(f"Normalized Returns vs Normalized GTE Metrics for left_right: {zadu_metrics_r_gt_norm}")
    print(f"Normalized Returns vs Normalized GTE Metrics for concave: {zadu_metrics_r_gt_concave_norm}")
    print(f"Normalized Returns vs TE Metrics for left_right: {zadu_metrics_r_t_norm}")
    print(f"Normalized Returns vs TE Metrics for concave: {zadu_metrics_r_t_concave_norm}")
    print(f"Normalized GTE vs TE Metrics for left_right: {zadu_metrics_gt_t_norm}")
    print(f"Normalized GTE vs TE Metrics for concave: {zadu_metrics_gt_t_concave_norm}")

    #visualise 3D embeddings concave in 3d, gt embeddings in 2d and returns in 2d seperetly, next to each other
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(18, 6))

    # 3D scatter for transformer embeddings
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(all_t_emb_concave[:,0], all_t_emb_concave[:,1], all_t_emb_concave[:,2], c='b', marker='o')
    ax1.set_title('Concave TE in 3D')
    ax1.set_xlabel('TE Dim 1')
    ax1.set_ylabel('TE Dim 2')
    ax1.set_zlabel('TE Dim 3')

    # 2D scatter for returns
    ax2 = fig.add_subplot(132)
    ax2.scatter(all_returns_concave[:,0], all_returns_concave[:,1], c='r', marker='o')
    ax2.set_title('Concave Returns in 2D')
    ax2.set_xlabel('Return Dim 1')
    ax2.set_ylabel('Return Dim 2')

    # 2D scatter for ground truth embeddings
    ax3 = fig.add_subplot(133)
    ax3.scatter(all_gt_emb_concave[:,0], all_gt_emb_concave[:,1], c='g', marker='o')
    ax3.set_title('Concave GTE in 2D')
    ax3.set_xlabel('GTE Dim 1')
    ax3.set_ylabel('GTE Dim 2')

    plt.tight_layout()
    plt.show()
