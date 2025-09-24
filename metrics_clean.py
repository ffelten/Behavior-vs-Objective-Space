'''Script to compute metrics for evaluating the quality of embeddings using well-established libraries.'''

import numpy as np
import torch
from utils import create_ground_truth_dst, get_returns
from reptrix import alpha, rankme, lidar
from zadu import zadu
import json
import ast

def compute_reptrix_metrics(embeddings):
    '''Compute RepTrix metrics for the given embeddings.'''

    embeddings_tensor = torch.from_numpy(embeddings).float()
    
    metrics = {}
    
    metrics['alpha'] = alpha.get_alpha(embeddings_tensor) #OPTIMUM=1
    print(f"Alpha computed successfully: {metrics['alpha']}")

    metrics['rankme'] = rankme.get_rankme(embeddings_tensor) #OPTIMUM>1 (the higher the better)
    print(f"RankMe computed successfully: {metrics['rankme']}")
        
    return metrics
    

def compute_zadu_metrics(raw_data, embeddings):
    '''Compute ZADU metrics for the given raw data and embeddings.'''
    
    print(f"Computing ZADU metrics for raw_data shape: {raw_data.shape}, embeddings shape: {embeddings.shape}")
    
    spec = [{ #Trustworthiness & Continuity #OPTIMUM=1
        "id"    : "tnc",
        "params": { "k": 2, "return_local": True }
    }, { #Mean Relative Rank Errors #OPTIMUM=0
        "id"    : "mrre",
        "params": { "k": 2, "return_local": True }
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
    print("\n--- Computing ZADU Metrics ---")
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
    zadu_metrics_concave = compute_zadu_metrics(all_returns_concave, all_gt_emb_concave)
    print(f"Concave Metrics: {zadu_metrics_concave}")


    #########Transformer embeddings###############
    dim=["3D", "4D"]
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
        print(f"Transformer Metrics for {dim_i} and left_right: {zadu_metrics_t}")


    #########FINAL check - GTE vs TE##################
    zadu_metrics_concave = compute_zadu_metrics(all_t_emb_concave, all_gt_emb_concave)
    print(f"GTE vs TE Metrics for concave: {zadu_metrics_concave}")
    zadu_metrics_lr = compute_zadu_metrics(all_t_emb, all_gt_emb)
    print(f"GTE vs TE Metrics for left_right: {zadu_metrics_lr}")
