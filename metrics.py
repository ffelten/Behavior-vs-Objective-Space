"""
Diagnostic metrics for comparing behavior and objective space representations.
Each function takes representations in two different spaces and computes relevant metrics.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness, Isomap
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
from scipy.linalg import orthogonal_procrustes
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
from utils import create_ground_truth_dst, get_returns

def compute_pca_metrics(embedding_space, n_components=None):
    """
    Compute PCA-based metrics for structure analysis.
    
    Args:
        embedding_space (np.ndarray): Embedding representations (n_samples, n_features)
        n_components (int, optional): Number of PCA components. If None, uses min(n_features, n_samples-1)
    
    Returns:
        dict: Dictionary containing PCA metrics
    """
    try:
        n_samples, n_features = embedding_space.shape
        if n_components is None:
            n_components = min(n_features, max(1, n_samples - 1))
        
        pca = PCA(n_components=n_components).fit(embedding_space)
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = explained_variance_ratio.cumsum()
        eigenvalues = pca.explained_variance_
        
        # Participation ratio
        participation_ratio = (eigenvalues.sum()**2) / (eigenvalues**2).sum()
        
        return {
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'eigenvalues': eigenvalues,
            'participation_ratio': participation_ratio
        }
    except Exception as e:
        return {
            'explained_variance_ratio': np.array([]),
            'cumulative_variance': np.array([]),
            'eigenvalues': np.array([]),
            'participation_ratio': float('nan')
        }


def compute_intrinsic_dimension(embedding_space, k=None):
    """
    Compute intrinsic dimension using MLE method.
    
    Args:
        embedding_space (np.ndarray): Embedding representations
        k (int, optional): Number of neighbors. If None, uses min(5, n_samples-1)
    
    Returns:
        dict: Dictionary containing intrinsic dimension estimate
    """
    try:
        from skdim import id as intrinsic_dimension_estimators
        
        n_samples = embedding_space.shape[0]
        if k is None:
            k = max(2, min(5, n_samples - 1))
        
        # Add small jitter to avoid numerical issues
        jittered_data = embedding_space + 1e-9 * np.random.randn(*embedding_space.shape)
        
        mle_estimator = intrinsic_dimension_estimators.MLE(k=k)
        intrinsic_dim = mle_estimator.fit_transform(jittered_data)
        
        return {
            'intrinsic_dimension': float(intrinsic_dim),
            'k_neighbors': k
        }
    except Exception as e:
        return {
            'intrinsic_dimension': float('nan'),
            'k_neighbors': k if k is not None else None
        }


def compute_distance_correlations(embedding_space, objective_space, standardize_objective=True):
    """
    Compute correlations between distance matrices in embedding and objective spaces.
    
    Args:
        embedding_space (np.ndarray): Embedding representations
        objective_space (np.ndarray): Objective space representations
        standardize_objective (bool): Whether to standardize objective space
    
    Returns:
        dict: Dictionary containing correlation metrics
    """
    try:
        if standardize_objective:
            scaler = StandardScaler()
            objective_space_std = scaler.fit_transform(objective_space)
        else:
            objective_space_std = objective_space
        
        # Compute distance matrices
        dist_embedding = pairwise_distances(embedding_space, metric='euclidean')
        dist_objective = pairwise_distances(objective_space_std, metric='euclidean')
        
        # Get upper triangular indices (excluding diagonal)
        n = embedding_space.shape[0]
        upper_tri_indices = np.triu_indices(n, k=1)
        
        # Extract upper triangular distances
        dist_emb_vec = dist_embedding[upper_tri_indices]
        dist_obj_vec = dist_objective[upper_tri_indices]
        
        # Spearman correlation
        rho, p_value = spearmanr(dist_emb_vec, dist_obj_vec)
        
        return {
            'spearman_correlation': float(rho),
            'spearman_p_value': float(p_value),
            'distance_matrix_embedding': dist_embedding,
            'distance_matrix_objective': dist_objective
        }
    except Exception as e:
        return {
            'spearman_correlation': float('nan'),
            'spearman_p_value': float('nan'),
            'distance_matrix_embedding': None,
            'distance_matrix_objective': None
        }


def compute_mantel_test(embedding_space, objective_space, perms=2000, seed=42, 
                       save_histogram=False, image_dir=None):
    """
    Compute Mantel permutation test for distance matrix correlation.
    
    Args:
        embedding_space (np.ndarray): Embedding representations
        objective_space (np.ndarray): Objective space representations
        perms (int): Number of permutations
        seed (int): Random seed
        save_histogram (bool): Whether to save histogram of null distribution
        image_dir (str): Directory to save histogram
    
    Returns:
        dict: Dictionary containing Mantel test results
    """
    try:
        from scipy.spatial.distance import pdist, squareform
        
        # Standardize objective space
        scaler = StandardScaler()
        objective_space_std = scaler.fit_transform(objective_space)
        
        # Compute distance matrices
        dist_embedding = pairwise_distances(embedding_space, metric='euclidean')
        dist_objective = pairwise_distances(objective_space_std, metric='euclidean')
        
        # Get upper triangular elements
        n = embedding_space.shape[0]
        upper_tri_indices = np.triu_indices(n, k=1)
        
        # Observed correlation
        observed_corr, _ = spearmanr(dist_embedding[upper_tri_indices], 
                                   dist_objective[upper_tri_indices])
        
        # Permutation test
        np.random.seed(seed)
        null_correlations = []
        
        for _ in range(perms):
            # Permute rows and columns of one distance matrix
            perm_indices = np.random.permutation(n)
            dist_embedding_perm = dist_embedding[np.ix_(perm_indices, perm_indices)]
            
            null_corr, _ = spearmanr(dist_embedding_perm[upper_tri_indices],
                                   dist_objective[upper_tri_indices])
            null_correlations.append(null_corr)
        
        null_correlations = np.array(null_correlations)
        
        # Two-sided p-value
        p_value = np.mean(np.abs(null_correlations) >= np.abs(observed_corr))
        
        # Save histogram if requested
        histogram_path = None
        if save_histogram and image_dir:
            os.makedirs(image_dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(null_correlations, bins=50, alpha=0.7, density=True, color='lightblue')
            ax.axvline(observed_corr, color='red', linestyle='--', linewidth=2, 
                      label=f'Observed: {observed_corr:.3f}')
            ax.set_xlabel('Correlation')
            ax.set_ylabel('Density')
            ax.set_title(f'Mantel Test Null Distribution (p={p_value:.3f})')
            ax.legend()
            histogram_path = os.path.join(image_dir, 'mantel_histogram.pdf')
            plt.tight_layout()
            plt.savefig(histogram_path)
            plt.close(fig)
        
        return {
            'mantel_correlation': float(observed_corr),
            'mantel_p_value': float(p_value),
            'null_distribution': null_correlations,
            'histogram_path': histogram_path
        }
    except Exception as e:
        return {
            'mantel_correlation': float('nan'),
            'mantel_p_value': float('nan'),
            'null_distribution': None,
            'histogram_path': None
        }


def compute_neighborhood_metrics(embedding_space, objective_space, k=None, add_jitter=True):
    """
    Compute local neighborhood agreement metrics.
    
    Args:
        embedding_space (np.ndarray): Embedding representations
        objective_space (np.ndarray): Objective space representations
        k (int, optional): Number of neighbors. If None, computed adaptively
        add_jitter (bool): Whether to add small jitter to break ties
    
    Returns:
        dict: Dictionary containing neighborhood metrics
    """
    try:
        n_samples = embedding_space.shape[0]
        
        # Determine k adaptively
        if k is None:
            k = min(3, max(1, n_samples - 3))
        
        # Standardize objective space
        scaler = StandardScaler()
        objective_space_std = scaler.fit_transform(objective_space)
        
        # Add jitter to break ties if requested
        if add_jitter:
            rng = np.random.RandomState(42)
            objective_space_std += 1e-9 * rng.normal(size=objective_space_std.shape)
        
        # Trustworthiness (how well objective space neighborhoods are preserved in embedding)
        try:
            trustworthiness_score = trustworthiness(objective_space_std, embedding_space, 
                                                  n_neighbors=k)
        except Exception:
            # Fallback to k=1 if needed
            trustworthiness_score = trustworthiness(objective_space_std, embedding_space, 
                                                  n_neighbors=1)
        
        # kNN overlap
        def knn_overlap(X, Y, k):
            """Compute k-nearest neighbor overlap between two spaces."""
            dist_X = pairwise_distances(X)
            dist_Y = pairwise_distances(Y)
            
            nn_X = np.argsort(dist_X, axis=1)[:, 1:k+1]  # Exclude self
            nn_Y = np.argsort(dist_Y, axis=1)[:, 1:k+1]  # Exclude self
            
            overlaps = []
            for i in range(X.shape[0]):
                overlap = len(set(nn_X[i]) & set(nn_Y[i])) / float(k)
                overlaps.append(overlap)
            
            overlaps = np.array(overlaps)
            return np.mean(overlaps), np.percentile(overlaps, [25, 50, 75])
        
        mean_overlap, quartiles = knn_overlap(objective_space_std, embedding_space, k)
        
        # Pareto-like kNN overlap
        dist_obj = pairwise_distances(objective_space_std)
        dist_emb = pairwise_distances(embedding_space)
        
        nn_obj = np.argsort(dist_obj, axis=1)[:, 1:k+1]
        nn_emb = np.argsort(dist_emb, axis=1)[:, 1:k+1]
        
        pareto_overlaps = []
        for i in range(n_samples):
            overlap = len(set(nn_obj[i]) & set(nn_emb[i])) / float(k)
            pareto_overlaps.append(overlap)
        
        pareto_knn_mean = float(np.mean(pareto_overlaps))
        
        return {
            'trustworthiness': float(trustworthiness_score),
            'knn_overlap_mean': float(mean_overlap),
            'knn_overlap_quartiles': [float(q) for q in quartiles],
            'pareto_knn_overlap_mean': pareto_knn_mean,
            'k_neighbors': k
        }
    except Exception as e:
        return {
            'trustworthiness': float('nan'),
            'knn_overlap_mean': float('nan'),
            'knn_overlap_quartiles': [float('nan')] * 3,
            'pareto_knn_overlap_mean': float('nan'),
            'k_neighbors': k
        }


def compute_procrustes_analysis(embedding_space, objective_space, seed=42):
    """
    Compute Procrustes analysis for shape alignment.
    
    Args:
        embedding_space (np.ndarray): Embedding representations
        objective_space (np.ndarray): Objective space representations
        seed (int): Random seed for PCA
    
    Returns:
        dict: Dictionary containing Procrustes metrics
    """
    try:
        p, q = objective_space.shape[1], embedding_space.shape[1]
        
        # Match dimensionalities using PCA
        if p != q:
            if q > p:
                # Reduce embedding dimensionality to match objective
                embedding_for_proc = PCA(n_components=p, random_state=seed).fit_transform(embedding_space)
                objective_for_proc = objective_space
            else:
                # Reduce objective dimensionality to match embedding
                objective_for_proc = PCA(n_components=q, random_state=seed).fit_transform(objective_space)
                embedding_for_proc = embedding_space
        else:
            objective_for_proc, embedding_for_proc = objective_space, embedding_space
        
        # Procrustes analysis
        # Center the data
        objective_centered = objective_for_proc - np.mean(objective_for_proc, axis=0)
        embedding_centered = embedding_for_proc - np.mean(embedding_for_proc, axis=0)
        
        # Compute the optimal rotation matrix
        R, scale = orthogonal_procrustes(objective_centered, embedding_centered)
        
        # Apply the transformation
        obj_aligned = objective_centered
        emb_aligned = embedding_centered @ R * scale
        
        # Compute disparity (sum of squared differences after alignment)
        disparity = np.sum((obj_aligned - emb_aligned) ** 2) / objective_centered.shape[0]
        
        # Compute correlation after alignment
        dist_emb_aligned = pairwise_distances(emb_aligned)
        dist_obj_aligned = pairwise_distances(obj_aligned)
        
        n = embedding_space.shape[0]
        upper_tri_indices = np.triu_indices(n, k=1)
        
        rho_aligned, p_val_aligned = spearmanr(dist_emb_aligned[upper_tri_indices],
                                             dist_obj_aligned[upper_tri_indices])
        
        return {
            'procrustes_disparity': float(disparity),
            'procrustes_correlation': float(rho_aligned),
            'procrustes_p_value': float(p_val_aligned),
            'aligned_objective': obj_aligned,
            'aligned_embedding': emb_aligned
        }
    except Exception as e:
        return {
            'procrustes_disparity': float('nan'),
            'procrustes_correlation': float('nan'),
            'procrustes_p_value': float('nan'),
            'aligned_objective': None,
            'aligned_embedding': None
        }


def compute_geodesic_metrics(embedding_space, objective_space):
    """
    Compute geodesic distances using minimum spanning tree.
    
    Args:
        embedding_space (np.ndarray): Embedding representations
        objective_space (np.ndarray): Objective space representations
    
    Returns:
        dict: Dictionary containing geodesic metrics
    """
    try:
        # Standardize objective space
        scaler = StandardScaler()
        objective_space_std = scaler.fit_transform(objective_space)
        
        # Compute distance matrices
        dist_embedding = pairwise_distances(embedding_space)
        dist_objective = pairwise_distances(objective_space_std)
        
        def mst_geodesic(distance_matrix):
            """Compute geodesic distances via MST."""
            try:
                mst = minimum_spanning_tree(distance_matrix)
                geodesic_distances = shortest_path(mst, directed=False)
                return geodesic_distances
            except Exception:
                return distance_matrix
        
        # Compute geodesic distances
        geodesic_embedding = mst_geodesic(dist_embedding)
        geodesic_objective = mst_geodesic(dist_objective)
        
        # Correlation of geodesic distances
        n = embedding_space.shape[0]
        upper_tri_indices = np.triu_indices(n, k=1)
        
        rho_geodesic, p_val_geodesic = spearmanr(geodesic_embedding[upper_tri_indices],
                                               geodesic_objective[upper_tri_indices])
        
        return {
            'geodesic_correlation': float(rho_geodesic),
            'geodesic_p_value': float(p_val_geodesic),
            'geodesic_distances_embedding': geodesic_embedding,
            'geodesic_distances_objective': geodesic_objective
        }
    except Exception as e:
        return {
            'geodesic_correlation': float('nan'),
            'geodesic_p_value': float('nan'),
            'geodesic_distances_embedding': None,
            'geodesic_distances_objective': None
        }


def compute_order_correlation(embedding_space, objective_space, k_neighbors=None):
    """
    Compute 1D order correlation using principal directions.
    
    Args:
        embedding_space (np.ndarray): Embedding representations
        objective_space (np.ndarray): Objective space representations
        k_neighbors (int, optional): Number of neighbors for Isomap
    
    Returns:
        dict: Dictionary containing order correlation metrics
    """
    try:
        n_samples = embedding_space.shape[0]
        
        if k_neighbors is None:
            k_neighbors = min(4, max(2, n_samples - 2))
        
        # Standardize objective space
        scaler = StandardScaler()
        objective_space_std = scaler.fit_transform(objective_space)
        
        # Use Isomap to find principal 1D manifold
        iso_objective = Isomap(n_neighbors=k_neighbors, n_components=1)
        iso_embedding = Isomap(n_neighbors=k_neighbors, n_components=1)
        
        obj_1d = iso_objective.fit_transform(objective_space_std).ravel()
        emb_1d = iso_embedding.fit_transform(embedding_space).ravel()
        
        # Allow for reversed direction
        rho_forward = spearmanr(emb_1d, obj_1d)[0]
        rho_reverse = spearmanr(-emb_1d, obj_1d)[0]
        rho_order = max(rho_forward, rho_reverse)
        
        return {
            'order_correlation': float(rho_order),
            'k_neighbors': k_neighbors,
            'objective_1d': obj_1d,
            'embedding_1d': emb_1d
        }
    except Exception as e:
        return {
            'order_correlation': float('nan'),
            'k_neighbors': k_neighbors,
            'objective_1d': None,
            'embedding_1d': None
        }


def compute_shepard_stress(embedding_space, objective_space, save_plot=False, image_dir=None, 
                          embed_dim=None):
    """
    Compute Shepard diagram and stress metric.
    
    Args:
        embedding_space (np.ndarray): Embedding representations
        objective_space (np.ndarray): Objective space representations
        save_plot (bool): Whether to save Shepard diagram
        image_dir (str): Directory to save plot
        embed_dim (int): Embedding dimension for filename
    
    Returns:
        dict: Dictionary containing stress metrics
    """
    try:
        # Standardize objective space
        scaler = StandardScaler()
        objective_space_std = scaler.fit_transform(objective_space)
        
        # Compute distance matrices
        dist_embedding = pairwise_distances(embedding_space)
        dist_objective = pairwise_distances(objective_space_std)
        
        # Get upper triangular distances
        n = embedding_space.shape[0]
        upper_tri_indices = np.triu_indices(n, k=1)
        
        x = dist_objective[upper_tri_indices].astype(np.float64)
        y = dist_embedding[upper_tri_indices].astype(np.float64)
        
        # Compute stress
        alpha = (x * y).sum() / (x * x).sum() if (x * x).sum() > 0 else 0.0
        stress = float(np.sqrt(np.sum((y - alpha * x) ** 2) / (np.sum(y ** 2) + 1e-12)))
        
        # Save Shepard diagram if requested
        plot_path = None
        if save_plot and image_dir:
            os.makedirs(image_dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(x, y, s=12, alpha=0.7)
            
            if x.size > 0:
                xs = np.linspace(float(x.min()), float(x.max()), 50)
                ax.plot(xs, alpha * xs, 'r--', lw=1, label=f'Fit (α={alpha:.3f})')
            
            ax.set_xlabel('Distance in Objective Space')
            ax.set_ylabel('Distance in Embedding Space')
            ax.set_title(f'Shepard Diagram (Stress = {stress:.3f})')
            ax.legend()
            
            embed_suffix = f'_embdim{embed_dim}' if embed_dim else ''
            plot_path = os.path.join(image_dir, f'shepard_diagram{embed_suffix}.pdf')
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close(fig)
        
        return {
            'stress': stress,
            'alpha': alpha,
            'shepard_plot_path': plot_path
        }
    except Exception as e:
        return {
            'stress': float('nan'),
            'alpha': float('nan'),
            'shepard_plot_path': None
        }


def compute_lipschitz_metrics(embedding_space, objective_space, k_neighbors=None, 
                             save_histograms=False, image_dir=None, embed_dim=None):
    """
    Compute Lipschitz constant estimates in both directions.
    
    Args:
        embedding_space (np.ndarray): Embedding representations
        objective_space (np.ndarray): Objective space representations
        k_neighbors (int, optional): Number of neighbors
        save_histograms (bool): Whether to save histograms
        image_dir (str): Directory to save histograms
        embed_dim (int): Embedding dimension for filename
    
    Returns:
        dict: Dictionary containing Lipschitz metrics
    """
    try:
        n_samples = embedding_space.shape[0]
        
        if k_neighbors is None:
            k_neighbors = min(3, max(1, n_samples - 3))
        
        # Standardize objective space
        scaler = StandardScaler()
        objective_space_std = scaler.fit_transform(objective_space)
        
        eps, min_sep = 1e-12, 1e-6
        
        if k_neighbors >= 1 and n_samples > 1:
            # Objective -> Embedding direction
            nn_obj = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(objective_space_std)
            distances_obj, indices_obj = nn_obj.kneighbors(objective_space_std)
            distances_obj = distances_obj[:, 1:]  # Exclude self
            indices_obj = indices_obj[:, 1:]      # Exclude self
            
            # Compute embedding distances to objective neighbors
            embedding_distances_to_obj_neighbors = np.linalg.norm(
                embedding_space[:, None, :] - embedding_space[indices_obj], axis=2
            )
            
            # Compute Lipschitz ratios (objective -> embedding)
            mask_obj = distances_obj > min_sep
            ratios_obj_to_emb = np.full_like(embedding_distances_to_obj_neighbors, np.nan, dtype=np.float64)
            ratios_obj_to_emb[mask_obj] = (embedding_distances_to_obj_neighbors[mask_obj] / 
                                          (distances_obj[mask_obj] + eps))
            
            lipschitz_obj_to_emb_median = float(np.nanmedian(ratios_obj_to_emb))
            lipschitz_obj_to_emb_p90 = float(np.nanpercentile(ratios_obj_to_emb, 90))
            
            # Embedding -> Objective direction
            nn_emb = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(embedding_space)
            distances_emb, indices_emb = nn_emb.kneighbors(embedding_space)
            distances_emb = distances_emb[:, 1:]  # Exclude self
            indices_emb = indices_emb[:, 1:]      # Exclude self
            
            # Compute objective distances to embedding neighbors
            objective_distances_to_emb_neighbors = np.linalg.norm(
                objective_space_std[:, None, :] - objective_space_std[indices_emb], axis=2
            )
            
            # Compute Lipschitz ratios (embedding -> objective)
            mask_emb = distances_emb > min_sep
            ratios_emb_to_obj = np.full_like(objective_distances_to_emb_neighbors, np.nan, dtype=np.float64)
            ratios_emb_to_obj[mask_emb] = (objective_distances_to_emb_neighbors[mask_emb] / 
                                          (distances_emb[mask_emb] + eps))
            
            lipschitz_emb_to_obj_median = float(np.nanmedian(ratios_emb_to_obj))
            lipschitz_emb_to_obj_p90 = float(np.nanpercentile(ratios_emb_to_obj, 90))
            
            # Save histograms if requested
            def save_lipschitz_histogram(data, name):
                if not save_histograms or not image_dir:
                    return None
                
                finite_data = data[np.isfinite(data)]
                if finite_data.size == 0:
                    return None
                
                os.makedirs(image_dir, exist_ok=True)
                fig, ax = plt.subplots(figsize=(5, 3.2))
                sns.histplot(finite_data, bins=40, stat='density', color='C1', alpha=0.85, ax=ax)
                
                # Use log scale if range is very large
                if np.max(finite_data) / max(np.median(finite_data), 1e-9) > 1e3:
                    ax.set_xscale('log')
                
                ax.set_title(f'Lipschitz {name}')
                ax.set_xlabel('Lipschitz Ratio')
                ax.set_ylabel('Density')
                
                embed_suffix = f'_embdim{embed_dim}' if embed_dim else ''
                output_path = os.path.join(image_dir, f'lipschitz_{name}{embed_suffix}.pdf')
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close(fig)
                return output_path
            
            # Take median across neighbors for each point
            median_ratios_obj_to_emb = np.nanmedian(ratios_obj_to_emb, axis=1)
            median_ratios_emb_to_obj = np.nanmedian(ratios_emb_to_obj, axis=1)
            
            obj_to_emb_hist_path = save_lipschitz_histogram(median_ratios_obj_to_emb, 'obj_to_emb')
            emb_to_obj_hist_path = save_lipschitz_histogram(median_ratios_emb_to_obj, 'emb_to_obj')
            
        else:
            lipschitz_obj_to_emb_median = lipschitz_obj_to_emb_p90 = float('nan')
            lipschitz_emb_to_obj_median = lipschitz_emb_to_obj_p90 = float('nan')
            obj_to_emb_hist_path = emb_to_obj_hist_path = None
        
        return {
            'lipschitz_obj_to_emb_median': lipschitz_obj_to_emb_median,
            'lipschitz_obj_to_emb_p90': lipschitz_obj_to_emb_p90,
            'lipschitz_emb_to_obj_median': lipschitz_emb_to_obj_median,
            'lipschitz_emb_to_obj_p90': lipschitz_emb_to_obj_p90,
            'k_neighbors': k_neighbors,
            'obj_to_emb_histogram_path': obj_to_emb_hist_path,
            'emb_to_obj_histogram_path': emb_to_obj_hist_path
        }
    except Exception as e:
        return {
            'lipschitz_obj_to_emb_median': float('nan'),
            'lipschitz_obj_to_emb_p90': float('nan'),
            'lipschitz_emb_to_obj_median': float('nan'),
            'lipschitz_emb_to_obj_p90': float('nan'),
            'k_neighbors': k_neighbors,
            'obj_to_emb_histogram_path': None,
            'emb_to_obj_histogram_path': None
        }


def compute_within_between_distances(embedding_space, labels, pairs_definition=None):
    """
    Compute within/between group distances for specific policy pairs.
    Useful for Deep Sea Treasure environment with known policy pairs.
    
    Args:
        embedding_space (np.ndarray): Embedding representations
        labels (np.ndarray): Policy labels/IDs
        pairs_definition (set, optional): Set of tuples defining which pairs are "within"
    
    Returns:
        dict: Dictionary containing within/between distance metrics
    """
    try:
        if pairs_definition is None:
            # Default for Deep Sea Treasure
            pairs_definition = {(0, 1), (2, 3), (4, 5)}
        
        # Compute distance matrix
        distances = pairwise_distances(embedding_space)
        unique_labels = np.unique(labels)
        n_policies = len(unique_labels)
        
        # Create label mapping
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        
        within_distances = []
        between_distances = []
        
        for i in range(n_policies):
            for j in range(i + 1, n_policies):
                label_a = unique_labels[i]
                label_b = unique_labels[j]
                
                # Check if this pair is defined as "within"
                if ((label_a, label_b) in pairs_definition or 
                    (label_b, label_a) in pairs_definition):
                    within_distances.append(distances[i, j])
                else:
                    between_distances.append(distances[i, j])
        
        mean_within = float(np.mean(within_distances)) if within_distances else float('nan')
        mean_between = float(np.mean(between_distances)) if between_distances else float('nan')
        
        return {
            'mean_within_distance': mean_within,
            'mean_between_distance': mean_between,
            'within_distances': within_distances,
            'between_distances': between_distances,
            'pairs_definition': pairs_definition
        }
    except Exception as e:
        return {
            'mean_within_distance': float('nan'),
            'mean_between_distance': float('nan'),
            'within_distances': [],
            'between_distances': [],
            'pairs_definition': pairs_definition
        }


def compute_all_metrics(embedding_space, objective_space, labels=None, 
                       save_plots=False, image_dir=None, embed_dim=None,
                       dst_pairs=False, seed=42):
    """
    Compute all diagnostic metrics for comparing behavior and objective spaces.
    
    Args:
        embedding_space (np.ndarray): Embedding representations (n_samples, n_features)
        objective_space (np.ndarray): Objective space representations (n_samples, n_objectives)
        labels (np.ndarray, optional): Policy labels for within/between analysis
        save_plots (bool): Whether to save diagnostic plots
        image_dir (str): Directory to save plots
        embed_dim (int): Embedding dimension for filenames
        dst_pairs (bool): Whether to compute Deep Sea Treasure specific metrics
        seed (int): Random seed for reproducibility
    
    Returns:
        dict: Dictionary containing all computed metrics
    """
    np.random.seed(seed)
    
    results = {}
    
    # Basic structure metrics
    results['pca'] = compute_pca_metrics(embedding_space)
    results['intrinsic_dimension'] = compute_intrinsic_dimension(embedding_space)
    
    # Distance correlations
    results['distance_correlations'] = compute_distance_correlations(embedding_space, objective_space)
    
    # Mantel test
    results['mantel'] = compute_mantel_test(
        embedding_space, objective_space, 
        save_histogram=save_plots, image_dir=image_dir
    )
    
    # Neighborhood metrics
    results['neighborhood'] = compute_neighborhood_metrics(embedding_space, objective_space)
    
    # Shape alignment
    results['procrustes'] = compute_procrustes_analysis(embedding_space, objective_space, seed=seed)
    
    # Geodesic metrics
    results['geodesic'] = compute_geodesic_metrics(embedding_space, objective_space)
    
    # Order correlation
    results['order'] = compute_order_correlation(embedding_space, objective_space)
    
    # Shepard stress
    results['shepard'] = compute_shepard_stress(
        embedding_space, objective_space,
        save_plot=save_plots, image_dir=image_dir, embed_dim=embed_dim
    )
    
    # Lipschitz metrics
    results['lipschitz'] = compute_lipschitz_metrics(
        embedding_space, objective_space,
        save_histograms=save_plots, image_dir=image_dir, embed_dim=embed_dim
    )
    
    # Within/between distances (for DST)
    if dst_pairs and labels is not None:
        results['within_between'] = compute_within_between_distances(embedding_space, labels)
    
    return results


def print_metrics_summary(results, embedding_dim=None, n_policies=None):
    """
    Print a compact summary of all computed metrics.
    
    Args:
        results (dict): Results from compute_all_metrics
        embedding_dim (int, optional): Embedding dimension
        n_policies (int, optional): Number of policies
    """
    print(f'\n[Policy-level] diagnostics for emb_dim {embedding_dim} (n_policies={n_policies}):')
    
    # PCA metrics
    pca = results.get('pca', {})
    cum_var = pca.get('cumulative_variance', np.array([]))
    participation_ratio = pca.get('participation_ratio', float('nan'))
    print('  PCA cumulative explained:', cum_var[:8])
    print(f'  Participation ratio = {participation_ratio:.3f}')
    
    # Intrinsic dimension
    intrinsic = results.get('intrinsic_dimension', {})
    id_mle = intrinsic.get('intrinsic_dimension', float('nan'))
    k_mle = intrinsic.get('k_neighbors', None)
    print(f'  MLE intrinsic dim (k={k_mle}) = {id_mle:.3f}')
    
    # Distance correlations
    dist_corr = results.get('distance_correlations', {})
    spearman_rho = dist_corr.get('spearman_correlation', float('nan'))
    spearman_p = dist_corr.get('spearman_p_value', float('nan'))
    print(f'  Spearman(D_emb, D_obj) = {spearman_rho:.3f} (p~{spearman_p})')
    
    # Mantel test
    mantel = results.get('mantel', {})
    mantel_r = mantel.get('mantel_correlation', float('nan'))
    mantel_p = mantel.get('mantel_p_value', float('nan'))
    print(f'  Mantel (two-sided) spearman r={mantel_r:.3f}, p={mantel_p:.3f}')
    
    # Neighborhood metrics
    neighborhood = results.get('neighborhood', {})
    trustworthiness = neighborhood.get('trustworthiness', float('nan'))
    knn_mean = neighborhood.get('knn_overlap_mean', float('nan'))
    knn_quartiles = neighborhood.get('knn_overlap_quartiles', [float('nan')] * 3)
    pareto_knn = neighborhood.get('pareto_knn_overlap_mean', float('nan'))
    print(f'  Trustworthiness (obj -> emb) = {trustworthiness:.3f}')
    print(f'  kNN overlap mean={knn_mean:.3f}, quartiles={knn_quartiles}; pareto-like kNN mean={pareto_knn:.3f}')
    
    # Procrustes
    procrustes = results.get('procrustes', {})
    disparity = procrustes.get('procrustes_disparity', float('nan'))
    proc_corr = procrustes.get('procrustes_correlation', float('nan'))
    proc_p = procrustes.get('procrustes_p_value', float('nan'))
    print(f'  Procrustes disparity={disparity:.3f}; Spearman after Procrustes={proc_corr:.3f} (p~{proc_p})')
    
    # Geodesic
    geodesic = results.get('geodesic', {})
    geo_corr = geodesic.get('geodesic_correlation', float('nan'))
    geo_p = geodesic.get('geodesic_p_value', float('nan'))
    print(f'  Geodesic (MST) Spearman={geo_corr:.3f} (p~{geo_p})')
    
    # Order correlation
    order = results.get('order', {})
    order_corr = order.get('order_correlation', float('nan'))
    print(f'  1D order Spearman (PCA1)={order_corr:.3f}')
    
    # Shepard stress
    shepard = results.get('shepard', {})
    stress = shepard.get('stress', float('nan'))
    shepard_path = shepard.get('shepard_plot_path', None)
    print(f'  Shepard stress (centroids)={stress:.3f}; saved={shepard_path}')
    
    # Within/between (if available)
    if 'within_between' in results:
        wb = results['within_between']
        mean_within = wb.get('mean_within_distance', float('nan'))
        mean_between = wb.get('mean_between_distance', float('nan'))
        print(f'  Within/Between means (policy-level): within={mean_within:.4f}, between={mean_between:.4f}')
    
    # Lipschitz
    lipschitz = results.get('lipschitz', {})
    lip_o2e_med = lipschitz.get('lipschitz_obj_to_emb_median', float('nan'))
    lip_o2e_p90 = lipschitz.get('lipschitz_obj_to_emb_p90', float('nan'))
    lip_e2o_med = lipschitz.get('lipschitz_emb_to_obj_median', float('nan'))
    lip_e2o_p90 = lipschitz.get('lipschitz_emb_to_obj_p90', float('nan'))
    print(f'  Lipschitz obj->emb median/p90: {lip_o2e_med} / {lip_o2e_p90}')
    print(f'  Lipschitz emb->obj median/p90: {lip_e2o_med} / {lip_e2o_p90}')




if __name__ == "__main__":
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

    metrics = compute_all_metrics(all_gt_emb, all_returns)
    print_metrics_summary(metrics)

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

    metrics_concave = compute_all_metrics(all_gt_emb_concave, all_returns_concave)
    print_metrics_summary(metrics_concave)
