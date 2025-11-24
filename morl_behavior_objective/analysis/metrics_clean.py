"""Script to compute metrics for evaluating the quality of embeddings using well-established libraries."""

from reptrix import alpha
from reptrix import rankme
import torch
from zadu import zadu


def compute_reptrix_metrics(embeddings):
    """Compute RepTrix metrics for the given embeddings."""
    embeddings_tensor = torch.from_numpy(embeddings).float()

    metrics = {}

    metrics["alpha"] = alpha.get_alpha(embeddings_tensor)  # OPTIMUM=1
    print(f"Alpha computed successfully: {metrics['alpha']}")

    metrics["rankme"] = rankme.get_rankme(embeddings_tensor)  # OPTIMUM>1 (the higher the better)
    print(f"RankMe computed successfully: {metrics['rankme']}")

    return metrics


def compute_zadu_metrics(raw_data, embeddings, k=2):
    """Compute ZADU metrics for the given raw data and embeddings."""
    spec = [
        {  # Trustworthiness & Continuity #OPTIMUM=1
            "id": "tnc",
            "params": {"k": k, "return_local": True},
        },
        {  # Mean Relative Rank Errors #OPTIMUM=0
            "id": "mrre",
            "params": {"k": k, "return_local": True},
        },
    ]

    zadu_instance = zadu.ZADU(spec, raw_data)
    result = zadu_instance.measure(embeddings)
    return result
