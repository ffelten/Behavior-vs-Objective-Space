"""Script to compute metrics for evaluating the quality of embeddings using well-established libraries."""

from zadu import zadu


def compute_zadu_metrics(raw_data, embeddings, k=2) -> dict:
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
    return zadu_instance.measure(embeddings)
