import numpy as np


def calculate_hit_rate_at_n(relevance_ndarray, n):
    """
    Calculate Hit Rate (HR) at N from a NumPy ndarray of relevance scores.

    Parameters:
    relevance_ndarray (np.ndarray): A 2D NumPy array where each row is a list of relevance scores (0 or 1) for a user's items.
    n (int): The number of top recommendations to consider.

    Returns:
    float: The Hit Rate at N.
    """
    hits = 0
    total_users = relevance_ndarray.shape[0]

    # Iterate through each user's relevance scores
    for relevance_list in relevance_ndarray:
        # Check if there is at least one relevant item in the top N
        if np.sum(relevance_list[:n]) > 0:
            hits += 1

    # Calculate Hit Rate
    hit_rate = hits / total_users
    return hit_rate


def calculate_mrr_from_relevance_ndarray(relevance_ndarray):
    """
    Calculate Mean Reciprocal Rank (MRR) from a NumPy ndarray of relevance scores.

    Parameters:
    relevance_ndarray (np.ndarray): A 2D NumPy array where each row is a list of relevance scores (0 or 1) for a user's items.

    Returns:
    float: The Mean Reciprocal Rank.
    """
    reciprocal_ranks = []

    # Iterate through each user's relevance scores
    for relevance_list in relevance_ndarray:
        relevant_indices = np.where(relevance_list == 1)[0]  # Find indices of relevant items
        if relevant_indices.size > 0:
            rank = relevant_indices[0] + 1  # Get the rank (1-based index)
            reciprocal_ranks.append(1 / rank)

    # Calculate MRR
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    return mrr