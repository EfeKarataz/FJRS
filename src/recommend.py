from __future__ import annotations
import numpy as np
import pandas as pd

def score_mf(user_factors: np.ndarray, item_factors: np.ndarray, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
    # dot product for paired ids
    return (user_factors[user_ids] * item_factors[item_ids]).sum(axis=1)

def recommend_topk(user_factors: np.ndarray, item_factors: np.ndarray, seen: dict[int, set[int]], topk: int = 10) -> dict[int, list[tuple[int, float]]]:
    n_users = user_factors.shape[0]
    n_items = item_factors.shape[0]
    recs: dict[int, list[tuple[int, float]]] = {}

    for u in range(n_users):
        scores = user_factors[u] @ item_factors.T  # (n_items,)
        if u in seen:
            scores[list(seen[u])] = -1e9
        top_items = np.argpartition(-scores, kth=min(topk, n_items-1))[:topk]
        top_items = top_items[np.argsort(-scores[top_items])]
        recs[u] = [(int(i), float(scores[i])) for i in top_items]
    return recs
