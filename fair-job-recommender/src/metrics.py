from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def dcg(rels: np.ndarray) -> float:
    # rels in rank order
    denom = np.log2(np.arange(2, rels.size + 2))
    return float((rels / denom).sum())

def ndcg_at_k(recommended: List[int], relevant_set: set[int], k: int) -> float:
    rec_k = recommended[:k]
    rels = np.array([1.0 if i in relevant_set else 0.0 for i in rec_k], dtype=float)
    ideal = np.sort(rels)[::-1]
    idcg = dcg(ideal)
    return 0.0 if idcg == 0 else dcg(rels) / idcg

def precision_at_k(recommended: List[int], relevant_set: set[int], k: int) -> float:
    rec_k = recommended[:k]
    if not rec_k:
        return 0.0
    hits = sum(1 for i in rec_k if i in relevant_set)
    return hits / min(k, len(rec_k))

def recall_at_k(recommended: List[int], relevant_set: set[int], k: int) -> float:
    if not relevant_set:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for i in rec_k if i in relevant_set)
    return hits / len(relevant_set)

def exposure_weights(k: int) -> np.ndarray:
    # position-based exposure: 1/log2(rank+2)
    ranks = np.arange(1, k+1)
    return 1.0 / np.log2(ranks + 1)

def exposure_to_condition(recommended: List[int], condition: np.ndarray, k: int) -> float:
    # condition is boolean array indexed by item_id
    rec_k = recommended[:k]
    w = exposure_weights(len(rec_k))
    cond = np.array([1.0 if condition[i] else 0.0 for i in rec_k], dtype=float)
    return float((w * cond).sum())
