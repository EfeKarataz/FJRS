from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from .utils import ensure_dir, save_json
from .recommend import recommend_topk
from .metrics import exposure_to_condition

def greedy_fair_rerank(recs: dict[int, list[tuple[int, float]]],
                       users: pd.DataFrame,
                       high_opportunity: np.ndarray,
                       topk: int,
                       alpha: float,
                       beta: float) -> dict[int, list[tuple[int, float]]]:
    # Objective: alpha * relevance - beta * unfairness_increment
    # Here unfairness_increment penalizes recommending high-opportunity items to group A if group B is under-exposed (and vice versa).
    # Simple global parity target: match mean exposure between groups.
    out: dict[int, list[tuple[int, float]]] = {}
    group = users.set_index("user_id")["group"].to_dict()

    # First pass: compute baseline exposure means
    expA, expB, nA, nB = 0.0, 0.0, 0, 0
    for u, items in recs.items():
        lst = [i for i, _ in items]
        e = exposure_to_condition(lst, high_opportunity, topk)
        if group[u] == 0:
            expA += e; nA += 1
        else:
            expB += e; nB += 1
    meanA = expA / max(nA, 1)
    meanB = expB / max(nB, 1)

    # Target: bring means closer by steering high-opportunity exposure towards the under-exposed group
    underexposed_group = 0 if meanA < meanB else 1

    for u, items in recs.items():
        candidates = items[:]  # list of (item_id, relevance)
        chosen: list[tuple[int, float]] = []
        chosen_ids: set[int] = set()

        for pos in range(topk):
            best = None
            best_score = -1e18
            for i, rel in candidates:
                if i in chosen_ids:
                    continue
                unfair = 0.0
                if high_opportunity[i]:
                    # If this user's group is the underexposed group, reward; else penalize.
                    unfair = -1.0 if group[u] == underexposed_group else 1.0
                score = alpha * rel - beta * unfair
                if score > best_score:
                    best_score = score
                    best = (i, rel)
            if best is None:
                break
            chosen.append(best)
            chosen_ids.add(best[0])

        out[u] = chosen
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/sim")
    ap.add_argument("--model_dir", type=str, default="outputs/baseline")
    ap.add_argument("--out_dir", type=str, default="outputs/fair")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.6)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    users = pd.read_parquet(data_dir / "users.parquet")
    jobs = pd.read_parquet(data_dir / "jobs.parquet")
    inter = pd.read_parquet(data_dir / "interactions.parquet")

    # seen items
    seen = inter.groupby("user_id")["job_id"].apply(lambda s: set(map(int, s.tolist()))).to_dict()

    U = np.load(Path(args.model_dir) / "user_factors.npy")
    V = np.load(Path(args.model_dir) / "item_factors.npy")

    base_recs = recommend_topk(U, V, seen=seen, topk=max(args.topk, 50))  # generate more candidates than topk
    high = jobs.sort_values("job_id")["high_opportunity"].to_numpy(dtype=bool)

    fair_recs = greedy_fair_rerank(base_recs, users, high, topk=args.topk, alpha=args.alpha, beta=args.beta)

    out = ensure_dir(args.out_dir)
    np.save(out / "user_factors.npy", U)
    np.save(out / "item_factors.npy", V)
    save_json({"topk": args.topk, "alpha": args.alpha, "beta": args.beta, "data_dir": str(data_dir), "model_dir": args.model_dir}, out / "rerank_config.json")

    # Save recs for evaluation/inspection
    rows = []
    for u, items in fair_recs.items():
        for rank, (i, rel) in enumerate(items, start=1):
            rows.append((u, i, rank, rel))
    pd.DataFrame(rows, columns=["user_id", "job_id", "rank", "relevance"]).to_parquet(out / "recommendations.parquet", index=False)

    print(f"Saved fair recommendations to {out}")

if __name__ == "__main__":
    main()
