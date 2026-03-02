from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from .utils import ensure_dir, save_json
from .recommend import recommend_topk
from .metrics import precision_at_k, recall_at_k, ndcg_at_k, exposure_to_condition

def leave_last_out(interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # sort per user and take last as test
    interactions = interactions.copy()
    interactions["idx"] = interactions.groupby("user_id").cumcount()
    last_idx = interactions.groupby("user_id")["idx"].transform("max")
    test = interactions[interactions["idx"] == last_idx].drop(columns=["idx"])
    train = interactions[interactions["idx"] != last_idx].drop(columns=["idx"])
    return train, test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/sim")
    ap.add_argument("--model_dir", type=str, default="outputs/baseline")
    ap.add_argument("--out_dir", type=str, default="outputs/eval")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    users = pd.read_parquet(data_dir / "users.parquet")
    jobs = pd.read_parquet(data_dir / "jobs.parquet")
    inter = pd.read_parquet(data_dir / "interactions.parquet")

    train, test = leave_last_out(inter)

    seen = train.groupby("user_id")["job_id"].apply(lambda s: set(map(int, s.tolist()))).to_dict()
    relevant = test.groupby("user_id")["job_id"].apply(lambda s: set(map(int, s.tolist()))).to_dict()

    U = np.load(Path(args.model_dir) / "user_factors.npy")
    V = np.load(Path(args.model_dir) / "item_factors.npy")

    # If precomputed recommendations exist, use them; else compute MF recs.
    rec_path = Path(args.model_dir) / "recommendations.parquet"
    if rec_path.exists():
        rec_df = pd.read_parquet(rec_path)
        recs = rec_df.sort_values(["user_id","rank"]).groupby("user_id")["job_id"].apply(list).to_dict()
    else:
        recs_scored = recommend_topk(U, V, seen=seen, topk=args.topk)
        recs = {u: [i for i, _ in items] for u, items in recs_scored.items()}

    group = users.set_index("user_id")["group"].to_dict()
    high = jobs.sort_values("job_id")["high_opportunity"].to_numpy(dtype=bool)

    # Metrics
    rows = []
    expA = expB = 0.0
    nA = nB = 0
    for u, rec_list in recs.items():
        rel_set = relevant.get(u, set())
        p = precision_at_k(rec_list, rel_set, args.topk)
        r = recall_at_k(rec_list, rel_set, args.topk)
        n = ndcg_at_k(rec_list, rel_set, args.topk)
        e = exposure_to_condition(rec_list, high, args.topk)

        rows.append((u, group.get(u, -1), p, r, n, e))
        if group.get(u, 0) == 0:
            expA += e; nA += 1
        else:
            expB += e; nB += 1

    df = pd.DataFrame(rows, columns=["user_id","group","precision","recall","ndcg","exposure_highopp"])
    overall = {
        "topk": args.topk,
        "precision_mean": float(df["precision"].mean()),
        "recall_mean": float(df["recall"].mean()),
        "ndcg_mean": float(df["ndcg"].mean()),
        "exposure_highopp_mean": float(df["exposure_highopp"].mean()),
    }
    meanA = expA / max(nA, 1)
    meanB = expB / max(nB, 1)
    fairness = {
        "exposure_highopp_groupA_mean": float(meanA),
        "exposure_highopp_groupB_mean": float(meanB),
        "exposure_parity_gap": float(abs(meanA - meanB)),
    }

    out = ensure_dir(args.out_dir)
    df.to_parquet(out / "per_user_metrics.parquet", index=False)
    save_json({"overall": overall, "fairness": fairness}, out / "metrics.json")
    print("Overall:", overall)
    print("Fairness:", fairness)
    print(f"Saved to {out}")

if __name__ == "__main__":
    main()
