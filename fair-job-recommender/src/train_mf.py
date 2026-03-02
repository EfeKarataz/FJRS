from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from .utils import ensure_dir, save_json

def train_implicit_mf(interactions: pd.DataFrame, n_users: int, n_items: int, k: int = 32,
                      epochs: int = 15, lr: float = 0.05, reg: float = 0.01, seed: int = 42):
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 0.1, size=(n_users, k))
    V = rng.normal(0, 0.1, size=(n_items, k))

    user = interactions["user_id"].to_numpy(dtype=int)
    item = interactions["job_id"].to_numpy(dtype=int)
    y = interactions["y"].to_numpy(dtype=float)

    for ep in range(epochs):
        idx = rng.permutation(len(interactions))
        user_sh = user[idx]
        item_sh = item[idx]
        y_sh = y[idx]

        total_loss = 0.0
        for u, i, r in tqdm(zip(user_sh, item_sh, y_sh), total=len(y_sh), desc=f"epoch {ep+1}/{epochs}"):
            pred = U[u] @ V[i]
            err = r - pred
            total_loss += err*err + reg*(np.sum(U[u]**2) + np.sum(V[i]**2))

            # SGD updates
            u_old = U[u].copy()
            U[u] += lr * (err * V[i] - reg * U[u])
            V[i] += lr * (err * u_old - reg * V[i])

        total_loss /= max(len(y_sh), 1)
        print(f"epoch {ep+1}: loss={total_loss:.6f}")

    return U, V

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/sim")
    ap.add_argument("--out_dir", type=str, default="outputs/baseline")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--reg", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    users = pd.read_parquet(data_dir / "users.parquet")
    jobs = pd.read_parquet(data_dir / "jobs.parquet")
    inter = pd.read_parquet(data_dir / "interactions.parquet")

    n_users = users["user_id"].max() + 1
    n_items = jobs["job_id"].max() + 1

    U, V = train_implicit_mf(inter, n_users, n_items, k=args.k, epochs=args.epochs, lr=args.lr, reg=args.reg, seed=args.seed)

    out = ensure_dir(args.out_dir)
    np.save(out / "user_factors.npy", U)
    np.save(out / "item_factors.npy", V)
    save_json({
        "k": args.k, "epochs": args.epochs, "lr": args.lr, "reg": args.reg, "seed": args.seed,
        "data_dir": str(data_dir)
    }, out / "config.json")
    print(f"Saved model to {out}")

if __name__ == "__main__":
    main()
