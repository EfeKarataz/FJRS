from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

from .utils import ensure_dir

@dataclass
class SimConfig:
    seed: int = 42
    n_users: int = 2000
    n_jobs: int = 1500
    n_skills: int = 50
    interactions_per_user: int = 20
    group_share: float = 0.5  # fraction of users in group B
    # bias knobs
    exposure_bias_to_high_opportunity_for_groupB: float = 0.75  # <1 means group B sees fewer high-opportunity jobs
    click_noise: float = 0.15

def simulate(cfg: SimConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)

    # Users: group A/B, education, experience, skills
    user_ids = np.arange(cfg.n_users)
    group = (rng.random(cfg.n_users) < cfg.group_share).astype(int)  # 0=A, 1=B
    education = rng.integers(0, 4, size=cfg.n_users)  # 0..3
    experience = rng.integers(0, 11, size=cfg.n_users)  # 0..10 years
    user_skills = (rng.random((cfg.n_users, cfg.n_skills)) < 0.10).astype(int)

    users = pd.DataFrame({
        "user_id": user_ids,
        "group": group,  # 0=A, 1=B
        "education": education,
        "experience_years": experience,
    })

    # Jobs: wage/seniority proxy, required skills, occupation group
    job_ids = np.arange(cfg.n_jobs)
    occupation = rng.integers(0, 12, size=cfg.n_jobs)  # 12 coarse occupations
    seniority = rng.integers(0, 5, size=cfg.n_jobs)  # 0..4
    wage = (seniority * 10 + rng.normal(0, 3, size=cfg.n_jobs)).clip(0, None)
    job_skills = (rng.random((cfg.n_jobs, cfg.n_skills)) < 0.10).astype(int)

    high_opportunity = (wage >= np.quantile(wage, 0.75)).astype(int)

    jobs = pd.DataFrame({
        "job_id": job_ids,
        "occupation": occupation,
        "seniority": seniority,
        "wage_proxy": wage,
        "high_opportunity": high_opportunity,
    })

    # True relevance: skill overlap + education/experience fit
    # Compute overlap matrix efficiently by sampling interactions rather than full matrix.
    interactions = []
    for u in user_ids:
        # candidate pool: random subset
        cand = rng.choice(job_ids, size=min(cfg.n_jobs, 400), replace=False)
        overlap = (user_skills[u] & job_skills[cand]).sum(axis=1)

        edu_fit = -np.abs(education[u] - (seniority[cand] // 2))
        exp_fit = -np.abs(experience[u] - (seniority[cand] * 2))

        rel = overlap + 0.5 * edu_fit + 0.2 * exp_fit
        # exposure bias: group B gets downweighted relevance for high-opportunity jobs (simulating structural barrier)
        if users.loc[u, "group"] == 1:
            rel = rel - (1 - cfg.exposure_bias_to_high_opportunity_for_groupB) * (high_opportunity[cand] * 2.0)

        # turn relevance into click probability
        rel_norm = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)
        p_click = (0.1 + 0.9 * rel_norm) * (1 - cfg.click_noise) + cfg.click_noise * rng.random(len(cand))

        # sample interactions_per_user jobs with probabilities
        probs = p_click / (p_click.sum() + 1e-12)
        chosen = rng.choice(cand, size=cfg.interactions_per_user, replace=False, p=probs)
        for j in chosen:
            # implicit feedback strength
            interactions.append((u, int(j), 1))

    interactions_df = pd.DataFrame(interactions, columns=["user_id", "job_id", "y"])

    return users, jobs, interactions_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/sim")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_users", type=int, default=2000)
    ap.add_argument("--n_jobs", type=int, default=1500)
    ap.add_argument("--n_skills", type=int, default=50)
    ap.add_argument("--interactions_per_user", type=int, default=20)
    args = ap.parse_args()

    cfg = SimConfig(
        seed=args.seed,
        n_users=args.n_users,
        n_jobs=args.n_jobs,
        n_skills=args.n_skills,
        interactions_per_user=args.interactions_per_user,
    )

    users, jobs, inter = simulate(cfg)
    out = ensure_dir(args.out_dir)
    users.to_parquet(out / "users.parquet", index=False)
    jobs.to_parquet(out / "jobs.parquet", index=False)
    inter.to_parquet(out / "interactions.parquet", index=False)

    print(f"Wrote: {out/'users.parquet'}")
    print(f"Wrote: {out/'jobs.parquet'}")
    print(f"Wrote: {out/'interactions.parquet'}")

if __name__ == "__main__":
    main()
