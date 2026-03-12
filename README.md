# Fair Job Recommender — Exposure Inequality Study

A course project for **Recommender Systems in Society**.

This repo implements an end-to-end pipeline to:
1) Data,
2) train a **baseline collaborative filtering** model (implicit MF via SGD),
3) compute **accuracy + exposure fairness** metrics,
4) apply a **fairness-aware re-ranking** step and compare trade-offs.

> Goal: treat *ranking exposure* as a scarce resource and measure whether different user groups receive different exposure to “high-opportunity” jobs (e.g., high-wage / senior roles), and how mitigation changes accuracy.

---

## Quickstart

### 1) Create env + install deps
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Clean data
```bash
python -m src.simulate_data --out_dir data/sim --seed 42 --n_users 2000 --n_jobs 1500
```

### 3) Train baseline model
```bash
python -m src.train_mf --data_dir data/sim --out_dir outputs/baseline --epochs 15 --k 32 --lr 0.05 --reg 0.01
```

### 4) Evaluate baseline
```bash
python -m src.evaluate --data_dir data/sim --model_dir outputs/baseline --out_dir outputs/baseline_eval --topk 10
```

### 5) Fairness-aware re-ranking + evaluate
```bash
python -m src.rerank_fair --data_dir data/sim --model_dir outputs/baseline --out_dir outputs/fair --topk 10 --alpha 1.0 --beta 0.6
python -m src.evaluate --data_dir data/sim --model_dir outputs/fair --out_dir outputs/fair_eval --topk 10
```

---

## Repo structure
```
fair-job-recommender/
  data/                 # generated locally (not committed)
  outputs/              # results, plots, metric json (not committed)
  notebooks/
    01_tradeoffs.ipynb  # optional (create later)
  src/
    simulate_data.py
    train_mf.py
    recommend.py
    rerank_fair.py
    metrics.py
    evaluate.py
    utils.py
  .gitignore
  requirements.txt
  LICENSE
  README.md
```

---

## What we measure

### Accuracy (consumer utility)
- Precision@K, Recall@K, nDCG@K

### Exposure (societal utility)
We use a position-based exposure weight, default:
`w(rank) = 1 / log2(rank + 2)`.

We compute:
- **Exposure to high-opportunity jobs** per user and average by group
- **Exposure Parity Gap**: `abs(mean_exposure_groupA - mean_exposure_groupB)`
- Optional: concentration measures (Gini / Herfindahl) by job group

---

## Notes
- This is a template: you can extend with better credibility/quality proxies, real datasets, or more advanced mitigation.
- If you want to run this on a real dataset later, keep the same interfaces: `interactions.parquet`, `users.parquet`, `jobs.parquet`.

---

## License
MIT
