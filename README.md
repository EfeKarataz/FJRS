# FJRS — Fair Job Recommender System

A course project for **Recommender Systems in Society** that investigates **algorithmic fairness in job recommendation**. We examine whether a standard collaborative filtering model treats different groups of job seekers equitably — specifically whether certain groups receive systematically less exposure to high-opportunity jobs in ranked recommendation lists.

---

## Key Concepts

| Concept | Description |
|---|---|
| **Implicit feedback** | An application is treated as a binary positive signal (`rating = 1`). Non-applications are unknown, not negative. |
| **Groups** | Derived from the `ManagedOthers` field: **Group A** (managed others — proxy for experienced) vs **Group B** (has not). |
| **Job tier** | `high_opportunity` (senior, lead, manager, engineer, analyst, …) vs `standard` (all other roles). |
| **Exposure** | Position-based discount: `exposure(rank) = 1 / log₂(rank + 2)`. Fairness = gap in average exposure to high-opportunity jobs between groups. |

---

## Project Structure

```
FJRS/
├── data/                           # Cleaned dataset — not committed
│   ├── interactions.parquet
│   ├── jobs.parquet
│   ├── users.parquet
│   ├── user_history_agg.parquet
│   └── meta.json
├── notebooks/
│   └── FJRS Notebook.ipynb         # Main pipeline: train, evaluate, re-rank
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Pipeline Overview

The full analysis runs in a single notebook: **`notebooks/FJRS Notebook.ipynb`**

| Step | Description |
|---|---|
| 0 | Setup & imports |
| 1 | Load cleaned data (`interactions.parquet`, `users.parquet`, `jobs.parquet`) |
| 2 | Train / test split (80/20) & sparse interaction matrix |
| 3 | Baseline model: Implicit Matrix Factorization (BPR-SGD) |
| 4 | Generate baseline recommendations |
| 5 | Accuracy metrics (Precision@K, Recall@K, nDCG@K) |
| 6 | Exposure fairness metrics (position-based exposure parity) |
| 7 | Fairness-aware re-ranking (greedy, tunable α/β) |
| 8 | Evaluate fair model |
| 9 | Compare baseline vs. fair model |
| 10 | Trade-off frontier: sweep β |
| 11 | Conclusion |

---

## Quickstart

### 1. Create environment & install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Prepare the data

Place the cleaned dataset files (`interactions.parquet`, `users.parquet`, `jobs.parquet`, `user_history_agg.parquet`, `meta.json`) in the `data/` directory.

### 3. Run the pipeline

Open `notebooks/FJRS Notebook.ipynb` and run all cells top-to-bottom.

> **Colab users:** upload the data files when prompted or mount your Google Drive.  
> **Jupyter / JupyterLab:** ensure the `data/` directory is populated before running.

---

## Dataset

The cleaned dataset in `data/` contains:

| File | Description |
|---|---|
| `interactions.parquet` | One row per (user, job) application — columns: `user_id`, `job_id`, `rating` |
| `users.parquet` | One row per user — columns: `user_id`, `group` (`A` or `B`) |
| `jobs.parquet` | One row per job — columns: `job_id`, `tier` (`high_opportunity` or `standard`) |
| `user_history_agg.parquet` | Aggregated past job titles per user (for future content-based extensions) |
| `meta.json` | Scalars `N_USERS` and `N_JOBS` |

**Scale:** ~308,000 users · ~350,000 jobs · ~1.4M interactions.

---

## Model

We train a latent-factor model using **Bayesian Personalized Ranking (BPR)** with stochastic gradient descent.

For each observed positive interaction `(u, i)`, a random unobserved item `j` is sampled. The model learns to rank `i` above `j`:

```
L = −log σ(x_ui − x_uj) + λ · ‖θ‖²
```

### Default hyperparameters

| Parameter | Value |
|---|---|
| `K_FACTORS` | 32 |
| `EPOCHS` | 15 |
| `LR` | 0.05 |
| `REG` | 0.01 |
| `SAMPLE_SIZE` | 50,000 interactions |

---

## Metrics

### Accuracy (consumer utility)

- Precision@K
- Recall@K
- nDCG@K

### Exposure (societal utility)

- **Exposure to high-opportunity jobs** per user, averaged by group
- **Exposure Parity Gap:** `|mean_exposure_A − mean_exposure_B|`

---

## Fairness-Aware Re-ranking

A greedy post-processing algorithm re-orders each user's candidate list:

```
score(item) = α × relevance(item) − β × exposure_penalty(item)
```

| Parameter | Default | Effect |
|---|---|---|
| `alpha` | 1.0 | Relevance weight |
| `beta` | 0.2 | Fairness strength — higher = more fairness, less accuracy |

The trade-off frontier (sweeping β from 0 to 0.5) shows achievable accuracy/fairness combinations.

---

## Results

- BPR loss decreases steadily across 15 epochs, confirming the model learns meaningful ranking relationships.
- Accuracy metrics are low but expected given extreme data sparsity.
- Baseline exposure gap between Group A and Group B is very small (~0.002).
- Fairness re-ranking has minimal impact on accuracy while maintaining exposure parity.

---

## Requirements

- Python ≥ 3.10
- numpy ≥ 1.26
- pandas ≥ 2.2
- scikit-learn ≥ 1.4
- scipy ≥ 1.11
- pyarrow ≥ 15.0
- matplotlib ≥ 3.8
- tqdm ≥ 4.66

---

## License

MIT — see [LICENSE](LICENSE) for details.
