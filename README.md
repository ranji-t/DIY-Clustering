# DIY Clustering

Clustering algorithms written from scratch — no sklearn wrappers, no black boxes. Each notebook picks one algorithm and implements it as fast as possible using JAX: JIT compilation, vectorized ops, and hardware acceleration.

This is a learning-by-doing project. If you're curious about *how* these algorithms actually work under the hood, or want to see what happens when you throw JAX at them, this is for you.

---

## Why?

Most ML tutorials hand you `sklearn.cluster.KMeans().fit(X)` and call it a day. That's fine for production, but it tells you nothing about what's actually happening. This project goes the other way — start from math, write every operation explicitly, then optimize it until it's fast.

The constraint: **no calling a library's clustering implementation**. Distance functions, centroid updates, convergence loops — all written by hand.

---

## Stack

- **[JAX](https://github.com/google/jax)** — JIT compilation, `vmap` for batched ops, `lax` control flow
- **[Marimo](https://marimo.io)** — reactive notebooks that are actually Python files
- **[Polars](https://pola.rs)** — fast dataframes when needed
- **Plotly** — interactive visualizations

---

## Notebooks

| # | Algorithm | Key Ideas |
|---|-----------|-----------|
| [N01](notebook/N01-KNN.py) | K-Means | `vmap` over centroids, `segment_sum` for updates, `lax.while_loop` for iteration |
| [N01](notebook/N01-KNN.py) | K-Medoids | Medoid selection via `einsum` affinity, swap-based updates, robust to outliers |

### Evaluation Metrics

Both notebooks compute:
- **Inertia** — sum of squared distances from each point to its assigned centroid
- **Silhouette Score** — vectorized via `einsum` on the full affinity matrix; no Python loops

---

## Running a Notebook

```bash
# Install uv if you don't have it
pip install uv

# Install dependencies
uv sync

# Run a notebook
uv run marimo edit notebook/N01-KNN.py
```

---

## What's Next

- DBSCAN — density-based, no need to pick `k` upfront
- Gaussian Mixture Models — soft assignments with EM
- Spectral Clustering — graph Laplacian approach
- Mini-batch K-Means — stochastic updates for large data
- Hierarchical Clustering — dendrogram-based, no `k` required upfront

---

## Contributing

Found a faster way to do something? Open a PR. The goal is correctness first, then speed — as long as both are explained.
