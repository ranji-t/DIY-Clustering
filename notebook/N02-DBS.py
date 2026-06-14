import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    # Third party Imports
    import jax
    import marimo as mo
    import jax.numpy as jnp
    from functools import partial
    import plotly.graph_objects as go
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import minmax_scale

    return go, jax, jnp, load_iris, minmax_scale, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **DB Scan**
    """)
    return


@app.cell
def _(jax, jnp):
    def centroid_distance(X: jax.Array, x: jax.Array) -> jax.Array:
        return jnp.sum(jnp.square(X - x), axis=1)


    def distance_matrix(X1: jax.Array, X2: jax.Array) -> jax.Array:
        return jax.jit(jax.vmap(centroid_distance, in_axes=(None, 0), out_axes=1))(
            X1, X2
        )


    @jax.jit
    def propagate(core_adjacency: jax.Array, labels: jax.Array):
        # Get the Index
        min_index = jnp.where(core_adjacency, labels[None, :], jnp.inf).min(axis=1)
        # Update the labels
        new_labels = jnp.minimum(labels, min_index)
        # Any point changed at all
        changed = jnp.any(new_labels != labels)
        # return the data new labesl & change ins labels
        return changed, new_labels


    def dbscan(
        pairwise_dist: jax.Array,
        eps: float,
        min_samples: int = 3,
        n_iter: int = 100,
    ):
        # Construction of Adjancency Matrix
        adjacency = pairwise_dist <= (eps**2)

        # Neighbout Counts
        neighbor_counts = adjacency.sum(axis=1)
        is_core = neighbor_counts >= min_samples

        # Adjacency
        core_adjacency = adjacency * jnp.einsum("i, j -> ij", is_core, is_core)

        # Initial Lables
        init_labels = jnp.where(
            is_core,
            jnp.arange(core_adjacency.shape[1], dtype=jnp.float32),
            jnp.inf,
        )

        # Iteration through the label updates
        actual_n_iters, _, labels = jax.lax.while_loop(
            lambda carry: (carry[0] < n_iter) & carry[1],
            lambda carry: (
                carry[0] + 1,
                *propagate(core_adjacency=core_adjacency, labels=carry[2]),
            ),
            (0, True, init_labels),
        )

        # Give proper labes
        proper_labels = jnp.searchsorted(jnp.unique(labels), labels, side="right")

        # Put outlier as -1 label
        proper_labels = jnp.where(jnp.isinf(labels), -1, proper_labels)

        # Return
        return proper_labels, actual_n_iters


    def compute_inertia(
        X: jax.Array, centroids: jax.Array, labels: jax.Array
    ) -> jax.Array:
        return jnp.sum(jnp.sum(jnp.square(X - centroids[labels]), axis=1))


    def silhouette_score(labels: jax.Array, X: jax.Array, k: int):
        # One hot encoded cluster labels
        target = jax.nn.one_hot(labels, num_classes=k)

        mean_point_to_cluster_distances = (
            # Affinity Matrix and One hot target
            jnp.einsum("nm, mk -> nk", distance_matrix(X, X), target)
            # Count of points in each cluster
            / target.sum(axis=0)
        )

        # The average betweek self cluster
        intra_cluster_distance = jnp.where(
            target, mean_point_to_cluster_distances, jnp.inf
        ).min(axis=1)

        # Average inter cluster distance to the nearest cluster
        nearest_inter_cluster_distance = jnp.where(
            1 - target, mean_point_to_cluster_distances, jnp.inf
        ).min(axis=1)

        # Silhouette Score
        silhouette_score = (
            nearest_inter_cluster_distance - intra_cluster_distance
        ) / jnp.maximum(
            jnp.maximum(nearest_inter_cluster_distance, intra_cluster_distance),
            1e-9,
        )

        # Mean Silluette Score
        return silhouette_score.mean()

    return dbscan, distance_matrix


@app.cell
def _(jnp, load_iris, minmax_scale):
    # Load the data
    X, y = load_iris(return_X_y=True, as_frame=True)

    # Convert to array
    X_arr = jnp.array(minmax_scale(X.to_numpy(), axis=0))[:, 2:]
    return (X_arr,)


@app.cell
def _(X_arr, distance_matrix):
    # Compute Self Dist Array
    pairwise_dist = distance_matrix(X_arr, X_arr)
    return (pairwise_dist,)


@app.cell
def _():
    # DBScan parameters
    eps = 0.08
    min_samples = 10

    # Num inertations
    n_iter: int = 100
    return eps, min_samples, n_iter


@app.cell
def _(dbscan, eps, min_samples, n_iter: int, pairwise_dist):
    labels, n_inter_stop = dbscan(
        pairwise_dist=pairwise_dist,
        eps=eps,
        min_samples=min_samples,
        n_iter=n_iter,
    )

    n_inter_stop
    return (labels,)


@app.cell
def _(X_arr, eps, go, jnp, labels, min_samples):
    # Get final cluster assignments
    _labels = labels
    _feat_x, _feat_y = 0, 1  # petal length, petal width

    _fig = go.Figure()

    # Plot each cluster
    for _c, _v in enumerate(jnp.unique(_labels)):
        _mask = _labels == _v
        _fig.add_trace(
            go.Scatter(
                x=X_arr[_mask, _feat_x],
                y=X_arr[_mask, _feat_y],
                mode="markers",
                name=f"Cluster {_v}",
                marker=dict(size=8, opacity=0.7),
            )
        )

    _fig.update_layout(
        title=dict(
            text=f"DB-Scan Clusters (Iris), eps = {eps}, min_samples={min_samples}",
            x=0.5,
        ),
        xaxis_title="Petal Length (scaled)",
        yaxis_title="Petal Width (scaled)",
        legend_title="Legend",
        template="plotly_white",
        width=700,
    )

    _fig.show()
    return


if __name__ == "__main__":
    app.run()
