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

    return go, jax, jnp, load_iris, minmax_scale, mo, partial


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **KNN**
    """)
    return


@app.cell
def _(jax, jnp, partial):
    def centroid_distance(X: jax.Array, centroids: jax.Array) -> jax.Array:
        return jnp.sum(jnp.square(X - centroids), axis=1)


    def all_centroid_distance(X: jax.Array, centroids: jax.Array) -> jax.Array:
        return jax.jit(jax.vmap(centroid_distance, in_axes=(None, 0), out_axes=1))(
            X, centroids
        )


    @partial(jax.jit, static_argnames=["k"])
    def update_centroid(
        X: jax.Array, centroids: jax.Array, X_ones: jax.Array, k: int
    ) -> jax.Array:
        # centroid numbers
        max_index = all_centroid_distance(X=X, centroids=centroids).argmin(axis=1)

        # updated cetroids
        x_sums = jax.ops.segment_sum(X, max_index, num_segments=k)
        x_counts = jax.ops.segment_sum(X_ones, max_index, num_segments=k)
        new_centoids = jnp.where(
            x_counts > 0, x_sums / jnp.maximum(x_counts, 1), centroids
        )

        # Return
        return new_centoids


    def knn(
        X: jax.Array, k: int = 3, seed: int = 54545, n_iter: int = 50
    ) -> jax.Array:
        # Ones array:
        X_ones = jnp.ones_like(X)

        # Init the centoid
        init_centroid = jax.random.choice(
            jax.random.key(343), X, axis=0, shape=(k,), replace=False
        )

        # Runn Knn Iteration
        _, final_centroid = jax.lax.while_loop(
            lambda arr: arr[0] < n_iter,
            lambda arr: (
                arr[0] + 1,
                update_centroid(X, arr[1], X_ones, k),
            ),
            (0, init_centroid),
        )

        # Return data
        return final_centroid


    def compute_inertia(
        X: jax.Array, centroids: jax.Array, labels: jax.Array
    ) -> jax.Array:
        return jnp.sum(jnp.sum(jnp.square(X - centroids[labels]), axis=1))


    def silhouette_score(labels: jax.Array, X: jax.Array, k: int):
        # One hot encoded cluster labels
        target = jax.nn.one_hot(labels, num_classes=k)

        mean_point_to_cluster_distances = (
            # Affinity Matrix and One hot target
            jnp.einsum("nm, mk -> nk", all_centroid_distance(X, X), target)
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

    return all_centroid_distance, compute_inertia, knn, silhouette_score


@app.cell
def _(jnp, load_iris, minmax_scale):
    # Load the data
    X, y = load_iris(return_X_y=True, as_frame=True)

    # Convert to array
    X_arr = jnp.array(minmax_scale(X.to_numpy(), axis=0))[:, 2:]
    return (X_arr,)


@app.cell
def _(X_arr, knn):
    # N clusters
    k: int = 4

    # Get Centroids
    centroids = knn(X=X_arr, k=k, seed=42342, n_iter=100)
    return centroids, k


@app.cell
def _(centroids):
    centroids
    return


@app.cell
def _(X_arr, all_centroid_distance, centroids, go, k: int):
    # Get final cluster assignments
    _labels = all_centroid_distance(X=X_arr, centroids=centroids).argmin(axis=1)

    _feat_x, _feat_y = 0, 1  # petal length, petal width

    _fig = go.Figure()

    # Plot each cluster
    for _c in range(k):
        _mask = _labels == _c
        _fig.add_trace(
            go.Scatter(
                x=X_arr[_mask, _feat_x],
                y=X_arr[_mask, _feat_y],
                mode="markers",
                name=f"Cluster {_c}",
                marker=dict(size=8, opacity=0.7),
            )
        )

    # Plot centroids
    _fig.add_trace(
        go.Scatter(
            x=centroids[:, _feat_x],
            y=centroids[:, _feat_y],
            mode="markers",
            name="Centroids",
            marker=dict(size=16, symbol="x", color="black", line=dict(width=2)),
        )
    )

    _fig.update_layout(
        title=dict(text="K-Means Clusters (Iris)", x=0.5),
        xaxis_title="Petal Length (scaled)",
        yaxis_title="Petal Width (scaled)",
        legend_title="Legend",
        template="plotly_white",
    )

    _fig.show()
    return


@app.cell
def _(X_arr, all_centroid_distance, centroids, compute_inertia):
    # Inertia
    compute_inertia(
        X=X_arr,
        centroids=centroids,
        labels=all_centroid_distance(X=X_arr, centroids=centroids).argmin(axis=1),
    )
    return


@app.cell
def _(X_arr, all_centroid_distance, centroids, k: int, silhouette_score):
    silhouette_score(
        k=k,
        labels=all_centroid_distance(X=X_arr, centroids=centroids).argmin(axis=1),
        X=X_arr,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **K Medoids**
    """)
    return


@app.cell
def _(all_centroid_distance, jax, jnp, partial):
    @partial(jax.jit, static_argnames=["k"])
    def update_medoids(X: jax.Array, medoids: jax.Array, k: int) -> jax.Array:
        # Get Labels
        labels = all_centroid_distance(X=X, centroids=medoids).argmin(axis=1)

        # Masks
        M = jax.nn.one_hot(labels, num_classes=k)

        # Calulate The distance
        self_distance = all_centroid_distance(X=X, centroids=X)

        # New Lables for the cadindates
        medoid_idx = jnp.where(
            M == 1, jnp.einsum("nm, mk -> nk", self_distance, M), jnp.inf
        ).argmin(axis=0)

        # New medoids
        new_medoids = X[medoid_idx]

        # Return new medoids
        return new_medoids


    def k_medoids(X: jax.Array, k: int = 4, seed: int = 364):
        # Init the centroid
        init_medoids = jax.random.choice(
            jax.random.key(seed), X, axis=0, shape=(k,), replace=False
        )

        # medoids upodates
        _, medoids = jax.lax.while_loop(
            lambda t: t[0] < 10_000,
            lambda t: (
                t[0] + 1,
                update_medoids(X=X, medoids=t[1], k=k),
            ),
            (0, init_medoids),
        )

        # Get Labels
        labels = all_centroid_distance(X=X, centroids=medoids).argmin(axis=1)

        # return
        return medoids, labels

    return (k_medoids,)


@app.cell
def _(X_arr, k_medoids):
    # medoid Labels and clusters
    medoids, medoids_labels = k_medoids(X=X_arr, k=4, seed=364)
    return medoids, medoids_labels


@app.cell
def _(X_arr, go, medoids, medoids_labels):
    # Get final cluster assignments
    _labels = medoids_labels
    _k = 4
    _feat_x, _feat_y = 0, 1  # petal length, petal width

    _fig = go.Figure()

    # Plot each cluster
    for _c in range(_k):
        _mask = _labels == _c
        _fig.add_trace(
            go.Scatter(
                x=X_arr[_mask, _feat_x],
                y=X_arr[_mask, _feat_y],
                mode="markers",
                name=f"Cluster {_c}",
                marker=dict(size=8, opacity=0.7),
            )
        )

    # Plot centroids
    _fig.add_trace(
        go.Scatter(
            x=medoids[:, _feat_x],
            y=medoids[:, _feat_y],
            mode="markers",
            name="Centroids",
            marker=dict(size=16, symbol="x", color="black", line=dict(width=2)),
        )
    )

    _fig.update_layout(
        title=dict(text="K-Medoids Clusters (Iris)", x=0.5),
        xaxis_title="Petal Length (scaled)",
        yaxis_title="Petal Width (scaled)",
        legend_title="Legend",
        template="plotly_white",
    )

    _fig.show()
    return


@app.cell
def _(X_arr, compute_inertia, medoids, medoids_labels):
    # Inertia
    compute_inertia(
        X=X_arr,
        centroids=medoids,
        labels=medoids_labels,
    )
    return


@app.cell
def _(X_arr, k: int, medoids_labels, silhouette_score):
    # Silhouette Score
    silhouette_score(
        k=k,
        labels=medoids_labels,
        X=X_arr,
    )
    return


@app.cell
def _(medoids_labels):
    # Medoids labels
    medoids_labels
    return


if __name__ == "__main__":
    app.run()
