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
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import minmax_scale

    return jax, jnp, load_iris, minmax_scale, partial


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

    return all_centroid_distance, knn


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
    k: int = 5

    # Get Centroids
    centroids = knn(X=X_arr, k=k, seed=42342, n_iter=100)
    return centroids, k


@app.cell
def _(centroids):
    centroids
    return


@app.cell
def _(X_arr, all_centroid_distance, centroids, k: int):
    import plotly.graph_objects as go

    # Get final cluster assignments
    labels = all_centroid_distance(X=X_arr, centroids=centroids).argmin(axis=1)

    feat_x, feat_y = 0, 1  # petal length, petal width

    fig = go.Figure()

    # Plot each cluster
    for c in range(k):
        mask = labels == c
        fig.add_trace(
            go.Scatter(
                x=X_arr[mask, feat_x],
                y=X_arr[mask, feat_y],
                mode="markers",
                name=f"Cluster {c}",
                marker=dict(size=8, opacity=0.7),
            )
        )

    # Plot centroids
    fig.add_trace(
        go.Scatter(
            x=centroids[:, feat_x],
            y=centroids[:, feat_y],
            mode="markers",
            name="Centroids",
            marker=dict(size=16, symbol="x", color="black", line=dict(width=2)),
        )
    )

    fig.update_layout(
        title=dict(text="K-Means Clusters (Iris)", x=0.5),
        xaxis_title="Petal Length (scaled)",
        yaxis_title="Petal Width (scaled)",
        legend_title="Legend",
        template="plotly_white",
    )

    fig.show()
    return


if __name__ == "__main__":
    app.run()
