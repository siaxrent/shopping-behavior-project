import os
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .visualization import (
    plot_dbscan_pca,
    plot_kmeans_pca_scatter,
    plot_kmeans_silhouette,
)


def run_kmeans_with_pca(
    X,
    out_fig_dir: str,
    k_range: Iterable[int] = range(2, 16),
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Run KMeans on PCA(15)-reduced features, search over k, and save plots.

    Returns dict with:
        - best_k
        - best_silhouette
    """
    os.makedirs(out_fig_dir, exist_ok=True)

    X_dense = X.toarray() if hasattr(X, "toarray") else X

    pca = PCA(n_components=15, random_state=random_state)
    X_km = pca.fit_transform(X_dense)

    ks: Sequence[int] = list(k_range)
    scores = []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X_km)
        scores.append(silhouette_score(X_km, labels))

    best_idx = int(np.argmax(scores))
    best_k = int(ks[best_idx])
    best_score = float(scores[best_idx])

    # Silhouette vs k plot
    plot_kmeans_silhouette(ks, scores, out_fig_dir)

    # Scatter plot in first two PCs for best_k
    best_kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init="auto")
    k_labels = best_kmeans.fit_predict(X_km)

    plot_kmeans_pca_scatter(
        X_km,
        k_labels,
        out_fig_dir,
        title=f"KMeans on PCA(15) (k={best_k}) shown by PC1-PC2",
    )

    return {
        "best_k": best_k,
        "best_silhouette": best_score,
    }


def run_dbscan_numeric(
    df: pd.DataFrame,
    out_fig_dir: str,
    numeric_cols: Optional[Sequence[str]] = None,
    eps_grid: Optional[Sequence[float]] = None,
    min_samples_grid: Optional[Sequence[int]] = None,
    random_state: int = 42,
) -> Dict[str, Optional[float]]:
    """
    Run DBSCAN on numeric-only features with simple grid search and save PCA-2D plot.

    Returns dict with:
        - silhouette (None if not enough clusters)
        - n_clusters
        - n_noise
        - eps
        - min_samples
    """
    os.makedirs(out_fig_dir, exist_ok=True)

    if numeric_cols is None:
        numeric_cols = ["Age", "Purchase Amount (USD)", "Review Rating", "Previous Purchases"]

    if eps_grid is None:
        eps_grid = [0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.3, 1.6]
    if min_samples_grid is None:
        min_samples_grid = [3, 5, 8, 12, 20]

    X_num_df = df[numeric_cols].copy()

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    X_db = num_pipe.fit_transform(X_num_df)

    best_db = {
        "eps": None,
        "min_samples": None,
        "silhouette": None,
        "clusters": 0,
        "noise": None,
        "labels": None,
    }

    for eps in eps_grid:
        for ms in min_samples_grid:
            db = DBSCAN(eps=eps, min_samples=ms)
            labels = db.fit_predict(X_db)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise = int((labels == -1).sum())

            mask = labels != -1
            if mask.sum() > 0 and len(set(labels[mask])) >= 2:
                sil = float(silhouette_score(X_db[mask], labels[mask]))
                if (best_db["silhouette"] is None) or (sil > best_db["silhouette"]):
                    best_db.update(
                        {
                            "eps": eps,
                            "min_samples": ms,
                            "silhouette": sil,
                            "clusters": n_clusters,
                            "noise": noise,
                            "labels": labels,
                        }
                    )

    if best_db["eps"] is None:
        # no configuration produced >=2 clusters (excluding noise)
        db_sil = None
        n_clusters_final = 0
        n_noise_final = int(X_db.shape[0])
    else:
        # PCA 2D for numeric space
        pca = PCA(n_components=2, random_state=random_state)
        X2 = pca.fit_transform(X_db)
        plot_dbscan_pca(
            X2,
            best_db["labels"],
            out_fig_dir,
            title=(
                f"DBSCAN (numeric) on PCA-2D "
                f"(eps={best_db['eps']}, ms={best_db['min_samples']})"
            ),
        )

        db_sil = best_db["silhouette"]
        n_clusters_final = best_db["clusters"]
        n_noise_final = best_db["noise"]

    return {
        "silhouette": db_sil,
        "n_clusters": int(n_clusters_final),
        "n_noise": int(n_noise_final),
        "eps": best_db["eps"],
        "min_samples": best_db["min_samples"],
    }


