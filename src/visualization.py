import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay


def plot_kmeans_silhouette(
    ks: Sequence[int],
    scores: Sequence[float],
    out_dir: str,
    filename: str = "kmeans_pca15_silhouette_by_k.png",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(6, 3))
    plt.plot(list(ks), list(scores), marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette score")
    plt.title("KMeans on PCA(15): Silhouette by k")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close()


def plot_kmeans_pca_scatter(
    X2: np.ndarray,
    labels: np.ndarray,
    out_dir: str,
    filename: str = "kmeans_pca15_scatter_pc1_pc2.png",
    title: str | None = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=12)
    plt.title(title or "KMeans clusters on PCA-2D")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close()


def plot_dbscan_pca(
    X2: np.ndarray,
    labels: np.ndarray,
    out_dir: str,
    filename: str = "dbscan_numeric_pca.png",
    title: str | None = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=12)
    if title:
        plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close()


def plot_roc_curve(
    y_true,
    y_proba,
    out_path: str,
    title: str = "ROC curve",
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


