import os
import json
import joblib
import numpy as np
import pandas as pd
from src.data_preprocessing import fit_transform_preprocess
from src.clustering import run_dbscan_numeric, run_kmeans_with_pca
from src.modeling import train_subscription_models
from src.evaluation import save_metrics
from src.visualization import plot_roc_curve


DATA_PATH = os.path.join("data", "raw", "shopping_behavior.csv")

OUT_FIG = os.path.join("results", "figures")
OUT_REP = os.path.join("results", "reports")
OUT_MOD = os.path.join("models")

os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_REP, exist_ok=True)
os.makedirs(OUT_MOD, exist_ok=True)


def main():
    # 1) Load
    df = pd.read_csv(DATA_PATH)
    print("Loaded:", df.shape)

    # 2) Clustering preprocess (use all features, including Subscription Status as categorical,
    #    which is acceptable for descriptive segmentation)
    X_clust, _, _, _ = fit_transform_preprocess(df)

    # 3) KMeans (improved): PCA before clustering + wider k search
    print("KMeans MODE: PCA(15) + k=2..15")
    kmeans_result = run_kmeans_with_pca(X_clust, OUT_FIG)
    best_k = int(kmeans_result["best_k"])
    best_score = float(kmeans_result["best_silhouette"])
    print("Best KMeans k:", best_k, "silhouette:", best_score)

    # 4) DBSCAN on numeric-only (works better than one-hot sparse)
    dbscan_result = run_dbscan_numeric(df, OUT_FIG)
    db_sil = dbscan_result["silhouette"]
    n_clusters = int(dbscan_result["n_clusters"])
    n_noise = int(dbscan_result["n_noise"])

    if db_sil is None:
        print("DBSCAN (numeric): could not find parameters with >=2 clusters.")
    else:
        print(
            "DBSCAN (numeric) best:",
            "eps=",
            dbscan_result["eps"],
            "min_samples=",
            dbscan_result["min_samples"],
            "clusters=",
            n_clusters,
            "noise=",
            n_noise,
            "silhouette=",
            db_sil,
        )

    # 5) Classification: predict Subscription Status (the column is removed from features)
    modeling_result = train_subscription_models(df)
    rf_auc = float(modeling_result["rf_auc"])
    svm_auc = float(modeling_result["svm_auc"])
    best_name = modeling_result["best_model_name"]
    best_auc = float(modeling_result["best_auc"])
    best_model = modeling_result["best_model"]
    best_proba = modeling_result["best_proba"]
    y_test = modeling_result["y_test"]

    joblib.dump(best_model, os.path.join(OUT_MOD, "best_model.joblib"))

    metrics = {
        "kmeans_mode": "PCA(15)",
        "kmeans_best_k": int(best_k),
        "kmeans_best_silhouette": best_score,
        "dbscan_clusters": int(n_clusters),
        "dbscan_noise_points": int(n_noise),
        "dbscan_silhouette_no_noise": db_sil,
        "rf_auc": rf_auc,
        "svm_auc": svm_auc,
        "best_model": best_name,
        "best_auc": best_auc
    }
    save_metrics(metrics, os.path.join(OUT_REP, "metrics_main.json"))

    plot_roc_curve(
        y_test,
        best_proba,
        os.path.join(OUT_FIG, "best_roc_curve_main.png"),
        title=f"Best ROC curve: {best_name}",
    )

    print("Done. Saved figures/reports/models.")


if __name__ == "__main__":
    main()
