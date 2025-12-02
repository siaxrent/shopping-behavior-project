import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import RocCurveDisplay

from src.data_preprocessing import build_preprocess_pipeline, fit_transform_preprocess


DATA_PATH = os.path.join("data", "raw", "shopping_behavior.csv")

OUT_FIG = os.path.join("results", "figures")
OUT_REP = os.path.join("results", "reports")
OUT_MOD = os.path.join("models")

os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_REP, exist_ok=True)
os.makedirs(OUT_MOD, exist_ok=True)


def make_target_subscription(df: pd.DataFrame) -> pd.Series:
    # Subscription Status: Yes/No -> 1/0
    y = df["Subscription Status"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if y.isna().any():
        raise ValueError("Subscription Status содержит значения кроме Yes/No. Проверь данные.")
    return y.astype(int)


def save_pca_scatter(X, labels, title, filename):
    pca = PCA(n_components=2, random_state=42)
    X_dense = X.toarray() if hasattr(X, "toarray") else X
    X2 = pca.fit_transform(X_dense)

    plt.figure(figsize=(7, 5))
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=12)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, filename), dpi=150)
    plt.close()


def main():
    # 1) Load
    df = pd.read_csv(DATA_PATH)
    print("Loaded:", df.shape)

    # 2) Clustering preprocess (используем все признаки, включая Subscription Status как категориальный,
    #    потому что для сегментации это допустимо)
    X_clust, _, _, _ = fit_transform_preprocess(df)

    # 3) KMeans: choose best k by silhouette
    ks = range(2, 11)
    scores = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X_clust)
        scores.append(silhouette_score(X_clust, labels))

    best_k = list(ks)[int(np.argmax(scores))]
    best_score = float(max(scores))
    print("Best KMeans k:", best_k, "silhouette:", best_score)

    plt.figure(figsize=(6, 3))
    plt.plot(list(ks), scores, marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette score")
    plt.title("KMeans: Silhouette by k")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "kmeans_silhouette_by_k.png"), dpi=150)
    plt.close()

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
    k_labels = kmeans.fit_predict(X_clust)
    save_pca_scatter(X_clust, k_labels, f"KMeans clusters (k={best_k}) on PCA-2D", "kmeans_pca.png")

    # 4) DBSCAN (базовые параметры; при желании потом подберём лучше)
    db = DBSCAN(eps=0.8, min_samples=8)
    db_labels = db.fit_predict(X_clust)
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = int((db_labels == -1).sum())
    print("DBSCAN clusters:", n_clusters, "noise:", n_noise)

    # silhouette для DBSCAN считаем без шума, если есть >=2 кластера
    db_sil = None
    mask = db_labels != -1
    if mask.sum() > 0 and len(set(db_labels[mask])) >= 2:
        db_sil = float(silhouette_score(X_clust[mask], db_labels[mask]))
        print("DBSCAN silhouette (no noise):", db_sil)

    # 5) Classification target: Subscription Status (важно: эту колонку УБИРАЕМ из признаков)
    y = make_target_subscription(df)
    X_df = df.drop(columns=["Subscription Status"])

    preprocess, _, _ = build_preprocess_pipeline(X_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1
        ))
    ])
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_auc = float(roc_auc_score(y_test, rf_proba))
    print("RF ROC-AUC:", rf_auc)

    svm = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", SVC(C=2.0, kernel="rbf", probability=True, class_weight="balanced", random_state=42))
    ])
    svm.fit(X_train, y_train)
    svm_proba = svm.predict_proba(X_test)[:, 1]
    svm_auc = float(roc_auc_score(y_test, svm_proba))
    print("SVM ROC-AUC:", svm_auc)

    best_name, best_model, best_auc, best_proba = ("RF", rf, rf_auc, rf_proba) if rf_auc >= svm_auc else ("SVM", svm, svm_auc, svm_proba)
    print("Best model:", best_name, "AUC:", best_auc)

    joblib.dump(best_model, os.path.join(OUT_MOD, "best_model.joblib"))

    metrics = {
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
    with open(os.path.join(OUT_REP, "metrics_main.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    RocCurveDisplay.from_predictions(y_test, best_proba)
    plt.title(f"Best ROC curve: {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "best_roc_curve_main.png"), dpi=150)
    plt.close()

    print("Done. Saved figures/reports/models.")


if __name__ == "__main__":
    main()
