import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


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

    # 3) KMeans (improved): PCA before clustering + wider k search
    print("KMeans MODE: PCA(15) + k=2..15")

    X_dense = X_clust.toarray() if hasattr(X_clust, "toarray") else X_clust

    pca_kmeans = PCA(n_components=15, random_state=42)
    X_km = pca_kmeans.fit_transform(X_dense)

    ks = range(2, 16)
    scores = []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X_km)
        scores.append(silhouette_score(X_km, labels))

    best_k = int(list(ks)[int(np.argmax(scores))])
    best_score = float(max(scores))
    print("Best KMeans k:", best_k, "silhouette:", best_score)

    plt.figure(figsize=(6, 3))
    plt.plot(list(ks), scores, marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette score")
    plt.title("KMeans on PCA(15): Silhouette by k")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "kmeans_pca15_silhouette_by_k.png"), dpi=150)
    plt.close()

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
    k_labels = kmeans.fit_predict(X_km)

    # PCA(2) plot для картинки (берём первые 2 компоненты уже готового PCA(15))
    plt.figure(figsize=(7, 5))
    plt.scatter(X_km[:, 0], X_km[:, 1], c=k_labels, s=12)
    plt.title(f"KMeans on PCA(15) (k={best_k}) shown by PC1-PC2")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "kmeans_pca15_scatter_pc1_pc2.png"), dpi=150)
    plt.close()


        # 4) DBSCAN on numeric-only (works better than one-hot sparse)
    numeric_cols = ["Age", "Purchase Amount (USD)", "Review Rating", "Previous Purchases"]
    X_num_df = df[numeric_cols].copy()

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_db = num_pipe.fit_transform(X_num_df)

    best_db = {"eps": None, "min_samples": None, "silhouette": None, "clusters": 0, "noise": None, "labels": None}

    eps_grid = [0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.3, 1.6]
    min_samples_grid = [3, 5, 8, 12, 20]

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
                    best_db.update({
                        "eps": eps, "min_samples": ms, "silhouette": sil,
                        "clusters": n_clusters, "noise": noise, "labels": labels
                    })

    if best_db["eps"] is None:
        print("DBSCAN (numeric): could not find parameters with >=2 clusters.")
        db_sil = None
        n_clusters, n_noise = 0, int(X_db.shape[0])
    else:
        print(
            "DBSCAN (numeric) best:",
            "eps=", best_db["eps"],
            "min_samples=", best_db["min_samples"],
            "clusters=", best_db["clusters"],
            "noise=", best_db["noise"],
            "silhouette=", best_db["silhouette"]
        )

        # PCA 2D for numeric space
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X_db)
        plt.figure(figsize=(7, 5))
        plt.scatter(X2[:, 0], X2[:, 1], c=best_db["labels"], s=12)
        plt.title(f"DBSCAN (numeric) on PCA-2D (eps={best_db['eps']}, ms={best_db['min_samples']})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_FIG, "dbscan_numeric_pca.png"), dpi=150)
        plt.close()

        db_sil = best_db["silhouette"]
        n_clusters, n_noise = best_db["clusters"], best_db["noise"]

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
