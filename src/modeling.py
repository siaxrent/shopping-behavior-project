from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .data_preprocessing import build_preprocess_pipeline
from .utils import make_target_subscription


def train_subscription_models(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train RandomForest and SVM models to predict subscription status.

    Returns dict with:
        - rf_auc
        - svm_auc
        - best_model_name
        - best_auc
        - best_model
        - best_proba
        - y_test
    """
    y = make_target_subscription(df)
    X_df = df.drop(columns=["Subscription Status"])

    preprocess, _, _ = build_preprocess_pipeline(X_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    rf = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=random_state,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_auc = float(roc_auc_score(y_test, rf_proba))

    svm = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                SVC(
                    C=2.0,
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    svm.fit(X_train, y_train)
    svm_proba = svm.predict_proba(X_test)[:, 1]
    svm_auc = float(roc_auc_score(y_test, svm_proba))

    if rf_auc >= svm_auc:
        best_name = "RF"
        best_model = rf
        best_auc = rf_auc
        best_proba = rf_proba
    else:
        best_name = "SVM"
        best_model = svm
        best_auc = svm_auc
        best_proba = svm_proba

    print("RF ROC-AUC:", rf_auc)
    print("SVM ROC-AUC:", svm_auc)
    print("Best model:", best_name, "AUC:", best_auc)

    return {
        "rf_auc": rf_auc,
        "svm_auc": svm_auc,
        "best_model_name": best_name,
        "best_auc": float(best_auc),
        "best_model": best_model,
        "best_proba": best_proba,
        "y_test": y_test,
    }


