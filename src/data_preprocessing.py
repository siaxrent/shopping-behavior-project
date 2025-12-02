import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def build_preprocess_pipeline(df: pd.DataFrame):
    """
    Создаёт пайплайн предобработки:
    - числовые: заполнение пропусков медианой + StandardScaler (нормализация)
    - категориальные: заполнение модой + OneHotEncoder
    Возвращает (pipeline, num_cols, cat_cols)
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )

    return preprocess, num_cols, cat_cols


def fit_transform_preprocess(df: pd.DataFrame):
    preprocess, num_cols, cat_cols = build_preprocess_pipeline(df)
    X = preprocess.fit_transform(df)
    return X, preprocess, num_cols, cat_cols
