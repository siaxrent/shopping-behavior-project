import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def build_preprocess_pipeline(df: pd.DataFrame, drop_cols=None):
    """
    Пайплайн:
    - числовые: медиана + StandardScaler
    - категориальные: мода + OneHotEncoder
    drop_cols: список колонок, которые нельзя использовать (например таргет)
    """
    drop_cols = drop_cols or []
    work_df = df.drop(columns=drop_cols, errors="ignore")

    num_cols = work_df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in work_df.columns if c not in num_cols]

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


def fit_transform_preprocess(df: pd.DataFrame, drop_cols=None):
    drop_cols = drop_cols or []
    preprocess, num_cols, cat_cols = build_preprocess_pipeline(df, drop_cols=drop_cols)
    X = preprocess.fit_transform(df.drop(columns=drop_cols, errors="ignore"))
    return X, preprocess, num_cols, cat_cols

