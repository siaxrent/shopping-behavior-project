import pandas as pd


def make_target_subscription(df: pd.DataFrame, column: str = "Subscription Status") -> pd.Series:
    """
    Build binary target from subscription status.
    Maps 'Yes' -> 1, 'No' -> 0, raises an error if other values are present.
    """
    y = (
        df[column]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )
    if y.isna().any():
        raise ValueError(
            f"{column} contains values other than Yes/No. Please check the data."
        )
    return y.astype(int)


