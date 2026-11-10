import pandas as pd
from .config import DATASET_PATH, FEATURE_COLUMNS, TARGET

def load_dataset(path=DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download creditcard.csv from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place it in the dataset/ folder."
        )
    df = pd.read_csv(path)
    required = set(FEATURE_COLUMNS + [TARGET])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Dataset is empty.")
    if df[TARGET].isna().any():
        raise ValueError("Target column 'Class' contains missing values.")
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target column 'Class' must contain only 0 and 1.")
    return df
