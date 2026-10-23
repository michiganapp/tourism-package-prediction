"""Job 2 - clean the registered data and create stratified train/test splits."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "tourism_project" / "data" / "tourism.csv"
TARGET = "ProdTaken"

# Unnamed: 0 is a stale row index, CustomerID is an identifier and Designation
# is a perfect duplicate of ProductPitched (verified in the EDA).
DROP_COLUMNS = ["Unnamed: 0", "CustomerID", "Designation"]


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print("=" * 62)
    print("DATA PREPARATION")
    print("=" * 62)
    print(f"Loaded        : {df.shape[0]} rows x {df.shape[1]} columns")

    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df = df.dropna(subset=[TARGET])

    # Merge duplicated category labels created by manual data entry.
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c != TARGET]

    # Median / mode imputation keeps the script robust to future data drops.
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    numeric_features = [c for c in numeric_cols if c != TARGET]
    categorical_features = categorical_cols

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train.to_csv(ROOT / "Xtrain.csv", index=False)
    X_test.to_csv(ROOT / "Xtest.csv", index=False)
    y_train.to_csv(ROOT / "ytrain.csv", index=False)
    y_test.to_csv(ROOT / "ytest.csv", index=False)

    print(f"Modelling data: {X.shape[0]} rows x {X.shape[1]} features")
    print(f"Train         : {X_train.shape[0]} rows | buyer rate {y_train.mean():.2%}")
    print(f"Test          : {X_test.shape[0]} rows | buyer rate {y_test.mean():.2%}")
    print(f"Numeric ({len(numeric_features)})   : {numeric_features}")
    print(f"Categorical ({len(categorical_features)}): {categorical_features}")
    print("Splits written: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
    print("=" * 62)


if __name__ == "__main__":
    main()
