"""Job 1 - validate that the dataset exists, has the expected schema and is usable."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "tourism_project" / "data" / "tourism.csv"

REQUIRED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups",
    "ProductPitched", "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome",
]


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    print("=" * 62)
    print("DATA REGISTRATION SUMMARY")
    print("=" * 62)
    print(f"File            : {DATA_PATH}")
    print(f"Rows x columns  : {df.shape[0]} x {df.shape[1]}")
    print(f"Missing values  : {int(df.isna().sum().sum())}")
    print(f"Duplicate rows  : {int(df.duplicated().sum())}")
    print(f"Target column   : ProdTaken")
    print(f"Purchase rate   : {df['ProdTaken'].mean():.2%}")
    print("\nSchema:")
    print(df.dtypes.to_string())
    print("\nFirst 5 rows:")
    print(df.head().to_string())
    print("=" * 62)
    print("Dataset registration successful.")


if __name__ == "__main__":
    main()
