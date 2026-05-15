"""Automated preprocessing pipeline for the BankChurners dataset.

The script is intentionally self-contained so it can run from the repository
root or from the ``preprocessing`` directory used by the GitHub Actions job.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

TARGET_COLUMN = "Attrition_Flag"
TARGET_MAPPING = {
    "Existing Customer": 0,
    "Attrited Customer": 1,
}

LEAKAGE_COLUMNS = [
    "CLIENTNUM",
    (
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_"
        "12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1"
    ),
    (
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_"
        "12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"
    ),
]

DEFAULT_RAW_DATA_PATH = PROJECT_ROOT / "BankChurners.csv"
DEFAULT_PROCESSED_DATA_PATH = SCRIPT_DIR / "BankChurners_preprocessing.csv"


def configure_logging() -> None:
    """Configure console logging for local and CI runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def resolve_path(value: str | None, default: Path) -> Path:
    """Resolve an optional environment path against the current working dir."""
    if value:
        path = Path(value).expanduser()
        return path if path.is_absolute() else Path.cwd() / path
    return default


def get_input_path() -> Path:
    """Return raw dataset path from RAW_DATA_PATH or the repository default."""
    return resolve_path(os.getenv("RAW_DATA_PATH"), DEFAULT_RAW_DATA_PATH)


def get_output_path() -> Path:
    """Return processed dataset path from PROCESSED_DATA_PATH or default."""
    return resolve_path(os.getenv("PROCESSED_DATA_PATH"), DEFAULT_PROCESSED_DATA_PATH)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the raw BankChurners dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")

    LOGGER.info("Loading raw dataset from %s", path)
    return pd.read_csv(path)


def validate_columns(data: pd.DataFrame) -> None:
    """Validate columns needed by the preprocessing contract."""
    required_columns = [TARGET_COLUMN, *LEAKAGE_COLUMNS]
    missing_columns = [column for column in required_columns if column not in data.columns]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Raw dataset is missing required columns: {missing}")


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows and columns that should not enter training."""
    cleaned = data.copy()
    duplicate_count = int(cleaned.duplicated().sum())

    if duplicate_count:
        LOGGER.info("Dropping %s duplicate rows", duplicate_count)
        cleaned = cleaned.drop_duplicates()

    LOGGER.info("Dropping ID/leakage columns: %s", ", ".join(LEAKAGE_COLUMNS))
    return cleaned.drop(columns=LEAKAGE_COLUMNS)


def encode_target(data: pd.DataFrame) -> pd.DataFrame:
    """Encode Attrition_Flag into a binary classification target."""
    encoded = data.copy()
    unknown_targets = set(encoded[TARGET_COLUMN].dropna().unique()) - set(TARGET_MAPPING)

    if unknown_targets:
        unknown = ", ".join(sorted(str(value) for value in unknown_targets))
        raise ValueError(f"Unknown target labels found in {TARGET_COLUMN}: {unknown}")

    encoded[TARGET_COLUMN] = encoded[TARGET_COLUMN].map(TARGET_MAPPING).astype("int64")
    return encoded


def preprocess_features(data: pd.DataFrame) -> pd.DataFrame:
    """Scale numeric features and one-hot encode categorical features."""
    features = data.drop(columns=[TARGET_COLUMN]).copy()
    target = data[TARGET_COLUMN].copy()

    categorical_columns = list(features.select_dtypes(include=["object", "category"]).columns)
    numeric_columns = list(features.select_dtypes(include=["number"]).columns)

    if numeric_columns:
        LOGGER.info("Scaling %s numeric columns", len(numeric_columns))
        features[numeric_columns] = features[numeric_columns].apply(pd.to_numeric, errors="coerce")
        features[numeric_columns] = features[numeric_columns].fillna(features[numeric_columns].median())
        scaler = StandardScaler()
        features[numeric_columns] = scaler.fit_transform(features[numeric_columns])

    if categorical_columns:
        LOGGER.info("Encoding %s categorical columns", len(categorical_columns))
        features[categorical_columns] = features[categorical_columns].fillna("Unknown")
        features = pd.get_dummies(
            features,
            columns=categorical_columns,
            dtype="int64",
        )

    processed = features.copy()
    processed[TARGET_COLUMN] = target
    return processed


def save_dataset(data: pd.DataFrame, path: Path) -> None:
    """Save the processed dataset as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    data.to_csv(temp_path, index=False)
    temp_path.replace(path)
    LOGGER.info("Saved processed dataset to %s", path)


def run_pipeline(input_path: Path | None = None, output_path: Path | None = None) -> pd.DataFrame:
    """Run the full preprocessing pipeline and return the processed data."""
    raw_path = input_path or get_input_path()
    processed_path = output_path or get_output_path()

    data = load_dataset(raw_path)
    LOGGER.info("Raw dataset shape: %s rows, %s columns", *data.shape)

    validate_columns(data)
    cleaned = clean_dataset(data)
    encoded = encode_target(cleaned)
    processed = preprocess_features(encoded)

    save_dataset(processed, processed_path)
    LOGGER.info("Processed dataset shape: %s rows, %s columns", *processed.shape)
    LOGGER.info("Target distribution: %s", processed[TARGET_COLUMN].value_counts().to_dict())

    return processed


def main() -> None:
    """CLI entrypoint."""
    configure_logging()
    run_pipeline()


if __name__ == "__main__":
    main()
