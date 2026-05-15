"""Baseline model training with MLflow autologging."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "preprocessing" / "BankChurners_preprocessing.csv"
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000/"
DEFAULT_DAGSHUB_OWNER = "Ini-Amin"
DEFAULT_DAGSHUB_REPO = "Eksperimen_SML_Amin"

TARGET_COLUMN = "Attrition_Flag"
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "BankChurners Credit Card Churn")
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))


def configure_console_encoding() -> None:
    """Use UTF-8 streams so MLflow console output works on Windows."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def configure_logging() -> None:
    """Configure console logging."""
    configure_console_encoding()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def env_flag(name: str, default: str = "false") -> bool:
    """Return True when an environment flag is enabled."""
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}


def resolve_path(value: str | None, default: Path) -> Path:
    """Resolve an optional path environment variable."""
    if value:
        path = Path(value).expanduser()
        return path if path.is_absolute() else Path.cwd() / path
    return default


def configure_dagshub() -> bool:
    """Configure DagsHub MLflow tracking when USE_DAGSHUB=true."""
    if not env_flag("USE_DAGSHUB"):
        return False

    try:
        import dagshub
    except ImportError as exc:
        raise RuntimeError(
            "USE_DAGSHUB=true requires the 'dagshub' package. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    repo_owner = os.getenv("DAGSHUB_REPO_OWNER", DEFAULT_DAGSHUB_OWNER)
    repo_name = os.getenv("DAGSHUB_REPO_NAME", DEFAULT_DAGSHUB_REPO)
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    LOGGER.info("DagsHub MLflow tracking enabled for %s/%s", repo_owner, repo_name)
    return True


def configure_mlflow() -> None:
    """Configure MLflow tracking from environment variables."""
    dagshub_enabled = configure_dagshub()

    if dagshub_enabled and not os.getenv("MLFLOW_TRACKING_URI"):
        tracking_uri = mlflow.get_tracking_uri()
    else:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
        mlflow.set_tracking_uri(tracking_uri)

    project_run_id = os.getenv("MLFLOW_RUN_ID")
    if project_run_id:
        LOGGER.info("Using existing MLflow Project run: %s", project_run_id)
    else:
        mlflow.set_experiment(EXPERIMENT_NAME)

    LOGGER.info("MLflow tracking URI: %s", tracking_uri)
    LOGGER.info("MLflow experiment: %s", EXPERIMENT_NAME)


def load_processed_data(path: Path) -> pd.DataFrame:
    """Load preprocessed training data."""
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. "
            "Run preprocessing/automate_amin.py first."
        )

    data = pd.read_csv(path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing from {path}")

    return data


def split_features_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split processed data into train and test sets."""
    x = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN].astype("int64")
    return train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def train_baseline_model(data: pd.DataFrame) -> RandomForestClassifier:
    """Train and log a baseline RandomForest model."""
    x_train, x_test, y_train, y_test = split_features_target(data)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        registered_model_name=None,
    )

    project_run_id = os.getenv("MLFLOW_RUN_ID")
    start_run_kwargs = {"run_id": project_run_id} if project_run_id else {"run_name": "baseline_random_forest"}

    with mlflow.start_run(**start_run_kwargs):
        if project_run_id:
            mlflow.set_tag("mlflow.runName", "baseline_random_forest")

        mlflow.log_param("data_rows", data.shape[0])
        mlflow.log_param("data_columns", data.shape[1])
        mlflow.log_param("target_column", TARGET_COLUMN)

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        metrics = {
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
            "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
            "test_f1": float(f1_score(y_test, predictions, zero_division=0)),
        }
        mlflow.log_metrics(metrics)
        LOGGER.info("Baseline metrics: %s", {key: round(value, 4) for key, value in metrics.items()})

    return model


def main() -> None:
    """Run baseline training."""
    configure_logging()
    configure_mlflow()

    data_path = resolve_path(os.getenv("DATA_PATH"), DEFAULT_DATA_PATH)
    LOGGER.info("Loading processed dataset from %s", data_path)
    data = load_processed_data(data_path)
    LOGGER.info("Processed dataset shape: %s rows, %s columns", *data.shape)
    train_baseline_model(data)


if __name__ == "__main__":
    main()
