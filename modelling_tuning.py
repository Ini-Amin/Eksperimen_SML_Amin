"""Hyperparameter tuning with manual MLflow logging."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "preprocessing" / "BankChurners_preprocessing.csv"
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000/"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "tuning"
DEFAULT_DAGSHUB_OWNER = "Ini-Amin"
DEFAULT_DAGSHUB_REPO = "Eksperimen_SML_Amin"
DEFAULT_REGISTERED_MODEL_NAME = "BankChurners_RandomForest"

TARGET_COLUMN = "Attrition_Flag"
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "BankChurners Credit Card Churn")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", DEFAULT_REGISTERED_MODEL_NAME)
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
CV_FOLDS = int(os.getenv("CV_FOLDS", "3"))


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
    """Load processed dataset and validate the target column."""
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


def build_search() -> GridSearchCV:
    """Create a compact GridSearchCV object for RandomForest tuning."""
    estimator = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    param_grid = {
        "n_estimators": [120, 200],
        "max_depth": [None, 12],
        "min_samples_leaf": [1, 3],
    }

    return GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="f1",
        cv=CV_FOLDS,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )


def evaluate_model(
    model: RandomForestClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float], pd.Series, pd.Series]:
    """Evaluate a classifier and return metrics plus predictions."""
    predictions = pd.Series(model.predict(x_test), index=y_test.index, name="prediction")
    probabilities = pd.Series(
        model.predict_proba(x_test)[:, 1],
        index=y_test.index,
        name="attrition_probability",
    )
    metrics = {
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
        "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
        "test_f1": float(f1_score(y_test, predictions, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    return metrics, predictions, probabilities


def save_json(data: dict[str, object], path: Path) -> None:
    """Save a dictionary as formatted JSON."""
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_confusion_matrix(y_test: pd.Series, predictions: pd.Series, path: Path) -> None:
    """Save a confusion matrix image."""
    matrix = confusion_matrix(y_test, predictions)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Existing Customer", "Attrited Customer"],
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Tuned Random Forest")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_feature_importance(model: RandomForestClassifier, feature_names: list[str], output_dir: Path) -> None:
    """Save feature importance as CSV and PNG."""
    importance = (
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    top_features = importance.head(20).sort_values("importance", ascending=True)
    plt.figure(figsize=(10, 8))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("Top 20 Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_top20.png", dpi=150)
    plt.close()


def save_artifacts(
    output_dir: Path,
    search: GridSearchCV,
    model: RandomForestClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: pd.Series,
    probabilities: pd.Series,
    metrics: dict[str, float],
) -> None:
    """Save evaluation and tuning artifacts to a local directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    report_dict = classification_report(
        y_test,
        predictions,
        target_names=["Existing Customer", "Attrited Customer"],
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test,
        predictions,
        target_names=["Existing Customer", "Attrited Customer"],
        zero_division=0,
    )

    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    save_json(report_dict, output_dir / "classification_report.json")
    save_json(metrics, output_dir / "metrics.json")
    save_json(search.best_params_, output_dir / "best_params.json")
    pd.DataFrame(search.cv_results_).to_csv(output_dir / "cv_results.csv", index=False)
    pd.DataFrame(
        {
            "actual": y_test,
            "prediction": predictions,
            "attrition_probability": probabilities,
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)

    save_confusion_matrix(y_test, predictions, output_dir / "confusion_matrix.png")
    save_feature_importance(model, list(x_test.columns), output_dir)


def train_tuned_model(data: pd.DataFrame, artifact_dir: Path) -> RandomForestClassifier:
    """Run tuning, log outputs to MLflow, and return the best model."""
    x_train, x_test, y_train, y_test = split_features_target(data)
    search = build_search()

    project_run_id = os.getenv("MLFLOW_RUN_ID")
    start_run_kwargs = {"run_id": project_run_id} if project_run_id else {"run_name": "tuned_random_forest"}

    with mlflow.start_run(**start_run_kwargs):
        if project_run_id:
            mlflow.set_tag("mlflow.runName", "tuned_random_forest")

        mlflow.log_param("data_rows", data.shape[0])
        mlflow.log_param("data_columns", data.shape[1])
        mlflow.log_param("target_column", TARGET_COLUMN)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("scoring", "f1")

        LOGGER.info("Starting GridSearchCV with %s CV folds", CV_FOLDS)
        search.fit(x_train, y_train)

        best_model = search.best_estimator_
        metrics, predictions, probabilities = evaluate_model(best_model, x_test, y_test)

        mlflow.log_params({f"best_{key}": value for key, value in search.best_params_.items()})
        mlflow.log_metric("best_cv_f1", float(search.best_score_))
        mlflow.log_metrics(metrics)

        save_artifacts(
            output_dir=artifact_dir,
            search=search,
            model=best_model,
            x_test=x_test,
            y_test=y_test,
            predictions=predictions,
            probabilities=probabilities,
            metrics=metrics,
        )
        mlflow.log_artifacts(str(artifact_dir), artifact_path="evaluation")

        signature = infer_signature(x_train, best_model.predict(x_train))
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=x_train.head(5),
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        LOGGER.info("Best params: %s", search.best_params_)
        LOGGER.info("Registered model name: %s", REGISTERED_MODEL_NAME)
        LOGGER.info("Tuned metrics: %s", {key: round(value, 4) for key, value in metrics.items()})

    return best_model


def main() -> None:
    """Run tuned training."""
    configure_logging()
    configure_mlflow()

    data_path = resolve_path(os.getenv("DATA_PATH"), DEFAULT_DATA_PATH)
    artifact_dir = resolve_path(os.getenv("ARTIFACT_DIR"), DEFAULT_ARTIFACT_DIR)
    LOGGER.info("Loading processed dataset from %s", data_path)
    data = load_processed_data(data_path)
    LOGGER.info("Processed dataset shape: %s rows, %s columns", *data.shape)
    train_tuned_model(data, artifact_dir)


if __name__ == "__main__":
    main()
