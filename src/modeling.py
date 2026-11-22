"""Reproducible preprocessing, logistic regression, evaluation, and artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Ecommerce_Purchase_Prediction_DataSet.csv"
FIGURES_DIR = ROOT / "reports" / "figures"
RESULTS_DIR = ROOT / "results"
MODEL_PATH = ROOT / "models" / "logistic_regression_pipeline.joblib"
TARGET = "Purchase"
ID_COLUMN = "Customer_ID"
RANDOM_STATE = 42


class IQRCapper(BaseEstimator, TransformerMixin):
    """Cap numeric outliers using IQR fences learned from the training split."""

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        q1 = np.percentile(values, 25, axis=0)
        q3 = np.percentile(values, 75, axis=0)
        spread = q3 - q1
        self.lower_ = q1 - 1.5 * spread
        self.upper_ = q3 + 1.5 * spread
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)


def build_pipeline(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("iqr_cap", IQRCapper()),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE)),
        ]
    )


def _save_eda_plots(data: pd.DataFrame, target_counts: pd.Series, numeric_columns: list[str]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(["No purchase (0)", "Purchase (1)"], target_counts.reindex([0, 1]).values, color=["#7A91A8", "#167D8D"])
    ax.set_title("Purchase outcome distribution")
    ax.set_ylabel("Customer sessions")
    ax.bar_label(bars, fmt="%d", padding=3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "purchase_distribution.png", dpi=160)
    plt.close(fig)

    missing = data.isna().sum().drop(labels=[TARGET, ID_COLUMN], errors="ignore")
    missing = missing[missing > 0].sort_values()
    fig, ax = plt.subplots(figsize=(8, max(4, 0.38 * len(missing))))
    if len(missing):
        ax.barh(missing.index, 100 * missing / len(data), color="#D99045")
        ax.set_xlabel("Missing values (%)")
        ax.set_title("Fields with missing values")
    else:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "missing_values.png", dpi=160)
    plt.close(fig)

    if "Membership_Type" in data.columns:
        membership_rate = data.groupby("Membership_Type", dropna=False)[TARGET].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(membership_rate.index.astype(str), membership_rate.values * 100, color="#167D8D")
        ax.set_title("Observed purchase rate by membership type")
        ax.set_ylabel("Purchase rate (%)")
        ax.set_ylim(0, max(80, membership_rate.max() * 115))
        for i, value in enumerate(membership_rate.values * 100):
            ax.text(i, value + 1, f"{value:.1f}%", ha="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "purchase_rate_by_membership.png", dpi=160)
        plt.close(fig)

    if "Cart_Value_INR" in numeric_columns:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for label, color in [(0, "#7A91A8"), (1, "#167D8D")]:
            values = data.loc[data[TARGET] == label, "Cart_Value_INR"].dropna()
            values = values.clip(upper=values.quantile(0.99))
            ax.hist(values, bins=35, alpha=0.62, label=f"Purchase={label}", color=color)
        ax.set_title("Cart value distribution by outcome (99th-percentile display cap)")
        ax.set_xlabel("Cart value (INR)")
        ax.set_ylabel("Sessions")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "cart_value_by_purchase.png", dpi=160)
        plt.close(fig)


def _save_model_plots(model: Pipeline, y_test: pd.Series, probabilities: np.ndarray, predictions: np.ndarray) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    auc = roc_auc_score(y_test, probabilities)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    image = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title("Confusion matrix (test set)")
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("Actual label")
    axes[0].set_xticks([0, 1], ["No purchase", "Purchase"])
    axes[0].set_yticks([0, 1], ["No purchase", "Purchase"])
    for (row, col), value in np.ndenumerate(cm):
        axes[0].text(col, row, f"{value:,}", ha="center", va="center", color="black")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    axes[1].plot(fpr, tpr, color="#167D8D", lw=2, label=f"Logistic Regression (AUC={auc:.3f})")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Random ranking")
    axes[1].set_title("ROC curve (test set)")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_evaluation.png", dpi=160)
    plt.close(fig)

    preprocessor = model.named_steps["preprocessing"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]
    coef = pd.Series(coefficients, index=feature_names).sort_values()
    selected = pd.concat([coef.head(8), coef.tail(8)]).sort_values()
    display_names = [str(x).replace("numeric__", "").replace("categorical__", "").replace("_", " ") for x in selected.index]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#B75F55" if value < 0 else "#167D8D" for value in selected.values]
    ax.barh(display_names, selected.values, color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title("Largest logistic regression coefficients")
    ax.set_xlabel("Coefficient (positive values increase predicted purchase odds)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "largest_coefficients.png", dpi=160)
    plt.close(fig)


def run_analysis(data_path: Path | str = DATA_PATH) -> dict:
    data_path = Path(data_path)
    data = pd.read_csv(data_path)
    if TARGET not in data.columns:
        raise ValueError(f"Dataset must contain target column {TARGET!r}.")
    if data[TARGET].isna().any() or set(data[TARGET].unique()) != {0, 1}:
        raise ValueError("Purchase must contain only non-missing binary values 0 and 1.")

    numeric_columns = data.drop(columns=[TARGET, ID_COLUMN], errors="ignore").select_dtypes(include=np.number).columns.tolist()
    categorical_columns = data.drop(columns=[TARGET, ID_COLUMN], errors="ignore").select_dtypes(exclude=np.number).columns.tolist()
    if not numeric_columns and not categorical_columns:
        raise ValueError("No usable predictor columns were found.")

    X = data.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
    y = data[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    model = build_pipeline(numeric_columns, categorical_columns)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "project": "E-Commerce Customer Purchase Prediction",
        "algorithm": "Logistic Regression",
        "target": TARGET,
        "positive_class": "1 = Purchase",
        "dataset_rows": int(len(data)),
        "predictor_count": int(len(numeric_columns) + len(categorical_columns)),
        "numeric_predictors": numeric_columns,
        "categorical_predictors": categorical_columns,
        "target_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "purchase_rate": float(y.mean()),
        "missing_cells": int(data.drop(columns=[ID_COLUMN], errors="ignore").isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "decision_threshold": 0.5,
        "metrics": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "majority_class_baseline_accuracy": float(y.value_counts(normalize=True).max()),
        },
        "confusion_matrix_labels_0_then_1": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "preprocessing": {
            "numeric_missing_values": "Median imputation",
            "numeric_outliers": "Training-split IQR fences; values are clipped, not removed",
            "numeric_scaling": "StandardScaler",
            "categorical_missing_values": "Most-frequent imputation",
            "categorical_encoding": "One-hot encoding; unknown future categories ignored",
            "customer_id": "Excluded from model features",
        },
    }

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    _save_eda_plots(data, y.value_counts().sort_index(), numeric_columns)
    _save_model_plots(model, y_test, probabilities, predictions)
    joblib.dump(model, MODEL_PATH)

    predictions_table = pd.DataFrame(
        {
            "actual_purchase": y_test.reset_index(drop=True),
            "predicted_purchase": predictions,
            "purchase_probability": probabilities,
        }
    )
    predictions_table.to_csv(RESULTS_DIR / "test_set_predictions.csv", index=False)

    sample = pd.DataFrame(
        [
            {"Age": 32, "Gender": "Female", "Device_Type": "Mobile", "Traffic_Source": "Email", "Membership_Type": "Premium", "Payment_Method": "UPI", "Session_Duration_Min": 22.0, "Pages_Viewed": 12.0, "Products_Viewed": 8, "Cart_Items": 3, "Cart_Value_INR": 7200.0, "Discount_Percent": 10.0, "Previous_Orders": 5, "Days_Since_Last_Visit": 2.0, "Email_Open_Rate": 0.8, "Customer_Rating": 4.6, "Returns_Last_Year": 0},
            {"Age": 25, "Gender": "Male", "Device_Type": "Mobile", "Traffic_Source": "Social Media", "Membership_Type": "New", "Payment_Method": "Cash on Delivery", "Session_Duration_Min": 3.0, "Pages_Viewed": 2.0, "Products_Viewed": 1, "Cart_Items": 0, "Cart_Value_INR": 0.0, "Discount_Percent": 0.0, "Previous_Orders": 0, "Days_Since_Last_Visit": 30.0, "Email_Open_Rate": 0.1, "Customer_Rating": 3.0, "Returns_Last_Year": 0},
            {"Age": 41, "Gender": "Other", "Device_Type": "Desktop", "Traffic_Source": "Referral", "Membership_Type": "Regular", "Payment_Method": "Credit Card", "Session_Duration_Min": 14.0, "Pages_Viewed": 8.0, "Products_Viewed": 4, "Cart_Items": 1, "Cart_Value_INR": 2300.0, "Discount_Percent": 5.0, "Previous_Orders": 3, "Days_Since_Last_Visit": 8.0, "Email_Open_Rate": 0.5, "Customer_Rating": 4.0, "Returns_Last_Year": 1},
        ]
    )
    sample.to_csv(RESULTS_DIR / "sample_new_customers.csv", index=False)
    sample["purchase_probability"] = model.predict_proba(sample)[:, 1]
    sample["predicted_purchase"] = (sample["purchase_probability"] >= 0.5).astype(int)
    sample.to_csv(RESULTS_DIR / "sample_new_customer_predictions.csv", index=False)

    with (RESULTS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    # Import through its package name before saving the estimator. This ensures
    # joblib records IQRCapper as src.modeling.IQRCapper, not __main__.IQRCapper,
    # so the saved pipeline can be loaded by src.predict in a separate process.
    from src.modeling import run_analysis as portable_run_analysis

    output = portable_run_analysis()
    print(json.dumps(output, indent=2))
