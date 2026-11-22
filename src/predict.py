"""Score a CSV of new customer sessions with the saved Logistic Regression pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.modeling import MODEL_PATH, ROOT


def predict_file(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    model = joblib.load(MODEL_PATH)
    new_data = pd.read_csv(input_csv)
    id_values = new_data["Customer_ID"].copy() if "Customer_ID" in new_data.columns else None
    new_data = new_data.drop(columns=["Customer_ID", "Purchase"], errors="ignore")
    expected = list(model.named_steps["preprocessing"].feature_names_in_)
    missing = [column for column in expected if column not in new_data.columns]
    extra = [column for column in new_data.columns if column not in expected]
    if missing:
        raise ValueError(f"Input is missing required predictors: {', '.join(missing)}")
    if extra:
        print(f"Ignoring unrecognized columns: {', '.join(extra)}")
    new_data = new_data[expected]
    probabilities = model.predict_proba(new_data)[:, 1]
    result = pd.DataFrame({
        "purchase_probability": probabilities,
        "predicted_purchase": (probabilities >= 0.5).astype(int),
    })
    if id_values is not None:
        result.insert(0, "Customer_ID", id_values.to_numpy())
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict purchase probability for new customer sessions.")
    parser.add_argument("input_csv", type=Path, help="CSV containing customer predictors and no target column")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "predictions.csv", help="Where to save prediction results")
    args = parser.parse_args()
    result = predict_file(args.input_csv, args.output)
    print(result.to_string(index=False))
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
