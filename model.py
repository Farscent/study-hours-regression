"""Learn regression, a train/test split, and an honest mean-value baseline."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent
FEATURES = ["study_hours"]

def load_students(path):
    data = pd.read_csv(path)
    required = {"student_id", "study_hours", "exam_score"}
    if not required.issubset(data.columns):
        raise ValueError(f"Required columns: {sorted(required)}")
    if len(data) < 20:
        raise ValueError("Use at least 20 rows for this learning exercise.")
    if data["student_id"].isna().any() or data["student_id"].duplicated().any():
        raise ValueError("Student IDs must be present and unique.")
    for column in ["study_hours", "exam_score"]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    hours = data["study_hours"].dropna()
    if not hours.between(0, 24).all() or hours.nunique() < 3:
        raise ValueError("Known study hours must be in [0, 24], with at least 3 distinct values.")
    if not data["exam_score"].between(0, 100).all():
        raise ValueError("Every exam score must be a finite number in [0, 100].")
    return data

def build_model():
    # Fit this pipeline only on training data. The median must not see the test set.
    return make_pipeline(SimpleImputer(strategy="median"), LinearRegression())

def metrics(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }

def train_and_evaluate(data):
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    if train["study_hours"].dropna().nunique() < 2:
        raise ValueError("The training split needs at least two different known study-hour values.")
    model = build_model()
    baseline = make_pipeline(SimpleImputer(strategy="median"), DummyRegressor(strategy="mean"))
    model.fit(train[FEATURES], train["exam_score"])
    baseline.fit(train[FEATURES], train["exam_score"])
    predicted = model.predict(test[FEATURES])
    scores = {
        "training_rows": len(train), "test_rows": len(test),
        "linear_regression": metrics(test["exam_score"], predicted),
        "mean_baseline": metrics(test["exam_score"], baseline.predict(test[FEATURES])),
        "training_median_hours": float(model.named_steps["simpleimputer"].statistics_[0]),
        "intercept": float(model.named_steps["linearregression"].intercept_),
        "slope": float(model.named_steps["linearregression"].coef_[0]),
    }
    predictions = test[["student_id", "study_hours", "exam_score"]].copy()
    predictions["predicted_score"] = predicted
    predictions["residual"] = predictions["exam_score"] - predicted
    return model, scores, train, predictions

def run(input_path=ROOT / "data/students.csv", output_dir=ROOT / "outputs", hours=None):
    data = load_students(input_path)
    model, scores, train, predictions = train_and_evaluate(data)
    if hours is not None:
        low, high = train["study_hours"].min(), train["study_hours"].max()
        if not np.isfinite(hours) or not low <= hours <= high:
            raise ValueError(f"Use hours within the observed training range [{low}, {high}].")
        scores["example_prediction"] = {
            "hours": hours,
            "score": float(model.predict(pd.DataFrame({"study_hours": [hours]}))[0]),
        }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    grid = pd.DataFrame({"study_hours": np.linspace(train.study_hours.min(), train.study_hours.max(), 100)})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(train.study_hours, train.exam_score, alpha=0.45, label="Training rows", color="#246b74")
    axes[0].scatter(predictions.study_hours, predictions.exam_score, label="Test rows", color="#d67b36")
    axes[0].plot(grid.study_hours, model.predict(grid), color="#283342", label="Fitted line")
    axes[0].set(xlabel="Study hours per week", ylabel="Exam score / 100", title="A simple linear relationship")
    axes[0].legend(fontsize=8)
    axes[1].scatter(predictions.predicted_score, predictions.residual, color="#246b74")
    axes[1].axhline(0, color="#283342", linestyle="--")
    axes[1].set(xlabel="Predicted score", ylabel="Actual minus predicted", title="Held-out residuals")
    fig.suptitle("Synthetic students · regression learning exercise")
    fig.tight_layout()
    fig.savefig(output_dir / "regression.png", dpi=160)
    plt.close(fig)
    model_mae = scores["linear_regression"]["mae"]
    baseline_mae = scores["mean_baseline"]["mae"]
    report = (
        "# Regression findings\n\n"
        "This is a synthetic learning exercise, not evidence about real students.\n\n"
        f"Training rows: {len(train)}. Test rows: {len(predictions)}. Random seed: 42.\n\n"
        "| Model | Test MAE | Test RMSE | Test R² |\n|---|---:|---:|---:|\n")
    for name in ["linear_regression", "mean_baseline"]:
        result = scores[name]
        report += f"| {name} | {result['mae']:.2f} | {result['rmse']:.2f} | {result['r2']:.3f} |\n"
    report += (
        f"\nRegression {'improves on' if model_mae < baseline_mae else 'does not improve on'} "
        "the training-mean baseline on this fixed test split. "
        "MAE is the average absolute error in score points. "
        "The generator deliberately creates a linear relationship, so this is not a claim of real-world prediction quality.\n")
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(scores, indent=2))
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/students.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--hours", type=float, help="Optional example prediction inside the training range.")
    args = parser.parse_args()
    try:
        run(args.input, args.output, args.hours)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")
