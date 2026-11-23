"""Decision tree regression. Run: python decision_tree.py"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "decision_tree"
FEATURES = ([f"T{i}" for i in range(1, 10)]
            + [f"RH_{i}" for i in range(1, 10)]
            + ["T_out", "RH_out", "hour", "day_of_week", "month", "is_weekend"])

def load_data(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date").reset_index(drop=True)
    if df["date"].duplicated().any():
        raise ValueError("Duplicate timestamps found; investigate before splitting.")
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    numeric = df[FEATURES + ["Appliances"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Missing or non-finite measurements found; investigate before training.")
    if len(df) < 100:
        raise ValueError("At least 100 rows are needed for this workflow.")
    return df

def metrics(actual, predicted):
    mse = float(mean_squared_error(actual, predicted))
    return {"MAE_Wh": float(mean_absolute_error(actual, predicted)),
            "MSE_Wh2": mse,
            "RMSE_Wh": float(np.sqrt(mse)),
            "R2": float(r2_score(actual, predicted))}

def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def run_project(data_path=None, output_dir=None):
    out = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    df = load_data(Path(data_path) if data_path else ROOT / "energydata.csv")
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]
    X_train, y_train = train[FEATURES], train["Appliances"]
    X_test, y_test = test[FEATURES], test["Appliances"]

    # Explore training data only; reserve the test set for final evaluation.
    train.set_index("date")["Appliances"].resample("D").mean().plot(figsize=(11, 4))
    plt.title("Training data: daily mean appliance energy")
    plt.ylabel("Wh per 10-minute interval")
    save_plot(out / "daily_usage.png")
    train.groupby("hour")["Appliances"].mean().plot.bar(figsize=(10, 4))
    plt.ylabel("Mean Wh per 10-minute interval")
    plt.title("Training data: usage by hour")
    save_plot(out / "hourly_usage.png")

    # All model selection uses earlier-to-later folds within training data.
    search = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        {"max_depth": [3, 5, 8, 12], "min_samples_leaf": [10, 30, 60]},
        cv=TimeSeriesSplit(n_splits=5), scoring="neg_mean_absolute_error")
    search.fit(X_train, y_train)
    tree = search.best_estimator_
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    tree_pred = tree.predict(X_test)
    results = pd.DataFrame([
        {"Model": "Training mean baseline", **metrics(y_test, baseline_pred)},
        {"Model": "Tuned decision tree", **metrics(y_test, tree_pred)}])
    results.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(search.cv_results_).to_csv(out / "cross_validation.csv", index=False)

    threshold = float(y_train.quantile(0.90))
    predictions = test[["date", "Appliances"]].rename(columns={"Appliances": "actual_Wh"}).copy()
    predictions["tree_Wh"] = tree_pred
    predictions["baseline_Wh"] = baseline_pred
    predictions["high_demand"] = y_test.to_numpy() >= threshold
    predictions.to_csv(out / "predictions.csv", index=False)
    groups = []
    for high, group in predictions.groupby("high_demand"):
        for model, column in [("Decision tree", "tree_Wh"), ("Baseline", "baseline_Wh")]:
            signed_errors = group[column] - group["actual_Wh"]
            groups.append({"Model": model, "High_demand": bool(high), "Count": len(group),
                           "MAE_Wh": float(signed_errors.abs().mean()),
                           "Mean_signed_error_Wh": float(signed_errors.mean())})
    pd.DataFrame(groups).to_csv(out / "high_demand_errors.csv", index=False)

    importance = pd.Series(tree.feature_importances_, index=FEATURES).sort_values(ascending=False)
    importance.rename("importance").to_csv(out / "feature_importance.csv")
    importance.head(10).sort_values().plot.barh(figsize=(8, 5))
    plt.title("Tree impurity-based feature importance")
    save_plot(out / "feature_importance.png")
    plt.figure(figsize=(11, 4))
    plt.plot(test["date"].iloc[:200], y_test.iloc[:200], label="Actual")
    plt.plot(test["date"].iloc[:200], tree_pred[:200], label="Decision tree")
    plt.ylabel("Wh per 10-minute interval")
    plt.xticks(rotation=25)
    plt.legend()
    save_plot(out / "actual_vs_predicted.png")
    plt.figure(figsize=(22, 9))
    plot_tree(tree, feature_names=FEATURES, max_depth=2, filled=True, rounded=True, fontsize=8)
    plt.title("First three levels; deeper branches omitted")
    save_plot(out / "tree_preview.png")
    details = {"best_parameters": search.best_params_, "cv_MAE_Wh": -float(search.best_score_),
               "training_metrics": metrics(y_train, tree.predict(X_train)),
               "high_demand_threshold_Wh": threshold,
               "training_rows": len(train), "test_rows": len(test),
               "training_end": str(train.date.iloc[-1]), "test_start": str(test.date.iloc[0])}
    (out / "run_details.json").write_text(json.dumps(details, indent=2))
    print(results.to_string(index=False))
    print("Best settings:", search.best_params_)
    print("Saved results to", out)
    return results, details

if __name__ == "__main__":
    run_project()
