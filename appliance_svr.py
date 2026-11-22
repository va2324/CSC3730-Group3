"""Train and evaluate appliance-energy SVR from the UCI-style dataset."""

import argparse
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def read_data(path):
    suffix = path.suffix.lower()
    if suffix == ".zip":
        with ZipFile(path) as archive:
            candidates = [
                name for name in archive.namelist()
                if Path(name).suffix.lower() in {".csv", ".xlsx", ".xls"}
                and not Path(name).name.startswith((".", "~$"))
                and "__MACOSX" not in Path(name).parts
            ]
            for name in candidates:
                with archive.open(name) as stream:
                    frame = read_table(stream, Path(name).suffix.lower())
                if {"date", "appliances"} <= normalized_columns(frame):
                    return frame
        raise ValueError("No table with date and Appliances columns was found in the ZIP.")
    return read_table(path, suffix)


def normalized_columns(frame):
    return {str(column).strip().lower() for column in frame.columns}


def read_table(source, suffix):
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".xlsx", ".xls"}:
        with pd.ExcelFile(source) as workbook:
            for sheet in workbook.sheet_names:
                frame = workbook.parse(sheet)
                if {"date", "appliances"} <= normalized_columns(frame):
                    return frame
        raise ValueError("No worksheet contains date and Appliances columns.")
    raise ValueError("Use a CSV, Excel workbook, or ZIP containing one.")


def prepare(frame):
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if not {"date", "appliances"} <= set(frame.columns):
        raise ValueError("Expected date and Appliances columns.")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    target = pd.to_numeric(frame["appliances"], errors="coerce")
    valid = dates.notna() & np.isfinite(target) & target.ge(0)
    print(f"Dropped {int((~valid).sum())} rows with invalid timestamps or targets.")
    frame = frame.loc[valid].assign(date=dates[valid], appliances=target[valid])
    frame = frame.sort_values("date").set_index("date")
    if frame.index.has_duplicates:
        raise ValueError("Resolve duplicate timestamps before fitting.")
    sensors = [
        col for col in frame.columns
        if re.fullmatch(r"t\d+|t_out|tdewpoint|rh_?\d+|rh_out", col)
    ]
    if not sensors:
        raise ValueError("Expected temperature or humidity columns such as T1 or RH_1.")
    features = frame[sensors].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    hours = frame.index.hour + frame.index.minute / 60
    for name, phase in [
        ("hour", hours / 24),
        ("week", (frame.index.dayofweek + hours / 24) / 7),
    ]:
        features[f"{name}_sin"] = np.sin(2 * np.pi * phase)
        features[f"{name}_cos"] = np.cos(2 * np.pi * phase)
    return features, frame["appliances"]


def summarize(rows):
    error = rows["error_wh"]
    return pd.Series({
        "n": len(rows),
        "actual_mean_wh": rows["actual_wh"].mean(),
        "predicted_mean_wh": rows["predicted_wh"].mean(),
        "mae_wh": error.abs().mean(),
        "rmse_wh": np.sqrt(error.pow(2).mean()),
        "bias_wh": error.mean(),
        "underprediction_pct": 100 * error.lt(0).mean(),
        "high_demand_pct": 100 * rows["high_demand"].mean(),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, default=Path("svr_results"))
    args = parser.parse_args()
    features, target = prepare(read_data(args.data))
    split = int(len(target) * 0.8)
    if split < 12 or split == len(target):
        raise ValueError("Need enough timestamped rows for training and testing.")
    train_x, test_x = features.iloc[:split], features.iloc[split:]
    train_y, test_y = target.iloc[:split], target.iloc[split:]
    peak_threshold = train_y.quantile(0.9)

    # Fit imputation and both scalers inside each chronological validation fold.
    model = TransformedTargetRegressor(
        regressor=make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVR(kernel="rbf", cache_size=256),
        ),
        transformer=StandardScaler(),
    )
    search = GridSearchCV(
        model,
        {"regressor__svr__C": [0.3, 1, 3],
         "regressor__svr__epsilon": [0.1, 0.25],
         "regressor__svr__gamma": ["scale", 0.01]},
        scoring="neg_mean_absolute_error",
        cv=TimeSeriesSplit(n_splits=3),
    )
    search.fit(train_x, train_y)

    results = pd.DataFrame({
        "actual_wh": test_y,
        "predicted_wh": search.predict(test_x).clip(min=0),
        "high_demand": test_y.ge(peak_threshold),
    })
    results["error_wh"] = results["predicted_wh"] - results["actual_wh"]
    # A time-of-week baseline uses training medians and a global fallback.
    medians = train_y.groupby([train_y.index.dayofweek, train_y.index.hour]).median()
    slots = pd.MultiIndex.from_arrays([test_y.index.dayofweek, test_y.index.hour])
    baseline = results.copy()
    baseline["predicted_wh"] = medians.reindex(slots).fillna(train_y.median()).to_numpy()
    baseline["error_wh"] = baseline["predicted_wh"] - baseline["actual_wh"]
    results["baseline_wh"] = baseline["predicted_wh"]
    records = {}
    for name, rows in [("SVR", results), ("Time-of-week median", baseline)]:
        records[name, "All"] = summarize(rows)
        records[name, "High demand"] = summarize(rows.loc[rows["high_demand"]])
        records[name, "Other"] = summarize(rows.loc[~rows["high_demand"]])
    metrics = pd.DataFrame(records).T.rename_axis(["model", "demand"])
    hourly = pd.DataFrame({
        hour: summarize(rows) for hour, rows in results.groupby(results.index.hour)
    }).T.rename_axis("hour")
    peak = results.loc[results["high_demand"]]
    peak_by_hour = pd.DataFrame(
        {hour: summarize(rows) for hour, rows in peak.groupby(peak.index.hour)}
    ).T.rename_axis("hour")
    daily = results.groupby(results.index.normalize()).agg(
        actual_wh=("actual_wh", "sum"),
        predicted_wh=("predicted_wh", "sum"),
        intervals=("actual_wh", "size"),
        high_demand_intervals=("high_demand", "sum"),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output / "test_predictions.csv", index_label="date")
    metrics.to_csv(args.output / "metrics.csv")
    hourly.to_csv(args.output / "hourly_patterns.csv")
    peak_by_hour.to_csv(args.output / "high_demand_by_hour.csv")
    daily.to_csv(args.output / "daily_patterns.csv", index_label="date")
    results.nsmallest(20, "error_wh").to_csv(
        args.output / "largest_underpredictions.csv", index_label="date"
    )
    print(f"Training rows: {len(train_y):,}. Test rows: {len(test_y):,}.")
    print(f"Test period: {test_y.index.min()} through {test_y.index.max()}")
    print(f"High demand: actual consumption >= {peak_threshold:g} Wh per interval")
    print(f"Selected parameters: {search.best_params_}")
    print(metrics.round(2).to_string())
    print("\nHours with the highest mean actual test consumption")
    print(hourly.nlargest(5, "actual_mean_wh").round(2).to_string())
    if not peak_by_hour.empty:
        print("\nHours with the most high-demand test intervals")
        print(peak_by_hour.nlargest(5, "n").round(2).to_string())


if __name__ == "__main__":
    main()
