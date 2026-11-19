"""Predict appliance energy use with gradient-boosted decision trees."""

from numpy import sqrt
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from decision_tree import FEATURES, ROOT, load_data


def main():
    data = load_data(ROOT / "energydata.csv")
    split = int(len(data) * 0.8)
    train, test = data.iloc[:split], data.iloc[split:]
    X_train, y_train = train[FEATURES], train["Appliances"]
    X_test, y_test = test[FEATURES], test["Appliances"]

    # Disable the random validation split used by automatic early stopping.
    search = GridSearchCV(
        HistGradientBoostingRegressor(early_stopping=False, random_state=42),
        {
            "max_iter": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3],
        },
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_mean_absolute_error",
        error_score="raise",
    )
    search.fit(X_train, y_train)
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)

    print(f"Training rows = {len(train):,} | Test rows = {len(test):,}")
    print(f"Best parameters = {search.best_params_}")
    print(f"Cross-validation MAE = {-search.best_score_:.2f} Wh")
    for name, model in [("Training mean baseline", baseline), ("Gradient boosting", search)]:
        predicted = model.predict(X_test)
        mae = mean_absolute_error(y_test, predicted)
        mse = mean_squared_error(y_test, predicted)
        print(f"{name} | MAE = {mae:.2f} Wh | MSE = {mse:.2f} Wh² | RMSE = {sqrt(mse):.2f} Wh")


if __name__ == "__main__":
    main()
