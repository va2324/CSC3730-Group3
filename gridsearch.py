import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# Load df & check for duplicate rows & null values
df = pd.read_csv("energydata.csv")
print(df.duplicated().sum())
print(df.isna().sum())

# Convert date feature into separate numerical columns
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['hour'] = df['date'].dt.hour
df['minute'] = df['date'].dt.minute
df = df.drop(['date'], axis=1)
print(df[['year', 'month', 'day', 'hour', 'minute']])

# Splitting the dataset
X = df.drop(["Appliances"], axis=1)
y = df["Appliances"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)




parameter_grid = {
    "model__n_estimators": [10, 25, 50, 75, 100],
    "model__max_samples": [0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    "model__max_features": [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
}

kf = KFold(n_splits=5, shuffle=True)

scores = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error']

pipeline_mlp = Pipeline([
    ("scaler", StandardScaler()),
    ("model", BaggingRegressor(estimator=MLPRegressor(), bootstrap=True, bootstrap_features=False, oob_score=True))
])

grid_search_mlp = RandomizedSearchCV(pipeline_mlp, parameter_grid, cv=kf, n_iter=50, scoring=scores, refit='neg_mean_squared_error', n_jobs=-2, random_state=42)

grid_search_mlp.fit(X_train, y_train)

print("Best Parameters")
print(grid_search_mlp.best_params_)
print()
print("Best Cross-Validation Accuracy")
print(grid_search_mlp.best_score_)
print()
print("Best OOB Score")
print(grid_search_mlp.best_estimator_.named_steps["model"].oob_score_)

results = pd.DataFrame(grid_search_mlp.cv_results_)
results['mean_test_MAE'] = -results['mean_test_neg_mean_absolute_error']
results['mean_test_MSE'] = -results['mean_test_neg_mean_squared_error']
results['mean_test_RMSE'] = -results['mean_test_neg_root_mean_squared_error']
print(results.loc[grid_search_mlp.best_index_][['params', 'mean_test_MAE', 'mean_test_MSE', 'mean_test_RMSE', 'mean_test_r2']])

model_MLP = grid_search_mlp.best_estimator_
y_pred = model_MLP.predict(X_test)
MAE_MLP = mean_absolute_error(y_test, y_pred)
MSE_MLP = mean_squared_error(y_test, y_pred)
RMSE_MLP = root_mean_squared_error(y_test, y_pred)
print("MAE: ", MAE_MLP)
print("MSE: ", MSE_MLP)
print("RMSE: ", RMSE_MLP)





pipeline_svr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", BaggingRegressor(estimator=SVR(), bootstrap=True, bootstrap_features=False, oob_score=True))
])

grid_search_svr = RandomizedSearchCV(pipeline_svr, parameter_grid, cv=kf, n_iter=50, scoring=scores, refit='neg_mean_squared_error', n_jobs=-2, random_state=42)

grid_search_svr.fit(X_train, y_train)

print("Best Parameters")
print(grid_search_svr.best_params_)
print()
print("Best Cross-Validation Accuracy")
print(grid_search_svr.best_score_)
print()
print("Best OOB Score")
print(grid_search_svr.best_estimator_.named_steps["model"].oob_score_)

results = pd.DataFrame(grid_search_svr.cv_results_)
results['mean_test_MAE'] = -results['mean_test_neg_mean_absolute_error']
results['mean_test_MSE'] = -results['mean_test_neg_mean_squared_error']
results['mean_test_RMSE'] = -results['mean_test_neg_root_mean_squared_error']
print(results.loc[grid_search_svr.best_index_][['params', 'mean_test_MAE', 'mean_test_MSE', 'mean_test_RMSE', 'mean_test_r2']])


model_SVR = grid_search_svr.best_estimator_
y_pred = model_SVR.predict(X_test)
MAE_SVR = mean_absolute_error(y_test, y_pred)
MSE_SVR = mean_squared_error(y_test, y_pred)
RMSE_SVR = root_mean_squared_error(y_test, y_pred)
print("MAE: ", MAE_SVR)
print("MSE: ", MSE_SVR)
print("RMSE: ", RMSE_SVR)