# CSC3730-Group3

## Introduction
This group project attempts to solve a regression problem which involves predicting household energy consumption using a
systematic machine learning framework. The project begins with an exploratory analysis of the Appliances Energy Prediction
Dataset from UC Irvine. This initial step guides the selection of appropriate model strategies and evaluation metrics.
All models are implemented through the Scikit-learn library in Python, and grid search was performed on the models to 
determine the hyperparameters that achieve the best validation score. Final evaluation of the models was performed using
the scoring metrics mean absolute error (MAE), mean squared error (MSE), & root mean squared error (RMSE).

## Dataset
[Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)

This dataset consists of 28 features and 1 target feature "Appliances", which represents the household's energy consumption
in watt-hours (Wh). The features mainly consist of temperature and humidity measurements in various rooms of the house.
This data was monitored with a ZigBee wireless network, with each wireless node transmitting temperature and humidity data
every 3.3 minutes. The data was averaged over 10-minute periods. Other features represent weather data like pressure,
wind speed, visibility, and dew point, which were collected at the nearest airport station (Chievres Airport, Belgium) 
and downloaded from a public dataset from the Reliable Prognosis website (rp5.ru). There are also two random variable 
features for testing regression models and filtering out non-predictive attributes as well as a date feature that will 
need to be encoded into numerical features when building the regression model.

## Model Strategies
The model strategies that were implemented and evaluated include: 

### Linear Regression

This notebook predicts household appliance energy use in watt-hours (Wh) for each 10-minute interval in the UCI Appliances
Energy Prediction dataset.
Run the notebook
1. From the repository root, install the project dependencies.

```bash
   python -m pip install -r requirements.txt
```
2. Open [linear_regression.ipynb](linear_regression.ipynb) in Jupyter or VS Code and select the Python environment where
you installed the dependencies.
3. Run all cells from top to bottom. The notebook reads [energydata.csv](energydata.csv) from the repository root and 
displays the metrics, diagnostic plots, and fitted coefficients.

The model uses indoor sensor readings, outdoor weather measurements, lighting use, and month, day, hour, and minute as 
predictors. It excludes the constant year and the two random variables (rv1 and rv2). Scikit-learn's LinearRegression fits
the model using ordinary least squares.
### KNN Regression

Install all dependencies in [requirements.txt](requirements.txt) in your Python environment, select it as your kernel for the notebook
file [knn.ipynb](knn.ipynb), and run all cells. This file imports the `energydata.csv` file and runs the preprocessing & 
grid search and prints the final evaluation score metrics for the KNN Regressor model. 

### Support Vector Regression
Install the required packages in the Python environment used to run the script.

```bash
python -m pip install numpy pandas scikit-learn matplotlib openpyxl xlrd
```

Place appliance_svr.py and energydata_complete.csv in the same folder, open a terminal in that folder, and run the following command.

```bash
python appliance_svr.py
```

### Decision Tree Regression

Install the dependencies in your Python environment and run the decision tree from the project folder.

```bash
python -m pip install -r requirements.txt
python decision_tree.py
```

You can also open [decision_tree.ipynb](decision_tree.ipynb), select the same Python environment as the kernel, and run all cells. Both 
use the repository's `energydata.csv` and write metrics, predictions, and plots to `outputs/decision_tree/`.

The workflow trains on the earliest 80% of the data, tunes the tree with five time-ordered folds within that training period,
and compares it with a training-mean baseline on the latest 20%. Its scores use a different split from the randomized 
split in `gridsearch.py`.

`metrics.csv` reports MAE, MSE, RMSE, and R². `high_demand_errors.csv` reports MAE and mean signed error for test intervals
at or above the training set's 90th percentile and for the remaining intervals. Mean signed error is predicted usage minus
actual usage, so negative values indicate underprediction on average.

### Multi-Layer Perceptron

Install the dependencies in [requirements.txt](requirements.txt) in your Python environment, select it as your kernel for the notebook
file [mlp.ipynb](mlp.ipynb), and run all cells. This file imports the `energydata.csv` file, runs the same preprocessing
& 70/30 split as the KNN and Bagging notebooks, and saves plots to the `plots` directory.

The notebook first evaluates a training-mean baseline and an `MLPRegressor` with default parameters. It then runs a 
randomized search (30 parameter combinations, 5-fold cross validation) over hidden layer sizes, activation function,
`alpha` (L2 regularization), initial learning rate, and batch size, with early stopping enabled. The tuned model's MAE,
MSE, & RMSE on the test set are compared against the baseline, the default MLP, KNN, and the Bagging Regressor with an
MLP base estimator, and train vs test error is compared to check for overfitting. The grid search takes roughly 10-30
minutes depending on your machine.

### Random Forest

Install all dependencies in [requirements.txt](requirements.txt) in your Python environment, select it as your kernel for the notebook
file [randomForest.ipynb](randomForest.ipynb), and run all cells. This file imports the `energydata.csv` file and runs the same preprocessing 
as the rest of the files. Then, it uses gridsearch and a five fold k-split to determine the best parameters for n_estimators, max_depth, min_samples, and max_features, totaling 54 possible combinations. The gridsearch took about 13 minutes to run on my machine, but it could take longer. After all that, it prints the best CV RMSE, MAE, MSE, and RMSE, alongside the 10 most important features.

### Bagging Regressor

Install all dependencies in [requirements.txt](requirements.txt) in your Python environment, select it as your kernel for the notebook
file [knn.ipynb](knn.ipynb), and run all cells. This file imports the `energydata.csv` file and runs the preprocessing &
grid search and writes metrics, predictions, & saves plots to the `plots` directory. 

Grid search is done on 5 different BaggingRegressor models, each with a different base estimator: DecisionTreeRegressor,
KNeighborsRegressor, LinearRegression, MLPRegressor, & SVR. The final evaluation metrics of these 5 models were plotted
and compared to see which base estimator gives BaggingRegressor the best performance. 

### Gradient Boosting

From the project folder, install the dependencies and run the gradient boosting script.

```bash
python -m pip install -r requirements.txt
python gradient_boosting.py
```

The script reads `energydata.csv` and prints the best hyperparameters, cross-validation MAE, and test MAE, MSE, and RMSE
for both the gradient boosting model and the training-mean baseline.
