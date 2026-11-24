# CSC3730-Group3

## Introduction
This group project attempts to solve a regression problem which involves predicting household energy consumption using a systematic machine learning framework. The project begins with an exploratory analysis of the Appliances Energy Prediction Dataset from UC Irvine. This initial step guides the selection of appropriate model strategies and evaluation metrics. All models are implemented through the Scikit-learn library in Python, and grid search was performed on the models to determine the hyperparameters that achieve the best validation score. Final evaluation of the models was performed using the scoring metrics mean absolute error (MAE), mean squared error (MSE), & root mean squared error (RMSE).

## Dataset
[Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)

This dataset consists of 28 features and 1 target feature "Appliances", which represents the household's energy consumption in watt-hours (Wh). The features mainly consist of temperature and humidity measurements in various rooms of the house. This data was monitored with a ZigBee wireless network, with each wireless node transmitting temperature and humidity data every 3.3 minutes. The data was averaged over 10-minute periods. Other features represent weather data like pressure, wind speed, visibility, and dew point, which were collected at the nearest airport station (Chievres Airport, Belgium) and downloaded from a public dataset from the Reliable Prognosis website (rp5.ru). There are also two random variable features for testing regression models and filtering out non-predictive attributes as well as a date feature that will need to be encoded into numerical features when building the regression model.

## Model Strategies
The model strategies that were implemented and evaluated include: 
### Linear Regression
### KNN Regression

Install all dependencies (requirements.txt) in your Python environment, select it as your kernel for the notebook file [knn.ipynb](knn.ipynb), and run all cells. This file imports the `energydata.csv` file and runs the preprocessing & grid search and prints the final evaluation score metrics for the KNN Regressor model. 

### Support Vector Regression
Install the required packages in the Python environment used to run the script.

python -m pip install numpy pandas scikit-learn matplotlib openpyxl xlrd

Place appliance_svr.py and energydata_complete.csv in the same folder, open a terminal in that folder, and run the following command.

python appliance_svr.py


### Decision Tree Regression

Install the dependencies in your Python environment and run the decision tree from the project folder.

```bash
python -m pip install -r requirements.txt
python decision_tree.py
```

You can also open [decision_tree.ipynb](decision_tree.ipynb), select the same Python environment as the kernel, and run all cells. Both use the repository's `energydata.csv` and write metrics, predictions, and plots to `outputs/decision_tree/`.

The workflow trains on the earliest 80% of the data, tunes the tree with five time-ordered folds within that training period, and compares it with a training-mean baseline on the latest 20%. Its scores use a different split from the randomized split in `gridsearch.py`.

### Multi-Layer Perceptron
### Random Forest
### Bagging Regressor

Install all dependencies (requirements.txt) in your Python environment, select it as your kernel for the notebook file [knn.ipynb](knn.ipynb), and run all cells. This file imports the `energydata.csv` file and runs the preprocessing & grid search and writes metrics, predictions, & saves plots to the `plots` directory. 

Grid search is done on 5 different BaggingRegressor models, each with a different base estimator: DecisionTreeRegressor, KNeighborsRegressor, LinearRegression, MLPRegressor, & SVR. The final evaluation metrics of these 5 models were plotted and compared to see which base estimator gives BaggingRegressor the best performance. 

### Gradient Boosting
