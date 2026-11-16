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
### Support Vector Regression
### Decision Tree Regression
### Multi-Layer Perceptron
### Random Forest
### Bagging Regressor
### Gradient Boosting
