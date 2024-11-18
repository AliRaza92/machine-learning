
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.tsa.ar_model import ar_select_order


# Load the data
file_path = '/content/final_file_after_filtration (1) - Copy.xlsx'
data = pd.read_excel(file_path)

# data feeding
data_model = data[["fund_long_name","fund_return_2020","fund_return_2019","fund_return_2018","fund_return_2017","fund_return_2016","fund_return_2015","fund_return_2014","fund_return_2013","fund_return_2012","fund_return_2011","fund_return_2010","fund_return_2009","fund_return_2008","fund_return_2007","fund_return_2006","fund_return_2005","fund_return_2004","fund_return_2003","fund_return_2002","fund_return_2001","fund_return_2000", "year_to_date_return"]];
funds = data_model["fund_long_name"]

all_score = []

for I in funds[1:5]:
  current_data = data_model[data_model["fund_long_name"] == I]
  # print(current_data)
  x = current_data.drop(["year_to_date_return"], axis=1).T.values
  y = current_data["year_to_date_return"]
  print(x)
  xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1)
  p = ar_select_order(x)
  scores = []
  for q in [0,1,2,3,4]:
    for d in [0,1,2,3,4]:

        # Fit ARIMA model on the training set
        # Fit ARIMA model on the training set


        model = ARIMA(train, order=(p, q, d))
        model_fit = model.fit()

        # Forecast for both training and test sets
        train_forecast = model_fit.fittedvalues
        test_forecast = model_fit.forecast(steps=len(test))

        # Calculate evaluation metrics for training and test sets
        train_mse = mean_squared_error(train, train_forecast)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(train, train_forecast)

        test_mse = mean_squared_error(test, test_forecast)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(test, test_forecast)

        scores.appendall_scores(test_r2,train_r2,test_rmse,train_rmse,p,q,d,data_model["fund_name"])


        # Print evaluation metrics
        # print("Training Metrics:")
        # print("MSE:", train_mse)
        # print("RMSE:", train_rmse)
        # print("R^2 Score:", train_r2)

        # print("\nTesting Metrics:")
        # print("MSE:", test_mse)
        # print("RMSE:", test_rmse)
        # print("R^2 Score:", test_r2)

all_score.append(scores)

print(all_score)


exit()
