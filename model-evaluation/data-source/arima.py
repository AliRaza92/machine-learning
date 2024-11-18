import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Load the data
file_path = '/content/final_file_after_filtration (1) - Copy.xlsx'
data = pd.read_excel(file_path)

# data feeding 
data_model = data[["fund_name","fund_return_2020","fund_return_2019","fund_return_2018","fund_return_2017","fund_return_2016","fund_return_2015","fund_return_2014","fund_return_2013","fund_return_2012","fund_return_2011","fund_return_2010","fund_return_2009","fund_return_2008","fund_return_2007","fund_return_2006","fund_return_2005","fund_return_2004","fund_return_2003","fund_return_2002","fund_return_2001","fund_return_2000", "year_to_date_return"]];

for I in funds:
	current_data = data_model["fund_name"] == I





# Ensure 'year_to_date_return' is numeric and drop NaN values
data['year_to_date_return'] = pd.to_numeric(data['year_to_date_return'], errors='coerce')
data = data.dropna(subset=['year_to_date_return'])

# Define the target series
series = data['year_to_date_return']

# Split data into 80% training and 20% testing sets
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

# Fit ARIMA model on the training set
model = ARIMA(train, order=(1, 1, 1))
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

# Print evaluation metrics
print("Training Metrics:")
print("MSE:", train_mse)
print("RMSE:", train_rmse)
print("R^2 Score:", train_r2)

print("\nTesting Metrics:")
print("MSE:", test_mse)
print("RMSE:", test_rmse)
print("R^2 Score:", test_r2)

# Plotting the actual vs predicted values
plt.figure(figsize=(14, 7))
plt.plot(series.index, series, label="Actual Values", color="blue")
plt.plot(train.index, train_forecast, label="Training Predictions", color="green")
plt.plot(test.index, test_forecast, label="Test Predictions", color="red")

# Labels and title
plt.xlabel("Index")
plt.ylabel("Year to Date Return")
plt.title("ARIMA Model - Actual vs Predicted Values for Year to Date Return")
plt.legend()
plt.show()


Training Metrics:
MSE: 0.005296133550395496
RMSE: 0.07277453916305823
R^2 Score: 0.02799868428273422

Testing Metrics:
MSE: 0.005833304861288041
RMSE: 0.07637607518908025
R^2 Score: -0.029608953332122878
