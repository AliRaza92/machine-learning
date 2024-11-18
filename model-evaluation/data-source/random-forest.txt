import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load the data
file_path = 'contentfinal_file_after_filtration (1) - Copy.xlsx'
data = pd.read_excel(file_path)

# Ensure 'year_to_date_return' is numeric and drop NaN values
data['year_to_date_return'] = pd.to_numeric(data['year_to_date_return'], errors='coerce')
data = data.dropna(subset=['year_to_date_return'])

# Separate target and features
target = data['year_to_date_return']
features = data.drop(columns=['year_to_date_return'])

# Ensure all features are numeric and handle categorical data if present
features = features.apply(pd.to_numeric, errors='coerce').fillna(0)  # Convert all to numeric and fill NaNs
features = pd.get_dummies(features, drop_first=True)  # One-hot encode categorical variables

# Split the data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on both training and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate evaluation metrics for training set
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

# Calculate evaluation metrics for test set
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

# Display results
print(Training Metrics)
print(MSE, train_mse)
print(RMSE, train_rmse)
print(R^2 Score, train_r2)

print(nTesting Metrics)
print(MSE, test_mse)
print(RMSE, test_rmse)
print(R^2 Score, test_r2)

# Visualization
plt.figure(figsize=(14, 7))

# Plot training predictions vs actual
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, color=green, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
plt.xlabel(Actual Training Values)
plt.ylabel(Predicted Training Values)
plt.title(Training Set Actual vs Predicted)

# Plot testing predictions vs actual
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, color=blue, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel(Actual Testing Values)
plt.ylabel(Predicted Testing Values)
plt.title(Test Set Actual vs Predicted)

plt.tight_layout()
plt.show()



Training Metrics:
MSE: 8.857092226205762e-05
RMSE: 0.009411212581918316
R^2 Score: 0.9835188757662178

Testing Metrics:
MSE: 0.0008626063558269956
RMSE: 0.029370160977206027
R^2 Score: 0.8552815399395309
