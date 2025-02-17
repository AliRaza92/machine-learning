import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the data
file_path = '/content/final_file_after_filtration (1) - Copy.xlsx'
data = pd.read_excel(file_path)

# Ensure 'year_to_date_return' is numeric and drop NaN values
data['year_to_date_return'] = pd.to_numeric(data['year_to_date_return'], errors='coerce')
data = data.dropna(subset=['year_to_date_return'])

# Select the target series
series = data['year_to_date_return'].values.reshape(-1, 1)

# Scale the data for LSTM
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series)

# Create sequences for LSTM
def create_sequences(data, seq_length=5):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Define sequence length
seq_length = 5  # Number of time steps in each sequence
X, y = create_sequences(series_scaled, seq_length)

# Split into 80% training and 20% testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Make predictions on the training and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Invert scaling for predictions and actual values
y_train_pred = scaler.inverse_transform(y_train_pred)
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_pred = scaler.inverse_transform(y_test_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate evaluation metrics for training set
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

# Calculate evaluation metrics for test set
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

# Display results
print("Training Metrics:")
print("MSE:", train_mse)
print("RMSE:", train_rmse)
print("R^2 Score:", train_r2)

print("\nTesting Metrics:")
print("MSE:", test_mse)
print("RMSE:", test_rmse)
print("R^2 Score:", test_r2)

# Visualization
plt.figure(figsize=(14, 7))

# Plot training predictions vs actual
plt.subplot(1, 2, 1)
plt.plot(y_train, label="Actual Training Values", color="blue")
plt.plot(y_train_pred, label="Predicted Training Values", color="green")
plt.xlabel("Training Data Index")
plt.ylabel("Year to Date Return")
plt.title("Training Set: Actual vs Predicted")
plt.legend()

# Plot testing predictions vs actual
plt.subplot(1, 2, 2)
plt.plot(y_test, label="Actual Testing Values", color="blue")
plt.plot(y_test_pred, label="Predicted Testing Values", color="red")
plt.xlabel("Testing Data Index")
plt.ylabel("Year to Date Return")
plt.title("Test Set: Actual vs Predicted")
plt.legend()

plt.tight_layout()
plt.show()


Training Metrics:
MSE: 0.005243466019673442
RMSE: 0.072411780945323
R^2 Score: 0.04074451391449385

Testing Metrics:
MSE: 0.0055691370574264204
RMSE: 0.07462665112026949
R^2 Score: 0.020829563552196118
