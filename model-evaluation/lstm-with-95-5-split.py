import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
file_path = '/content/final_file_after_filtration (1) - Copy.xlsx'
data = pd.read_excel(file_path)

# Define the target column and features
target_column = 'year_to_date_return'
X = data.drop(columns=[target_column])
y = data[target_column]

# Convert categorical columns to numeric using Label Encoding
X = X.apply(lambda col: LabelEncoder().fit_transform(col.astype(str)) if col.dtype == 'object' else col)

# Drop columns with more than 50% missing values
threshold = 0.5 * len(X)
X = X.dropna(thresh=threshold, axis=1)

# Fill remaining missing values: median for numeric columns and mode for categorical
X = X.apply(lambda col: col.fillna(col.median()) if col.dtype in ['float64', 'int64'] else col.fillna(col.mode()[0]))

# Convert datetime columns to year and drop the original datetime columns
datetime_columns = X.select_dtypes(include=['datetime64']).columns
for col in datetime_columns:
    X[col + '_year'] = X[col].dt.year
    X = X.drop(columns=[col])

# Ensure target variable aligns with the filtered feature set
y = y[X.index]

# Scale data for LSTM
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Reshape data for LSTM (assuming each sample is one time step for simplicity)
time_steps = 1
X_reshaped = np.reshape(X_scaled, (X_scaled.shape[0], time_steps, X_scaled.shape[1]))

# Split data into training and test sets (95% training, 5% testing)
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_scaled, test_size=0.05, random_state=42)

# Build LSTM model with dropout layers and early stopping
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with increased epochs
model.fit(X_train, y_train, batch_size=16, epochs=50, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Predict and evaluate the model
y_pred = model.predict(X_test)
y_pred_rescaled = scaler_y.inverse_transform(y_pred)
y_test_rescaled = scaler_y.inverse_transform(y_test)

# Calculate performance metrics
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_rescaled, y_pred_rescaled)

# Output results
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R^2 Score:", r2)
