import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
file_path = '/content/final_file_after_filtration (1) - Copy.xlsx'
data = pd.read_excel(file_path)

# Identify quarterly return columns
quarterly_columns = [col for col in data.columns if 'fund_return' in col and '_q' in col]

# Dictionary to store results
results = {}

# Define a function to process each quarterly return column
for quarter in quarterly_columns:
    # Prepare data for this specific quarterly return column
    X = data.drop(columns=quarterly_columns)  # Drop other quarterly columns from features
    y = data[quarter]  # Target for the specific quarter

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

    # Reshape data for LSTM (assuming each sample is one time step)
    time_steps = 1
    X_reshaped = np.reshape(X_scaled, (X_scaled.shape[0], time_steps, X_scaled.shape[1]))

    # Split data into training and test sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_scaled, test_size=0.2, random_state=42)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model with 50 epochs
    model.fit(X_train, y_train, batch_size=16, epochs=50, validation_data=(X_test, y_test), verbose=0)

    # Predict and evaluate the model
    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler_y.inverse_transform(y_pred)
    y_test_rescaled = scaler_y.inverse_transform(y_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)

    # Store results for this quarter
    results[quarter] = {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

# Output results for each quarter
for quarter, metrics in results.items():
    print(f"Results for {quarter}:")
    print("Mean Squared Error (MSE):", metrics['MSE'])
    print("Root Mean Squared Error (RMSE):", metrics['RMSE'])
    print("R^2 Score:", metrics['R2'])
    print("\n")


