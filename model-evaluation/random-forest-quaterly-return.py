import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the data
file_path = '/content/final_file_after_filtration (1) - Copy.xlsx'
data = pd.read_excel(file_path)

# Identify quarterly return columns
quarterly_columns = [col for col in data.columns if 'fund_return' in col and '_q' in col]

# Define a function to process each quarterly return column
results = {}
for quarter in quarterly_columns:
    # Prepare data for this specific quarterly return column
    X = data.drop(columns=quarterly_columns)  # Drop other quarterly columns as features
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

    # Split data into training and test sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)

    # Predict on the test set and evaluate
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    r2 = r2_score(y_test, y_pred)

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
