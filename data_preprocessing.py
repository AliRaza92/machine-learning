import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the file
file_path = '/content/filtered-file.xlsx'  # Replace with the actual path
df = pd.read_excel(file_path)

# Step 2: Handle missing data
# Impute missing values for numeric columns with the median and for categorical columns with the mode
for column in df.columns:
    if df[column].dtype in ['float64', 'int64']:
        df[column].fillna(df[column].median(), inplace=True)
    elif df[column].dtype == 'object':
        df[column].fillna(df[column].mode()[0], inplace=True)

# Step 3: Cap Outliers
cap_lower_bound = df[['initial_investment', 'subsequent_investment']].quantile(0.01)
cap_upper_bound = df[['initial_investment', 'subsequent_investment']].quantile(0.99)

# Apply capping
df['initial_investment'] = df['initial_investment'].clip(lower=cap_lower_bound['initial_investment'], 
                                                                                        upper=cap_upper_bound['initial_investment'])
df['subsequent_investment'] = df['subsequent_investment'].clip(lower=cap_lower_bound['subsequent_investment'], 
                                                                                               upper=cap_upper_bound['subsequent_investment'])

# Step 4: Label Encoding
label_encoder = LabelEncoder()
df['fund_category_encoded'] = label_encoder.fit_transform(df['fund_category'])
df['fund_family_encoded'] = label_encoder.fit_transform(df['fund_family'])

# Step 5: Remove columns with more than 80% missing values
threshold = len(df) * 0.8
cleaned_data_final = df.dropna(thresh=threshold, axis=1)

# Step 6: Save the final cleaned file without standardization
cleaned_data_final.to_excel('final_file_with_removed_columns.xlsx', index=False)

print("Data cleaning and processing complete. The cleaned file has been saved.")
