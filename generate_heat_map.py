
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load  file
file_path = '/content/final_file_after_filtration.xlsx'
data = pd.read_excel(file_path)

# Check the data type of the interest rate columns
print(data[['interest_rate_2020']].dtypes)

# If the columns are not numeric, convert them to numeric
data['interest_rate_2020'] = pd.to_numeric(data['interest_rate_2020'], errors='coerce')

# Define the selected columns
selected_columns = ['fund_family', 'fund_category', 'total_net_assets', 'year_to_date_return', 
                    'fund_yield', 'last_dividend', 'annual_holdings_turnover', 
                    'fund_return_2019']

# Filter the dataset to include only the selected columns
selected_data = data[selected_columns]

# Convert non-numeric columns (fund_family, fund_category) into numeric using LabelEncoder
label_encoder = LabelEncoder()
selected_data['fund_family'] = label_encoder.fit_transform(selected_data['fund_family'])
selected_data['fund_category'] = label_encoder.fit_transform(selected_data['fund_category'])

# Calculate the correlation matrix for the selected columns
correlation_matrix = selected_data.corr()

# Plot the heatmap for the selected columns with the 'viridis' palette
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap for Mutual Funds (Viridis Palette)')
plt.show()


