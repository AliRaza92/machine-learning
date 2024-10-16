# Importing necessary libraries again
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Reload the file shared by the user
file_path = '/content/filtered-file.xlsx'
data = pd.read_excel(file_path)

# Group by fund category and calculate the total net assets for each category
category_net_assets = data.groupby('fund_category')['total_net_assets'].sum()

# Sort the categories by total net assets in descending order and select the top 10 categories
top_categories_by_assets = category_net_assets.sort_values(ascending=False).head(10)

# Plot the top categories by total net assets using seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x=top_categories_by_assets.values, y=top_categories_by_assets.index, palette="viridis")
plt.title("Top 10 Fund Categories by Total Net Assets", fontsize=14)
plt.xlabel("Total Net Assets (in billions)", fontsize=12)
plt.ylabel("Fund Category", fontsize=12)
plt.tight_layout()
plt.show()
