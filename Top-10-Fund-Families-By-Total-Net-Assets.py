import seaborn as sns
import matplotlib.pyplot as plt

# Group by fund family and calculate the total net assets for each family
family_net_assets = data.groupby('fund_family')['total_net_assets'].sum()

# Sort the fund families by total net assets in descending order and select the top 10 families
top_families_by_assets = family_net_assets.sort_values(ascending=False).head(10)

# Plot the top fund families by total net assets using seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x=top_families_by_assets.values, y=top_families_by_assets.index, palette="coolwarm")
plt.title("Top 10 Fund Families by Total Net Assets", fontsize=14)
plt.xlabel("Total Net Assets (in billions)", fontsize=12)
plt.ylabel("Fund Family", fontsize=12)
plt.tight_layout()
plt.show()
