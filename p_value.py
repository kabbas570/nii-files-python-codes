import pandas as pd
from scipy.stats import wilcoxon

# Paths to your CSV files
csv_path_method1 = r"C:\Users\Abbas Khan\Downloads\method_5.csv"
csv_path_method2 = r"C:\Users\Abbas Khan\Downloads\method_8.csv"

# Read both CSVs
df1 = pd.read_csv(csv_path_method1)
df2 = pd.read_csv(csv_path_method2)

# Ensure image names match
df1 = df1.sort_values(by='image_name').reset_index(drop=True)
df2 = df2.sort_values(by='image_name').reset_index(drop=True)

assert (df1['image_name'].values == df2['image_name'].values).all(), "Image names do not match!"

# Extract scores
dice1 = df1['Dice']
dice2 = df2['Dice']

hd1 = df1['HD']
hd2 = df2['HD']

# Perform Wilcoxon signed-rank test
w_dice = wilcoxon(dice1, dice2)
w_hd = wilcoxon(hd1, hd2)

# Print results
print(f"Wilcoxon test for Dice scores: statistic={w_dice.statistic}, p-value={w_dice.pvalue}")
print(f"Wilcoxon test for HD scores: statistic={w_hd.statistic}, p-value={w_hd.pvalue}")
