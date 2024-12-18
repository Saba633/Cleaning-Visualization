# We will be detecting and removing outliers from a dataset using the IQR method.

import pandas as pd
import seaborn as sns # Importing seaborn for boxplot visualization
from matplotlib import pyplot as plt # Importing matplotlib for visualization

# Load the dataset (update the file path if necessary)
df = pd.read_csv(r"D:\Code\Python\ML\dataasets\weight-height.csv") 

# Calculate the first (Q1) and third (Q3) quartiles for the 'Height' column
Q1 = df.Height.quantile(0.25)
Q3 = df.Height.quantile(0.75)
IQR = Q3 - Q1 # Interquartile range (Q3 - Q1)

# Calculate lower and upper bounds for detecting outliers
lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR

# Identify outliers in the 'Height' column (values outside the calculated bounds)
outlier = (df.Height < lower_limit) | (df.Height > upper_limit)
print(df[outlier])  # Display the outliers
print(f"Total outliers are: {outlier.sum()}")  # Print the count of outliers

# Remove the outliers by filtering the DataFrame
df_no_outlier = df[~outlier]

# Plotting the boxplots to compare the distribution with and without outliers

# Boxplot with outliers
plt.subplot(1, 2, 1)
sns.boxplot(x=df['Height'])
plt.title("With Outliers")
plt.xlabel("Height")

# Boxplot without outliers
plt.subplot(1, 2, 2)
sns.boxplot(x=df_no_outlier['Height'])
plt.title("Without Outliers")
plt.xlabel("Height")

# Show the plots
plt.show()

