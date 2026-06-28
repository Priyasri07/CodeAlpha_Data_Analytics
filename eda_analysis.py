import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Ask Meaningful Questions Before Analysis:
# - What percentage of passengers survived?
# - Did a passenger's gender or ticket class (Pclass) impact their survival rate?
# - Are there missing data points we need to worry about?

# 2. Load the Dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("--- 3. EXPLORING DATA STRUCTURE ---")
# Display the first 5 rows
print("\nFirst 5 Rows of the Dataset:")
print(df.head())

# Look at columns and data types
print("\nData Types and Missing Values Summary:")
print(df.info())

print("\n--- 4. DETECTING DATA ISSUES & ANOMALIES ---")
# Check for exact count of missing values in each column
missing_data = df.isnull().sum()
print("\nMissing Values Per Column:")
print(missing_data[missing_data > 0])

# Summary statistics for numerical data (checks for impossible values)
print("\nStatistical Summary (Numerical Columns):")
print(df.describe())

print("\n--- 5. IDENTIFYING TRENDS & PATTERNS (HYPOTHESIS TESTING) ---")
# Hypothesis: Did women have a higher survival rate than men?
survival_by_gender = df.groupby('Sex')['Survived'].mean() * 100
print(f"\nSurvival Rate by Gender:\n{survival_by_gender}")

# --- 6. VISUALIZATION ---
# Set up a clean plotting environment
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 5))

# Plot 1: Survival Count by Gender (Bar Chart)
plt.subplot(1, 2, 1) # 1 row, 2 columns, this is the 1st plot
sns.countplot(data=df, x='Survived', hue='Sex', palette='Set2')
plt.title('Survival Count by Gender (0 = Died, 1 = Survived)')
plt.xlabel('Survival Status')
plt.ylabel('Passenger Count')

# Plot 2: Survival Rate by Ticket Class (Bar Chart)
plt.subplot(1, 2, 2) # 1 row, 2 columns, this is the 2nd plot
sns.barplot(data=df, x='Pclass', y='Survived', hue='Pclass', palette='crest', legend=False)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Ticket Class (1st, 2nd, 3rd)')
plt.ylabel('Survival Probability')

# Adjust layout and show plots
plt.tight_layout()
print("\nDisplaying statistical plots... Close the window to complete the script.")
plt.show()