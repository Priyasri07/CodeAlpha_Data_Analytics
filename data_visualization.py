import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Data Prep: Clean the Age column slightly by filling missing values with the median age
df['Age'] = df['Age'].fillna(df['Age'].median())

# 2. Set up the overall canvas/dashboard theme
sns.set_theme(style="darkgrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12)) # Creates a 2x2 grid of plots

# --- PLOT 1: AGE DISTRIBUTION (Top-Left) ---
sns.histplot(data=df, x='Age', kde=True, color='#3498db', ax=axes[0, 0], bins=30)
axes[0, 0].set_title('Age Distribution of Passengers', fontsize=14, fontweight='semibold', pad=12)
axes[0, 0].set_xlabel('Passenger Age')
axes[0, 0].set_ylabel('Frequency')

# --- PLOT 2: SURVIVAL RATE BY CLASS & GENDER (Top-Right) ---
sns.barplot(data=df, x='Pclass', y='Survived', hue='Sex', palette='muted', ax=axes[0, 1], errorbar=None)
axes[0, 1].set_title('Survival Rate by Ticket Class & Gender', fontsize=14, fontweight='semibold', pad=12)
axes[0, 1].set_xlabel('Ticket Class (1 = 1st, 2 = 2nd, 3 = 3rd)')
axes[0, 1].set_ylabel('Survival Probability (0.0 to 1.0)')

# --- PLOT 3: FARE VS AGE SCATTER WITH SURVIVAL HUE (Bottom-Left) ---
sns.scatterplot(data=df, x='Age', y='Fare', hue='Survived', palette={0: '#e74c3c', 1: '#2ecc71'}, alpha=0.7, ax=axes[1, 0])
axes[1, 0].set_title('Ticket Fare vs. Passenger Age', fontsize=14, fontweight='semibold', pad=12)
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Fare Paid (£)')
axes[1, 0].get_legend().set_title('Status (0=Died, 1=Survived)')

# --- PLOT 4: CORRELATION HEATMAP (Bottom-Right) ---
numeric_cols = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
correlation_matrix = numeric_cols.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=axes[1, 1], cbar=True)
axes[1, 1].set_title('Variable Correlation Matrix', fontsize=14, fontweight='semibold', pad=12)

# 3. FIX THE OVERLAPPING LAYOUT
# pad=30 pushes the main title up, and hspace=0.35 adds extra vertical space between rows
fig.suptitle('Titanic Data Analysis Dashboard: A Visual Story of Survival', fontsize=20, fontweight='bold', color='#1a1a1a', y=0.96)
plt.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.08, hspace=0.35, wspace=0.25)

# Save the polished dashboard as an image file
plt.savefig('titanic_data_dashboard.png', dpi=300)
print("Polished dashboard successfully created and saved as 'titanic_data_dashboard.png'!")

# Display the interactive window
plt.show()
