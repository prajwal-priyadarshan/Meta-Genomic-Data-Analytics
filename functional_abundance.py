import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- File path ---
INPUT_FILE = r"D:\Desktop\Sem_3\IBS\Meta-Genomic-Data-Analytics\functional_predictions.csv"

# --- 1. Load Data and Group ---
try:
    df_pred = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"Error: File '{INPUT_FILE}' not found. Please ensure the functional_prediction.py script ran successfully.")
    exit()

# Define the columns for grouping and plotting
GROUPING_COL = 'Predicted_Functional_Role'
ABUNDANCE_COLS = ['Bacteria_Abundance (%)', 'Fungi_Abundance (%)']

# Calculate the mean abundance for each predicted group
df_grouped = df_pred.groupby(GROUPING_COL)[ABUNDANCE_COLS].mean().reset_index()

# Melt the DataFrame for easy plotting with seaborn (long format)
df_melted = df_grouped.melt(
    id_vars=GROUPING_COL, 
    value_vars=ABUNDANCE_COLS,
    var_name='Microbial Group',
    value_name='Mean Abundance (%)'
)

# --- 2. Generate Plot ---
plt.figure(figsize=(9, 6))
sns.barplot(
    x=GROUPING_COL, 
    y='Mean Abundance (%)', 
    hue='Microbial Group', 
    data=df_melted,
    palette={'Bacteria_Abundance (%)': '#1f77b4', 'Fungi_Abundance (%)': '#ff7f0e'}
)

plt.title('Microbial Abundance vs. Predicted Carbon Degradation Potential', fontsize=14)
plt.ylabel('Mean Abundance (%)')
plt.xlabel('Predicted Functional Role')
plt.ylim(0, df_melted['Mean Abundance (%)'].max() * 1.1)
plt.legend(title='Microbial Group', loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
plt.savefig("functional_abundance_plot.png")
print("\nFunctional Abundance Plot saved as functional_abundance_plot.png")