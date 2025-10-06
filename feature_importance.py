import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --- File path ---
INPUT_FILE = "Dataset\Soil_microbe_dataset.csv"
TARGET_COLUMN_NAME = 'β_Glucosidase (µmol/g/h)'


# --- 1. Data Preparation (Same as before) ---

def convert_range(val):
    if isinstance(val, str):
        if "–" in val:
            parts = val.split("–")
            try: return (float(parts[0]) + float(parts[1])) / 2
            except: return np.nan 
        try: return float(val)
        except: return np.nan 
    return val

df = pd.read_csv(INPUT_FILE)

if 'Soil_Depth_cm' in df.columns:
    df['Soil_Depth_cm'] = df['Soil_Depth_cm'].apply(convert_range)

# Define Target (y)
median_activity = df[TARGET_COLUMN_NAME].median()
y = (df[TARGET_COLUMN_NAME] > median_activity).astype(int)

# Define Features (X) - Crucially exclude the target column
X = df.select_dtypes(include=[np.number]).drop(columns=['ID', TARGET_COLUMN_NAME], errors='ignore')
X = X.dropna(axis=1, how='all').fillna(X.mean())

# Use a small test set (X_test) for faster SHAP calculation
_, X_test, _, _ = train_test_split(X, y, test_size=0.01, random_state=42, stratify=y) 

# --- 2. Model Training ---
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y) # Train on the full dataset for robust explanation

# --- 3. SHAP Analysis ---
print("\nStarting SHAP Feature Importance calculation...")

# NOTE: X_test_df is used for feature names, X_test_array can be used for calculation
# Ensure X_test is a DataFrame for feature names
X_test_df = X_test.copy()

# Create a TreeExplainer using the trained model
explainer = shap.TreeExplainer(model)

# Calculate SHAP values (returns a list of arrays for binary classification)
shap_values = explainer.shap_values(X_test_df)

# The list contains two arrays: shap_values[0] for Low_C_Degradation and shap_values[1] for High_C_Degradation.
# We focus on shap_values[1] (High_C_Degradation)
shap_values_class_1 = shap_values[1]

# --- 4. Generate Plot (Bar Plot FIX) ---

# FIX: Calculate the mean absolute SHAP value for each feature
# This bypasses the strict shape checking and only requires the feature names.
mean_abs_shap = np.abs(shap_values_class_1).mean(0)

plt.figure(figsize=(9, 6))
plt.title('Feature Importance for High Carbon Degradation (SHAP)', fontsize=14)

# Use the calculated mean values and the column names from the DataFrame
shap.summary_plot(
    mean_abs_shap, 
    feature_names=X_test_df.columns.tolist(), # Provide feature names explicitly
    plot_type="bar", 
    show=False
)

plt.tight_layout()
plt.savefig("shap_feature_importance.png")
plt.close()

print("SHAP Feature Importance Plot saved as shap_feature_importance.png")

# --- Optional: If you also want the detailed Beeswarm plot ---
# Note: The Beeswarm plot REQUIRES the data matrix (X_test)
# The shape error likely means you need to use the full SHAP values list/array for the beeswarm,
# or that the X_test columns were modified.
# If the above bar plot fix works, try this one next:
# --- 4. Generate Plot (Beeswarm Plot) ---
# NOTE: This plot requires the 2D matrix (samples x features)
try:
    print("\nGenerating Beeswarm plot...")
    
    plt.figure(figsize=(12, 8))
    
    # CORRECT: Pass the 2D SHAP matrix and the DataFrame/data matrix (X_test_df)
    shap.summary_plot(
        shap_values[1], # Use the full 2D matrix for the 'High_C_Degradation' class
        X_test_df,      # Pass the corresponding data matrix
        show=False
    )
    plt.title('SHAP Beeswarm Plot for High Carbon Degradation', fontsize=14)
    plt.tight_layout()
    plt.savefig("shap_beeswarm_plot.png")
    plt.close()
    
    print("SHAP Beeswarm Plot saved as shap_beeswarm_plot.png")

except Exception as e:
    print(f"\nSkipping Beeswarm plot due to error: {e}")
    print("Please ensure shap_values_class_1 is a 2D matrix.")