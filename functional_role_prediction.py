# # ============================================================
# # 🧠 Functional Potential Prediction (AI-Based Screening)
# # + Exports functional_predictions.csv for Eco-Metabolomic Map
# # ============================================================

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import shap
# import matplotlib.pyplot as plt

# # ------------------------------------------------------------
# # Step 1 — Load Dataset
# # ------------------------------------------------------------
# df = pd.read_csv(r"D:\Desktop\Sem_3\IBS\Meta-Genomic-Data-Analytics\Dataset\Soil_microbe_dataset.csv")

# # Add synthetic FunctionalRole labels if missing
# if 'FunctionalRole' not in df.columns:
#     np.random.seed(42)
#     df['FunctionalRole'] = np.random.choice(
#         ['CarbonFixation', 'NitrogenCycle', 'SulfurReduction', 'Decomposer'],
#         size=len(df)
#     )

# # ------------------------------------------------------------
# # Step 2 — Clean Numeric Columns Only
# # ------------------------------------------------------------
# def convert_range(val):
#     """Convert range strings like '10–20' → mean, else float"""
#     if isinstance(val, str):
#         if "–" in val:
#             parts = val.split("–")
#             try:
#                 return (float(parts[0]) + float(parts[1])) / 2
#             except:
#                 return np.nan
#         try:
#             return float(val)
#         except:
#             return np.nan
#     return val

# # Apply cleaning only on numeric-like columns
# for col in df.columns:
#     if col != 'FunctionalRole':
#         df[col] = df[col].apply(convert_range)

# # ------------------------------------------------------------
# # Step 3 — Prepare Features (X) and Target (y)
# # ------------------------------------------------------------
# X = df.drop(columns=['FunctionalRole'])
# y = df['FunctionalRole'].astype(str)

# # Keep only numeric columns for ML
# X = X.select_dtypes(include=[np.number])

# # Fill missing values instead of dropping everything
# X = X.fillna(X.mean())

# # Sanity check
# print(f"✅ After cleaning: {X.shape[0]} samples, {X.shape[1]} features")

# # ------------------------------------------------------------
# # Step 4 — Train/Test Split
# # ------------------------------------------------------------
# if len(X) < 5:
#     raise ValueError("Dataset too small after cleaning. Check your CSV content.")

# # Use stratify only if multiple classes exist
# if y.nunique() > 1:
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )
# else:
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

# # ------------------------------------------------------------
# # Step 5 — Train Model
# # ------------------------------------------------------------
# model = RandomForestClassifier(n_estimators=200, random_state=42)
# model.fit(X_train, y_train)

# # ------------------------------------------------------------
# # Step 6 — Evaluation
# # ------------------------------------------------------------
# y_pred = model.predict(X_test)
# print("\n✅ Classification Report:")
# print(classification_report(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))

# # ------------------------------------------------------------
# # Step 7 — SHAP Explainability
# # ------------------------------------------------------------
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X_test)

# plt.title("💡 Feature Importance via SHAP (Functional Role Prediction)")
# shap.summary_plot(shap_values, X_test, plot_type="bar")
# shap.summary_plot(shap_values, X_test)

# # ------------------------------------------------------------
# # Step 8 — Export Predictions (for Eco-Metabolomic Map)
# # ------------------------------------------------------------
# full_predictions = model.predict(X)
# pred_df = pd.DataFrame({
#     "Node": X.columns if X.shape[0] == len(X.columns) else X.index.astype(str),
#     "PredictedFunction": np.random.choice(
#         ['CarbonFixation', 'NitrogenCycle', 'SulfurReduction', 'Decomposer'],
#         size=len(X)  # replace with actual labels if known
#     )
# })

# # Overwrite with model-based predictions if mapping exists
# pred_df['PredictedFunction'] = full_predictions[:len(pred_df)]

# pred_df.to_csv("functional_predictions.csv", index=False)
# print("\n✅ Saved functional predictions → functional_predictions.csv")

# ============================================================
# 🧠 Functional Potential Prediction (AI-Based Screening)
# ============================================================

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler
# #from imblearn.over_sampling import SMOTE
# import shap
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings("ignore")

# # ------------------------------------------------------------
# # Step 1 — Load Data
# # ------------------------------------------------------------
# df = pd.read_csv(r"D:\Desktop\Sem_3\IBS\Meta-Genomic-Data-Analytics\Dataset\Soil_microbe_dataset.csv")

# # Synthetic functional labels if missing
# if "FunctionalRole" not in df.columns:
#     np.random.seed(42)
#     df["FunctionalRole"] = np.random.choice(
#         ["CarbonFixation", "NitrogenCycle", "SulfurReduction", "Decomposer"], 
#         size=len(df)
#     )

# # ------------------------------------------------------------
# # Step 2 — Clean numeric columns
# # ------------------------------------------------------------
# def convert_range(val):
#     if isinstance(val, str):
#         if "–" in val:
#             try:
#                 a, b = map(float, val.split("–"))
#                 return (a + b) / 2
#             except:
#                 return np.nan
#         try:
#             return float(val)
#         except:
#             return np.nan
#     return val

# for col in df.columns:
#     if col != "FunctionalRole":
#         df[col] = df[col].apply(convert_range)

# # Keep only numeric columns
# X = df.drop(columns=["FunctionalRole"]).select_dtypes(include=[np.number])

# # ✅ Drop columns that are completely NaN
# X = X.dropna(axis=1, how="all")

# # ✅ Impute remaining NaN with column median
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(strategy="median")
# X_array = imputer.fit_transform(X)
# X_imputed = pd.DataFrame(X_array, columns=X.columns)

# # ✅ Standardize features
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)


# # 1. Separate features (X) and target (y)
# # Replace 'Target_Column' with the actual name of your target variable column
# target_column = 'Functional_Role'  # **Replace this with your column name**

# X = df.drop(columns=[target_column])
# y = df[target_column]

# # 2. Apply scaling to the features X (if necessary)
# # Assuming 'scaler' and 'StandardScaler()' are defined/imported earlier
# # If you are scaling X, you'll use X_scaled for the split
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # NOTE: If you are NOT scaling, simply use X instead of X_scaled below

# # ------------------------------------------------------------
# # Step 3 — Train/Test Split
# # ------------------------------------------------------------
# # Assuming you are using X_scaled as your feature set:
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled,  # Features (the data you split)
#     y,         # Target (the labels you split)
#     test_size=0.2, 
#     random_state=42, 
#     stratify=y # The variable to stratify the split by (must be defined!)
# )

# # # ------------------------------------------------------------
# # # Step 4 — Handle Class Imbalance with SMOTE
# # # ------------------------------------------------------------
# # sm = SMOTE(random_state=42, k_neighbors=3)
# # X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
# # print(f"✅ Balanced training data: {len(y_train_bal)} samples")

# # Step 4 — (No SMOTE) Use data as-is
# X_train_bal, y_train_bal = X_train, y_train
# print("⚠️ SMOTE skipped — training on original class distribution")


# # ------------------------------------------------------------
# # Step 5 — Hyperparameter-tuned Random Forest
# # ------------------------------------------------------------
# rf = RandomForestClassifier(random_state=42)
# params = {
#     "n_estimators": [200, 300],
#     "max_depth": [8, 12, 16],
#     "min_samples_split": [2, 4, 6],
# }
# grid = GridSearchCV(rf, params, cv=3, n_jobs=-1, scoring="accuracy")
# grid.fit(X_train_bal, y_train_bal)

# best_model = grid.best_estimator_
# print("✅ Best RF Params:", grid.best_params_)

# # ------------------------------------------------------------
# # Step 6 — Evaluation
# # ------------------------------------------------------------
# y_pred = best_model.predict(X_test)
# print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))

# # ------------------------------------------------------------
# # Step 7 — SHAP Explainability
# # ------------------------------------------------------------
# explainer = shap.TreeExplainer(best_model)
# shap_values = explainer.shap_values(X_test)

# plt.title("💡 SHAP Feature Importance (Functional Roles)")
# shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
# plt.show()

# # ------------------------------------------------------------
# # Step 8 — Export Predictions for Eco-Metabolomic Map
# # ------------------------------------------------------------
# full_preds = best_model.predict(X_scaled)
# pred_df = pd.DataFrame({
#     "Node": X.columns if len(X.columns) == len(full_preds) else np.arange(len(full_preds)),
#     "PredictedFunction": full_preds
# })
# pred_df.to_csv("functional_predictions.csv", index=False)
# print("✅ functional_predictions.csv saved successfully!")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib # Used to save the model itself

# Define file paths
INPUT_FILE = "D:\Desktop\Sem_3\IBS\Meta-Genomic-Data-Analytics\Dataset\Soil_microbe_dataset.csv"
OUTPUT_FILE = "functional_predictions.csv"
MODEL_FILE = "rf_functional_predictor.joblib"
TARGET_COLUMN_NAME = 'β_Glucosidase (µmol/g/h)'
NEW_TARGET_NAME = 'Predicted_Functional_Role'


# --- 1. Data Cleaning and Preparation Functions ---

def convert_range(val):
    """Converts range strings (e.g., '10–20') to their mean float value."""
    if isinstance(val, str):
        if "–" in val: # Handles the long dash character
            parts = val.split("–")
            try:
                return (float(parts[0]) + float(parts[1])) / 2
            except:
                return np.nan 
        try:
            return float(val)
        except:
            return np.nan 
    return val

def prepare_data(df, target_col):
    """Cleans data, creates target, and separates features (X) and target (y)."""
    
    # Apply range conversion to 'Soil_Depth_cm'
    if 'Soil_Depth_cm' in df.columns:
        df['Soil_Depth_cm'] = df['Soil_Depth_cm'].apply(convert_range)
    
    # 1A. Define Target (y): High/Low Carbon Degradation
    median_activity = df[target_col].median()
    y = (df[target_col] > median_activity).astype(int)
    y = y.replace({1: 'High_C_Degradation', 0: 'Low_C_Degradation'})
    
    # 1B. Define Features (X): Drop ID and the raw target column (to prevent data leakage)
    X = df.select_dtypes(include=[np.number]).drop(columns=['ID', target_col], errors='ignore')
    
    # 1C. Handle Missing Data
    X = X.dropna(axis=1, how='all')
    X = X.fillna(X.mean())
    
    return X, y

# --- 2. Main Execution Block ---

if __name__ == "__main__":
    
    # Load data
    df_raw = pd.read_csv(INPUT_FILE)
    
    # Prepare features (X) and target (y)
    X, y = prepare_data(df_raw.copy(), TARGET_COLUMN_NAME)
    
    # Split data for evaluation (20% test set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train the final model
    print("Starting Random Forest Model Training...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Training Complete.")
    
    # Evaluate model performance (on the test set)
    y_pred_test = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    print(f"\nModel Test Accuracy: {accuracy:.4f} (Expected ~0.98)")
    print("--- Test Set Classification Report ---")
    print(classification_report(y_test, y_pred_test))
    
    # Predict functional role for THE ENTIRE DATASET
    # We use the full feature set 'X' to predict 'y_full_pred'
    y_full_pred = model.predict(X)
    
    # Save the model (optional, but good practice)
    joblib.dump(model, MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")
    
    # --- 3. Create and Save Final Output CSV ---
    
    # Combine original features (X), the derived prediction (y_full_pred), and the original ID column
    df_output = df_raw[['ID']].copy() # Start with ID
    df_output = df_output.merge(X.reset_index(drop=True), left_index=True, right_index=True, how='left')
    
    # Add the predicted functional role column
    df_output[NEW_TARGET_NAME] = y_full_pred
    
    # Save the final file for eco-metabolic mapping
    df_output.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nSuccessfully generated predictions for {len(df_output)} samples.")
    print(f"File saved as {OUTPUT_FILE}. Use this file for your eco-metabolic map.")