# train_model.py
import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def load_and_aggregate_data(data_dir="data"):
    search_path = os.path.join(data_dir, "*.csv")
    csv_files = glob.glob(search_path)
    
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {data_dir}. Ensure data_logger.py has collected datasets.")
    
    df_list = [pd.read_csv(f) for f in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def train_focus_model(csv_filepath_or_dir="data", model_output_path="focus_random_forest.pkl"):
    print("Loading and preprocessing logged datasets...")
    if os.path.isdir(csv_filepath_or_dir):
        df = load_and_aggregate_data(csv_filepath_or_dir)
    else:
        df = pd.read_csv(csv_filepath_or_dir)
        
    # FIX: Convert categorical text columns (like active_app) into numeric codes
    if 'active_app' in df.columns:
        df['active_app'], _ = pd.factorize(df['active_app'])
        
    feature_cols = [c for c in df.columns if c not in ['timestamp', 'focus_score', 'session_id']]
    target_col = 'focus_score'
    
    df = df.dropna(subset=feature_cols + [target_col])
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Features used for training: {feature_cols}")
    print(f"Total training samples: {len(X)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("--- Model Evaluation Results ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R^2) Score: {r2:.4f}")
    
    importances = model.feature_importances_
    for col, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        print(f"  {col}: {imp:.4f}")
        
    joblib.dump(model, model_output_path)
    print(f"Model successfully serialized and saved to: {model_output_path}")

if __name__ == "__main__":
    train_focus_model()