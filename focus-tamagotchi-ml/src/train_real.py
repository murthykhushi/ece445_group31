import os
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def train_real_model():
    data_path = Path(__file__).resolve().parents[1] / "data" / "real_training_data.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}.")
        return

    df = pd.read_csv(data_path)
    features = ['ear', 'perclos', 'is_closed', 'active_app', 'keystrokes', 'mouse_clicks']
    
    # Split into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(df[features], df['focus_score'], test_size=0.2, random_state=42)

    # Train on X_train, y_train
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    model_filename = Path(__file__).resolve().parents[1] / "focus_real_model.pkl"
    joblib.dump(model, model_filename)
    print(f"Personalized model trained on {len(X_train)} samples and saved to {model_filename}\n")
    
    # Evaluate Performance inside the function where the variables exist
    predictions = model.predict(X_test)
    print(f"Model R^2 Score: {r2_score(y_test, predictions):.2f}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, predictions):.2f}\n")

    # Extract and print feature importances
    for feature, importance in zip(features, model.feature_importances_):
        print(f"{feature}: {importance * 100:.1f}%")

if __name__ == "__main__":
    train_real_model()