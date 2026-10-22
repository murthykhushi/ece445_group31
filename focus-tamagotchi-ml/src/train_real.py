from pathlib import Path
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

def train_real_model():
    # My output csv at this path 
    data_path = Path(__file__).resolve().parents[1] / "data" / "real_training_data.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}. Please run the data collection script first.")
        return

    df = pd.read_csv(data_path)

    # x = raw data, y = my input score
    features = ['ear', 'perclos', 'is_closed', 'active_app', 'keystrokes', 'mouse_clicks']
    X = df[features]
    y = df['focus_score']

    # running model on it 
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    model_filename = Path(__file__).resolve().parents[1] / "focus_real_model.pkl"
    joblib.dump(model, model_filename)
    print(f"Personalized model trained on {len(df)} samples and saved to {model_filename}")

if __name__ == "__main__":
    train_real_model()