import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Model on fake data --> just used to test functionality 
def train_synthetic_model():
    print("Generating synthetic 6-feature dataset...")
    np.random.seed(42)
    n_samples = 2000

    # Generate synthetic feature data 
    ear = np.random.uniform(0.15, 0.35, n_samples)
    perclos = np.random.uniform(0.0, 1.0, n_samples)
    is_closed = np.where(ear < 0.21, 1, 0)
    
    active_app = np.random.randint(0, 5, n_samples)
    keystrokes = np.random.randint(0, 200, n_samples)
    mouse_clicks = np.random.randint(0, 100, n_samples)

    # Compute synthetic target variable
    focus_score = (ear * 100) - (perclos * 30) - (is_closed * 20) + (keystrokes * 0.2) + (mouse_clicks * 0.1)
    focus_score = np.where(active_app == 2, focus_score + 20, focus_score)
    focus_score = np.clip(focus_score, 0, 100)

    # Single "point" we use has 7 variables in it 
    df = pd.DataFrame({
        'ear': ear,
        'perclos': perclos,
        'is_closed': is_closed,
        'active_app': active_app,
        'keystrokes': keystrokes,
        'mouse_clicks': mouse_clicks,
        'focus_score': focus_score
    })

    X = df.drop('focus_score', axis=1)
    y = df['focus_score']

    # Model Training
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Evaluation
    predictions = model.predict(X)
    print(f"Mean Absolute Error: {mean_absolute_error(y, predictions):.2f}")
    print(f"R^2 Score: {r2_score(y, predictions):.2f}")

    # Serialization
    model_filename = "focus_random_forest.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")

if __name__ == "__main__":
    train_synthetic_model()