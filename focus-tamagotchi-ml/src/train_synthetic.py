import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("Generating synthetic 6-feature dataset...")

# Generate 2000 rows of fake data
np.random.seed(42)
n_samples = 2000

# Fake sensor data
ear = np.random.uniform(0.15, 0.35, n_samples)
perclos = np.random.uniform(0.0, 1.0, n_samples)
is_closed = np.where(ear < 0.21, 1, 0)

# Fake desktop data
active_app = np.random.randint(0, 5, n_samples) # Apps 0 to 4
keystrokes = np.random.randint(0, 200, n_samples)
mouse_clicks = np.random.randint(0, 100, n_samples)

# Calculate a logical fake focus score (0 to 100)
# High EAR (eyes open), high keystrokes, and coding (app 2) = High Focus
focus_score = (ear * 100) - (perclos * 30) - (is_closed * 20) + (keystrokes * 0.2) + (mouse_clicks * 0.1)
focus_score = np.where(active_app == 2, focus_score + 20, focus_score) # Boost for VS Code
focus_score = np.clip(focus_score, 0, 100) # Keep between 0 and 100

# Create DataFrame matching the exact order live_inference.py expects
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

print("Training Random Forest Regressor on 6 features...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Evaluate
predictions = model.predict(X)
print(f"Mean Absolute Error: {mean_absolute_error(y, predictions):.2f}")
print(f"R^2 Score: {r2_score(y, predictions):.2f}")

# Save the model using joblib (Matches live_inference.py!)
joblib.dump(model, "focus_random_forest.pkl")
print("\nSuccess! Model saved to focus_random_forest.pkl")