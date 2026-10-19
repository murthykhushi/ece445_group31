import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

print("Loading real training data...")
# Load your collected data
df = pd.read_csv("data/real_training_data.csv")

# Split features (X) and your focus score (y)
X = df[['ear', 'perclos', 'is_closed', 'active_app', 'keystrokes', 'mouse_clicks']]
y = df['focus_score']

print(f"Training personalized Random Forest Regressor on {len(df)} data points...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the real model
joblib.dump(model, "focus_real_model.pkl")
print("Success! Personalized model saved to focus_real_model.pkl")