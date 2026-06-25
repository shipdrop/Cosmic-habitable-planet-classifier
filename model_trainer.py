from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
import os


def train_exoplanet_model(df):
    """Trains a Random Forest Classifier on the processed NASA data."""
    X = df[['pl_bmasse', 'pl_rade', 'pl_orbper', 'st_teff']]
    y = df['planet_type']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model.joblib")

    print(f"Saving model to {model_path}")
    joblib.dump(model, model_path)

def predict_custom_planet(model, features, mass, radius, period, star_temp):
    """Predicts the category of a user-defined planet configuration."""
    user_planet = pd.DataFrame([[mass, radius, period, star_temp]], columns=features)
    prediction = model.predict(user_planet)[0]
    probabilities = model.predict_proba(user_planet)[0]
    return prediction, probabilities