import pandas as pd
import requests

def fetch_nasa_data():
    """Queries live data from the NASA Exoplanet Archive."""
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,pl_bmasse,pl_rade,pl_orbper,st_teff+from+pscomppars&format=json"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            df = pd.DataFrame(response.json())
            # ensure numeric types for strick math filtering
            numeric_cols = ['pl_bmasse', 'pl_rade', 'pl_orbper', 'st_teff']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df.dropna(subset=numeric_cols)
    except Exception as e:
        print(f"Error fetching NASA data: {e}")

    return pd.DataFrame()

def engineer_features(df):
    """Applies rule-based cosmic labels to the raw dataset."""
    if df.empty:
        return df

    def label_planet(row):
        if 0.5 <= row['pl_bmasse'] <= 5.0 and 4000 <= row['st_teff'] <= 7000:
            return "Habitable Candidate"
        elif row['pl_bmasse'] > 15:
            return "Gas Giant"
        else:
            return "Rock / Non-Habitable"

    df['planet_type'] = df.apply(label_planet, axis=1)
    return df

def get_sample_planets(limit=100):
    """Returns Formatted {x, y} coords for chart.js scatter plot."""
    df = fetch_nasa_data()

    if df.empty:
        # fallback dataset if api crashes
        return [
            {"x": 365, "y": 1.0},
            {"x": 687, "y": 0.11},
            {"x": 4333, "y": 317.8},
            {"x": 88, "y": 0.055}
        ]

    # filter out invalid numbers for logarithmic programs axes
    valid_df = df[(df['pl_orbper'] > 0) & (df['pl_bmasse'] > 0)].head(limit)

    #format for Chart.js
    points = [
        {"x": float(row['pl_orbper']), "y": float(row['pl_bmasse'])}
        for _, row in valid_df.iterrows()
    ]

    return points