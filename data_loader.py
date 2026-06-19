import pandas as pd
import requests

def fetch_nasa_data():
    """Queries live data from the NASA Exoplanet Archive."""
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,pl_bmasse,pl_rade,pl_orbper,st_teff+from+pscomppars&format=json"
    response = requests.get(url)
    if response.status_code == 200:
        df = pd.DataFrame(response.json())
        return df.dropna()
    return pd.DataFrame()

def engineer_features(df):
    """Applies rule-based cosmic labels to the raw dataset."""
    def label_planet(row):
        if 0.5 <= row['pl_bmasse'] <= 5.0 and 4000 <= row['st_teff'] <= 7000:
            return "Habitable Candidate"
        elif row['pl_bmasse'] > 15:
            return "Gas Giant"
        else:
            return "Rocky / Non-Habitable"

    df['planet_type'] = df.apply(label_planet, axis=1)
    return df