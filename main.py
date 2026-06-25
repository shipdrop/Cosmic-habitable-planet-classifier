from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from model_trainer import predict_custom_planet, train_exoplanet_model
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

#Talk to API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#load data
feature_cols = ['pl_bmasse', 'pl_rade', 'pl_orbper', 'st_teff']
MODEL_FILE = "model.joblib"

if not os.path.exists(MODEL_FILE):
    print(f" {MODEL_FILE} not found. training model automatically in the cloud...")
    try:
        from data_loader import fetch_nasa_data, engineer_features
        
        print(f" Fetching NASA data...")
        raw_data = fetch_nasa_data()
        print(f" Engineering features...")
        processed_df = engineer_features(raw_data)
        print(f" Training model...")
        trained_exoplanet_model(processed_df)
        print(f" Model trained successfully")
    except Exception as e:
        print(f" Error training model: {e}")

model = joblib.load(MODEL_FILE)

#Recieve from frontend
class PlanetData(BaseModel):
    pl_bmasse: float
    pl_rade: float
    pl_orbper: float
    st_teff: float

@app.get("/")
def home():
    return {"status": "Cosmic API is online"}

@app.post("/predict")
def get_prediction(data: PlanetData):
    prediction, probabilities = predict_custom_planet(
      model, feature_cols, data.pl_bmasse, data.pl_rade, data.pl_orbper, data.st_teff
    )

    #format the breakdom
    confidence = {cls: float(prob) for cls, prob in zip(model.classes_, probabilities)}

    return JSONResponse(content={
        "prediction": prediction,
        "confidence": confidence
    })