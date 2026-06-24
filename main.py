from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model_trainer import predict_custom_planet, train_exoplanet_model
from pydantic import BaseModel

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
df = engineer_features(fetch_nasa_data())
feature_cols = ['pl_bmasse', 'pl_rade', 'pl_orbper', 'st_teff']
model = train_exoplanet_model(df)

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
      model, feature_cols, data.mass, data.radius, data.period, data.star_temp
    )

    #format the breakdom
    confidence = {cls: float(prob) for cls, prob in zip(model.classes_, probabilities)}

    return {
        "prediction": prediction,
        "confidence": confidence
    }