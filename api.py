from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import pickle
import pandas as pd

with open("log_model.pkl", "rb") as f:   
    model = pickle.load(f)

with open("sex_encoder.pkl", "rb") as f:
    sex_enc = pickle.load(f)

with open("emb_encoder.pkl", "rb") as f:
    emb_enc = pickle.load(f)

app = FastAPI(title="Titanic Survival Prediction API")

class Passenger(BaseModel):
    Pclass:   int   = Field(..., ge=1, le=3)               
    sex:      Literal['male', 'female']                    
    Age:      int   = Field(..., ge=0,  le=120)
    SibSp:    int   = Field(..., ge=0,  le=120)
    Parch:    int   = Field(..., ge=0,  le=120)            
    Fare:     float = Field(..., ge=0.0)                   
    Embarked: Literal['S', 'C', 'Q']                       

@app.post("/predict")
def predict_survival(passenger: Passenger):
    sex_enc_val = sex_enc.transform([passenger.sex])[0]
    emb_enc_val = emb_enc.transform([passenger.Embarked])[0]

    input_df = pd.DataFrame([{
        "Pclass":   passenger.Pclass,
        "Sex":      sex_enc_val,
        "Age":      passenger.Age,
        "SibSp":    passenger.SibSp,
        "Parch":    passenger.Parch,
        "Fare":     passenger.Fare,
        "Embarked": emb_enc_val
    }])

    pred  = int(model.predict(input_df)[0])
    proba = float(model.predict_proba(input_df)[0, pred])
    return {"survived": pred, "probability": proba}
