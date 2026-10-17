# app.py — serve the artifact as a REST API
import json
import datetime
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_VERSION = "v1"
pipe = joblib.load("model.joblib")   # load the artifact once, at startup

app = FastAPI(title="Customer Buy Predictor")

# allow the browser page (different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # ponytail: "*" for local dev; pin to the real domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

class Customer(BaseModel):           # the input schema — FastAPI validates this
    age: int
    estimated_salary: float

@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}

# ponytail: append-to-file logging; swap for a DB/queue when volume grows
@app.post("/predict")
def predict(c: Customer):
    features = [[c.age, c.estimated_salary]]   # same order as training
    proba = pipe.predict_proba(features)[0][1]
    result = {
        "will_buy": int(proba >= 0.5),
        "probability": round(float(proba), 3),
        "model_version": MODEL_VERSION,
    }
    with open("predictions.log", "a") as f:   # one JSON line per request
        f.write(json.dumps({
            "at": datetime.datetime.now().isoformat(),
            "input": c.model_dump(),
            "output": result,
        }) + "\n")
    return result
