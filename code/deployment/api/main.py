# code/deployment/api/main.py
from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel

# Load the trained model
with open('../../../models/iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

# Define the request body
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict_iris(data: IrisRequest):
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
