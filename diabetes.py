from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
import numpy as np

app = FastAPI()

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target


model = LogisticRegression()
model.fit(X, y)

class DiabetesData(BaseModel):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    pedigree: float
    age: float

@app.post("/predict")
def predict_diabetes(data: DiabetesData):
    input_data = np.array([[
        data.pregnancies,
        data.glucose,
        data.blood_pressure,
        data.skin_thickness,
        data.insulin,
        data.bmi,
        data.pedigree,
        data.age
    ]])
    prediction = model.predict(input_data)
    return {"diabetes": int(prediction[0])}
