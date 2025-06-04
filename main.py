import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from infer import Predictor

predictor= Predictor()

app = FastAPI()

@app.post("/predict/")
async def predict(sentence: str):
    prediction = predictor.predict(sentence)
    return JSONResponse(conten= {prediction['Translated']})


    