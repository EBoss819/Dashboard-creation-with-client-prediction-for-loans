# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from class_client import Individual
import uvicorn

app = FastAPI()


with open('./final_model/model.pkl', 'rb') as f:
    model = pickle.load(f)

    
@app.post('/predict')
def scoring_endpoint(item:Individual = None):
    dico = item.dict()
    df = pd.DataFrame([dico.values()], columns= dico.keys())
    pred = str(model.predict(df)[0])
    return {"prediction" : pred}

# -


