from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Create FastAPI app
app = FastAPI()

# Mount static folder (if you have CSS/JS files)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# ------------------- Routes -------------------

# Home page
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction page (GET + POST)
@app.api_route("/predictdata", methods=["GET", "POST"], response_class=HTMLResponse)
async def predict_datapoint(
    request: Request,
    gender: str = Form(None),
    ethnicity: str = Form(None),
    parental_level_of_education: str = Form(None),
    lunch: str = Form(None),
    test_preparation_course: str = Form(None),
    reading_score: float = Form(None),
    writing_score: float = Form(None),
):
    if request.method == "GET":
        return templates.TemplateResponse("home.html", {"request": request})

    else:
        # Collect data into CustomData object
        data = CustomData(
            gender=gender,
            race_ethnicity=ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score,
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")

        return templates.TemplateResponse(
            "home.html", {"request": request, "results": results[0]}
        )

# ------------------- Run with Uvicorn -------------------
# Run using: uvicorn main:app --reload
