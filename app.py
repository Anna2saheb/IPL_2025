from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load models
models = {
    "runs": joblib.load("models/xgboost_runs.pkl"),  # Using just the best model
    "high_score": joblib.load("models/logistic_half_century.pkl"),
    "wicket": joblib.load("models/rf_classifier_wicket.pkl")
}

# Load player data
df = pd.read_csv("data/model_ready_data.csv")
players = sorted(df['Player_Name'].unique().tolist())

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "players": players,
        "current_year": datetime.now().year
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    batsman: str = Form(...),
    bowler: str = Form(...)
):
    # Get player stats
    batsman_stats = df[df['Player_Name'] == batsman].iloc[0]
    bowler_stats = df[df['Player_Name'] == bowler].iloc[0]
    
    # Prepare input features
    input_features = pd.DataFrame([{
        'Batting_Average': batsman_stats['Batting_Average'],
        'Batting_Strike_Rate': batsman_stats['Batting_Strike_Rate'],
        'Balls_Faced': batsman_stats['Balls_Faced'],
        'Bowling_Average': bowler_stats['Bowling_Average'],
        'Economy_Rate': bowler_stats['Economy_Rate'],
        'Batsman_Dominance': batsman_stats['Batsman_Dominance'],
        'Bowler_Dominance': bowler_stats['Bowler_Dominance'],
        'Rolling_Avg_Runs': batsman_stats['Rolling_Avg_Runs'],
        'Batsman_Type_Anchor': batsman_stats.get('Batsman_Type_Anchor', 0),
        'Bowler_Type_Spin': bowler_stats.get('Bowler_Type_Spin', 0)
    }])
    
    # Make predictions
    runs_pred = int(models["runs"].predict(input_features)[0])
    high_score_prob = int(models["high_score"].predict_proba(input_features)[0][1] * 100)
    wicket_prob = int(models["wicket"].predict_proba(input_features)[0][1] * 100)
    
    # Performance rating (simple star system)
    performance_rating = min(5, max(1, runs_pred // 20))
    
    return templates.TemplateResponse("results.html", {
        "request": request,
        "batsman": batsman,
        "bowler": bowler,
        "runs_pred": runs_pred,
        "high_score_prob": high_score_prob,
        "wicket_prob": wicket_prob,
        "performance_rating": performance_rating,
        "current_year": datetime.now().year
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)