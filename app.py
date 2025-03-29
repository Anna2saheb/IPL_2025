from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from contextlib import asynccontextmanager
import logging
from mlflow.exceptions import MlflowException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MLflow
mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

# Model Registry Configuration
MODEL_REGISTRY = {
    "runs": {
        "name": "xgboost_runs",
        "stage": "Production"
    },
    "high_score": {
        "name": "logistic_half_century", 
        "stage": "Production"
    },
    "wicket": {
        "name": "rf_classifier_wicket",
        "stage": "Production"
    }
}

# Load player data
try:
    df = pd.read_csv("data/model_ready_data.csv")
    players = sorted(df['Player_Name'].unique().tolist())
    logger.info("Successfully loaded player data")
except Exception as e:
    logger.error(f"Failed to load player data: {str(e)}")
    raise RuntimeError("Could not load player data")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler with MLflow Model Registry integration"""
    models = {}
    
    try:
        logger.info("Loading models from MLflow Model Registry...")
        
        for model_key, config in MODEL_REGISTRY.items():
            try:
                model_uri = f"models:/{config['name']}/{config['stage']}"
                models[model_key] = mlflow.sklearn.load_model(model_uri)
                logger.info(f"Successfully loaded {config['name']} from stage {config['stage']}")
                
                # Log model version info
                model_version = client.get_latest_versions(
                    config['name'], 
                    stages=[config['stage']]
                )[0]
                logger.info(f"Model Info - Name: {config['name']}, Version: {model_version.version}, Stage: {model_version.current_stage}")
                
            except MlflowException as e:
                logger.error(f"Failed to load {config['name']}: {str(e)}")
                raise RuntimeError(f"Could not load model {config['name']} from registry")
        
        app.state.models = models
        logger.info("All models loaded successfully from MLflow Model Registry")
        yield
        
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")
        raise

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

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
    try:
        # Validate players
        if batsman not in players or bowler not in players:
            raise HTTPException(status_code=400, detail="Invalid player selection")
        
        # Get player stats
        try:
            batsman_stats = df[df['Player_Name'] == batsman].iloc[0]
            bowler_stats = df[df['Player_Name'] == bowler].iloc[0]
        except IndexError:
            raise HTTPException(status_code=404, detail="Player data not found")

        # Prepare input features
        input_features = {
            'Batting_Average': float(batsman_stats['Batting_Average']),
            'Batting_Strike_Rate': float(batsman_stats['Batting_Strike_Rate']),
            'Balls_Faced': float(batsman_stats['Balls_Faced']),
            'Bowling_Average': float(bowler_stats['Bowling_Average']),
            'Economy_Rate': float(bowler_stats['Economy_Rate']),
            'Batsman_Dominance': float(batsman_stats['Batsman_Dominance']),
            'Bowler_Dominance': float(bowler_stats['Bowler_Dominance']),
            'Rolling_Avg_Runs': float(batsman_stats['Rolling_Avg_Runs']),
            'Batsman_Type_Anchor': float(batsman_stats.get('Batsman_Type_Anchor', 0)),
            'Bowler_Type_Spin': float(bowler_stats.get('Bowler_Type_Spin', 0))
        }
        
        # Convert to DataFrame for model input
        input_df = pd.DataFrame([input_features])

        # Get predictions
        try:
            runs_pred = int(app.state.models["runs"].predict(input_df)[0])
            high_score_prob = int(app.state.models["high_score"].predict_proba(input_df)[0][1] * 100)
            wicket_prob = int(app.state.models["wicket"].predict_proba(input_df)[0][1] * 100)
            
            # Ensure reasonable values
            runs_pred = max(0, runs_pred)
            high_score_prob = max(0, min(100, high_score_prob))
            wicket_prob = max(0, min(100, wicket_prob))
            
            performance_rating = min(5, max(1, runs_pred // 20))
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Model prediction error")

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

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)