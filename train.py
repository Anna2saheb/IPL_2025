import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import logging
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("IPL_Player_Performance")
client = MlflowClient()

def register_model(model, model_name, X_train, run_id):
    """Register a model in MLflow Model Registry"""
    try:
        # Create model version
        result = mlflow.register_model(
            f"runs:/{run_id}/{model_name}",
            model_name
        )
        logger.info(f"Registered model '{model_name}' version {result.version}")
        
        # Transition to Production stage
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Production"
        )
        logger.info(f"Transitioned {model_name} v{result.version} to Production")
    except Exception as e:
        logger.error(f"Failed to register model {model_name}: {str(e)}")
        raise

def load_data():
    """Load and prepare data"""
    df = pd.read_csv("data/model_ready_data.csv")
    scaler = joblib.load("data/scaler.pkl")
    
    # Define targets
    regression_targets = ["Runs_Scored", "Batting_Strike_Rate", "Wickets_Taken"]
    classification_targets = {
        "High_Score": (df["Runs_Scored"] >= 0.5).astype(int),
        "Wicket_Taken": (df["Wickets_Taken"] > 0).astype(int)
    }
    
    # Prepare features
    features = [
        'Batting_Average', 'Batting_Strike_Rate', 'Balls_Faced',
        'Bowling_Average', 'Economy_Rate', 'Batsman_Dominance',
        'Bowler_Dominance', 'Rolling_Avg_Runs', 'Batsman_Type_Anchor',
        'Bowler_Type_Spin'
    ]
    
    train_df = df[df["is_train"] == True].copy()
    test_df = df[df["is_train"] == False].copy()
    
    return train_df, test_df, features, regression_targets, classification_targets

def train_and_log_model(model, model_name, X_train, y_train, X_test, y_test, params=None, task_type="regression"):
    """Train model and log to MLflow"""
    with mlflow.start_run(run_name=model_name, nested=True) as run:
        # Set params and train
        if params:
            model.set_params(**params)
            mlflow.log_params(params)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = {
            "r2_score": r2_score(y_test, y_pred) if task_type == "regression" else np.nan,
            "roc_auc": roc_auc_score(y_test, y_pred) if task_type == "classification" else np.nan
        }
        mlflow.log_metrics(metrics)
        
        # Log model
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, model_name, signature=signature)
        
        # Register the model
        register_model(model, model_name, X_train, run.info.run_id)
        
        logger.info(f"Logged and registered {model_name} - {task_type} - Metrics: {metrics}")
        return model

def main():
    logger.info("Starting model training pipeline")
    
    # Load data
    train_df, test_df, features, reg_targets, clf_targets = load_data()
    X_train = train_df[features]
    X_test = test_df[features]
    
    # Dictionary to store the 5 models we want to save
    models_to_save = {}
    
    # Train and register Random Forest for Runs_Scored
    y_train = train_df["Runs_Scored"]
    y_test = test_df["Runs_Scored"]
    rf_runs = train_and_log_model(
        RandomForestRegressor(n_estimators=100, random_state=42),
        "random_forest_runs", X_train, y_train, X_test, y_test,
        params={'max_depth': 10, 'min_samples_split': 2}
    )
    models_to_save["random_forest_runs"] = rf_runs
    
    # Train and register XGBoost for Runs_Scored
    xgb_runs = train_and_log_model(
        XGBRegressor(random_state=42),
        "xgboost_runs", X_train, y_train, X_test, y_test,
        params={'learning_rate': 0.1, 'max_depth': 3}
    )
    models_to_save["xgboost_runs"] = xgb_runs
    
    # Train and register KNN for Runs_Scored
    knn_runs = train_and_log_model(
        KNeighborsRegressor(),
        "knn_runs", X_train, y_train, X_test, y_test,
        params={'n_neighbors': 5, 'weights': 'uniform'}
    )
    models_to_save["knn_runs"] = knn_runs
    
    # Train and register Logistic Regression for High_Score
    y_train_clf = clf_targets["High_Score"][train_df.index]
    y_test_clf = clf_targets["High_Score"][test_df.index]
    lr_high_score = train_and_log_model(
        LogisticRegression(max_iter=1000),
        "logistic_half_century", X_train, y_train_clf, X_test, y_test_clf,
        params={'C': 1}, task_type="classification"
    )
    models_to_save["logistic_half_century"] = lr_high_score
    
    # Train and register Random Forest Classifier for Wicket_Taken
    y_train_wicket = clf_targets["Wicket_Taken"][train_df.index]
    y_test_wicket = clf_targets["Wicket_Taken"][test_df.index]
    rf_wicket = train_and_log_model(
        RandomForestClassifier(random_state=42),
        "rf_classifier_wicket", X_train, y_train_wicket, X_test, y_test_wicket,
        params={'max_depth': 10, 'class_weight': 'balanced'}, 
        task_type="classification"
    )
    models_to_save["rf_classifier_wicket"] = rf_wicket
    
    # Save the selected models to disk (optional)
    for model_name, model in models_to_save.items():
        joblib.dump(model, f"models/{model_name}.pkl")
        logger.info(f"Saved model: {model_name}.pkl")
    
    logger.info("Model training and registration completed successfully")

if __name__ == "__main__":
    main()