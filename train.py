import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    accuracy_score, 
    classification_report,
    roc_auc_score
)
import joblib

# ==============================================
# 1. LOAD PREPROCESSED DATA
# ==============================================
df = pd.read_csv("data/model_ready_data.csv")
scaler = joblib.load("data/scaler.pkl")

# ==============================================
# 2. DEFINE TARGET VARIABLES
# ==============================================
# (A) For Regression (Continuous Outcomes)
regression_targets = {
    "Runs_Scored": "Predict runs scored by batsman",
    "Batting_Strike_Rate": "Predict strike rate",
    "Wickets_Taken": "Predict wickets taken by bowler"
}

# (B) For Classification (Binary Outcomes)
# Define classification targets using SCALED thresholds
SCALED_THRESHOLD = 0.5  # Adjust based on your data distribution
classification_targets = {
    "High_Score": (df["Runs_Scored"] >= SCALED_THRESHOLD).astype(int),
    "Wicket_Taken": (df["Wickets_Taken"] > 0).astype(int),
}

# ==============================================
# 3. TRAIN-TEST SPLIT
# ==============================================
train_df = df[df["is_train"] == True].copy()
test_df = df[df["is_train"] == False].copy()

# Features (excluding identifiers and targets)
features = [
    'Batting_Average', 'Batting_Strike_Rate', 'Balls_Faced',
    'Bowling_Average', 'Economy_Rate', 'Batsman_Dominance',
    'Bowler_Dominance', 'Rolling_Avg_Runs', 'Batsman_Type_Anchor',
    'Bowler_Type_Spin'
]

X_train = train_df[features]
X_test = test_df[features]

# ==============================================
# 4. MODEL TRAINING FUNCTIONS
# ==============================================
def train_regression_model(X, y, model, param_grid=None):
    """Train a regression model with optional hyperparameter tuning"""
    if param_grid:
        grid = GridSearchCV(model, param_grid, cv=5, scoring='r2')
        grid.fit(X, y)
        best_model = grid.best_estimator_
        print(f"Best Params: {grid.best_params_}")
        return best_model
    else:
        model.fit(X, y)
        return model

def train_classification_model(X, y, model, param_grid=None):
    """Train a classification model with optional hyperparameter tuning"""
    if param_grid:
        grid = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
        grid.fit(X, y)
        best_model = grid.best_estimator_
        print(f"Best Params: {grid.best_params_}")
        return best_model
    else:
        model.fit(X, y)
        return model

# ==============================================
# 5. TRAIN MODELS
# ==============================================
# (A) Regression Models
print("\n=== Training Regression Models ===")
for target, desc in regression_targets.items():
    print(f"\nTraining model for: {desc}")
    y_train = train_df[target]
    y_test = test_df[target]

    # Random Forest
    rf = train_regression_model(
        X_train, y_train,
        RandomForestRegressor(n_estimators=100, random_state=42),
        param_grid={
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        }
    )
    y_pred = rf.predict(X_test)
    print(f"Random Forest R²: {r2_score(y_test, y_pred):.3f}")

    # XGBoost
    xgb = train_regression_model(
        X_train, y_train,
        XGBRegressor(random_state=42),
        param_grid={
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 6]
        }
    )
    y_pred = xgb.predict(X_test)
    print(f"XGBoost R²: {r2_score(y_test, y_pred):.3f}")

    # KNN
    knn = train_regression_model(
        X_train, y_train,
        KNeighborsRegressor(),
        param_grid={
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }
    )
    y_pred = knn.predict(X_test)
    print(f"KNN R²: {r2_score(y_test, y_pred):.3f}")

# (B) Classification Models
print("\n=== Training Classification Models ===")
for target_name, target in classification_targets.items():
    print(f"\nTraining model for: {target_name}")
    y_train = target[train_df.index]
    y_test = target[test_df.index]

    # Logistic Regression
    lr = train_classification_model(
        X_train, y_train,
        LogisticRegression(max_iter=1000),
        param_grid={'C': [0.1, 1, 10]}
    )
    y_pred = lr.predict(X_test)
    print(f"Logistic Regression AUC: {roc_auc_score(y_test, y_pred):.3f}")

    # Random Forest Classifier
    rf_clf = train_classification_model(
        X_train, y_train,
        RandomForestClassifier(random_state=42),
        param_grid={
            'max_depth': [5, 10],
            'class_weight': ['balanced']
        }
    )
    y_pred = rf_clf.predict(X_test)
    print(f"Random Forest AUC: {roc_auc_score(y_test, y_pred):.3f}")

# ==============================================
# 6. SAVE MODELS
# ==============================================
joblib.dump(rf, "models/random_forest_runs.pkl")
joblib.dump(xgb, "models/xgboost_runs.pkl")
joblib.dump(knn, "models/knn_runs.pkl")
joblib.dump(lr, "models/logistic_half_century.pkl")
joblib.dump(rf_clf, "models/rf_classifier_wicket.pkl")

print("\n=== Models saved successfully! ===")