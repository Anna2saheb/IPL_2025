import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ==============================================
# 1. DATA LOADING AND INITIAL CLEANING
# ==============================================
def load_and_clean_data(filepath):
    """Load raw data and perform initial cleaning"""
    df = pd.read_csv(filepath)
    df.replace("No stats", np.nan, inplace=True)
    
    numeric_cols = [
        'Runs_Scored', 'Batting_Average', 'Balls_Faced', 'Wickets_Taken',
        'Economy_Rate', 'Batting_Strike_Rate', 'Bowling_Average'
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    return df

# ==============================================
# 2. FEATURE SELECTION AND ENGINEERING
# ==============================================
def engineer_features(df):
    """Create new features and select relevant columns"""
    cols_to_keep = [
        'Player_Name', 'Year', 'Matches_Batted', 'Runs_Scored', 'Batting_Average',
        'Batting_Strike_Rate', 'Balls_Faced', 'Matches_Bowled', 'Balls_Bowled',
        'Wickets_Taken', 'Bowling_Average', 'Economy_Rate'
    ]
    df = df[cols_to_keep].copy()
    
    # Player Type Classification
    df['Batsman_Type'] = np.where(
        df['Batting_Strike_Rate'] > 140, 
        'Aggressive', 
        'Anchor'
    )
    
    # Bowler Type with Non-Bowler category
    df['Bowler_Type'] = np.where(
        df['Balls_Bowled'] == 0,
        'Non-Bowler',
        np.where(df['Economy_Rate'] > 8.5, 'Pace', 'Spin')
    )
    
    # Dominance Metrics
    epsilon = 0.0001
    df['Batsman_Dominance'] = df['Batting_Strike_Rate'] / (df['Economy_Rate'] + epsilon)
    df['Bowler_Dominance'] = df['Bowling_Average'] / (df['Batting_Average'] + epsilon)
    
    # Performance Trends
    df['Rolling_Avg_Runs'] = df.groupby('Player_Name')['Runs_Scored'].expanding().mean().droplevel(0)
    
    return df

# ==============================================
# 3. DATA VALIDATION AND FINAL PROCESSING
# ==============================================
def validate_and_finalize(df):
    """Handle missing values and data quality checks"""
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Modern way to groupby and fill without warning
    cols_to_fill = df.columns.difference(['Player_Name'])
    df[cols_to_fill] = df.groupby('Player_Name')[cols_to_fill].transform(
        lambda x: x.ffill().bfill()
    )
    
    fill_values = {
        'Runs_Scored': 0,
        'Batting_Average': 0,
        'Batting_Strike_Rate': 0,
        'Balls_Faced': 0,
        'Wickets_Taken': 0,
        'Bowling_Average': 0,
        'Economy_Rate': 0,
        'Batsman_Dominance': 0,
        'Bowler_Dominance': 0,
        'Rolling_Avg_Runs': 0
    }
    return df.fillna(fill_values)

# ==============================================
# 4. MODEL PREPARATION
# ==============================================
def prepare_for_modeling(df):
    """Final transformations before modeling"""
    numeric_features = [
        'Runs_Scored', 'Batting_Average', 'Batting_Strike_Rate',
        'Balls_Faced', 'Wickets_Taken', 'Bowling_Average',
        'Economy_Rate', 'Batsman_Dominance', 'Bowler_Dominance',
        'Rolling_Avg_Runs'
    ]
    
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    df = pd.get_dummies(df, columns=['Batsman_Type', 'Bowler_Type'], drop_first=True)
    df['is_train'] = df['Year'] < 2024
    
    return df, scaler

# ==============================================
# MAIN EXECUTION
# ==============================================
if __name__ == "__main__":
    # Configure pandas to avoid future warnings
    pd.set_option('future.no_silent_downcasting', True)
    
    print("Loading and cleaning data...")
    df = load_and_clean_data("data/cricket_data_2025.csv")
    
    print("Engineering features...")
    df = engineer_features(df)
    
    print("Validating data quality...")
    df = validate_and_finalize(df)
    
    print("Preparing for modeling...")
    df, scaler = prepare_for_modeling(df)
    
    print("Saving processed data...")
    df.to_csv("data/model_ready_data.csv", index=False)
    pd.to_pickle(scaler, "data/scaler.pkl")
    
    print("\n=== Processing Complete ===")
    print(f"Final shape: {df.shape}")
    print("Sample data:")
    print(df.head(3).to_markdown(tablefmt="grid"))
    print("\nMissing values:", df.isna().sum().sum())