"""
model_training.py

Loads the cleaned CSV, performs multi-genre split, trains/tunes an XGBoost model,
and saves the best model to 'model/xgb_best_model.pkl'.
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

def load_preprocessed_data(csv_path: str) -> pd.DataFrame:
    """
    Loads the cleaned CSV with columns:
      Year, Duration, Votes, Log_Votes, Decade, Genre, etc.
    Returns a DataFrame ready for final transformations (genre dummies).
    """
    return pd.read_csv(csv_path, encoding='cp1252')

def create_genre_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits multi-genre strings into dummy columns via MultiLabelBinarizer.
    """
    if 'Genre' in df.columns:
        # Split Genre on comma into a list
        df['Genre_List'] = df['Genre'].apply(lambda x: [g.strip() for g in x.split(',')])
        mlb = MultiLabelBinarizer()
        genre_dummies = mlb.fit_transform(df['Genre_List'])
        genre_df = pd.DataFrame(genre_dummies, columns=mlb.classes_, index=df.index)
        df = pd.concat([df, genre_df], axis=1)
        df.drop(columns=['Genre_List'], inplace=True)
    return df

def train_and_tune_model(df: pd.DataFrame):
    """
    1. Defines features & target
    2. Splits train/test
    3. Runs XGBoost with RandomizedSearchCV
    4. Prints & returns best model
    """
    # Define potential genre columns (common in Indian movies dataset)
    genre_cols = [
        'Action','Adventure','Animation','Biography','Comedy','Crime','Documentary',
        'Drama','Family','Fantasy','History','Horror','Music','Musical','Mystery',
        'News','Romance','Sci-Fi','Sport','Thriller','Unknown','War','Western'
    ]

    # Basic feature set
    features = ['Duration', 'Log_Votes', 'Decade'] + genre_cols
    target = 'Rating'

    # Drop rows if any feature is missing
    df_model = df.dropna(subset=features).copy()
    X = df_model[features]
    y = df_model[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Parameter space for XGBoost
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5]
    }

    xgb_model = XGBRegressor(random_state=42)

    # Randomized Search
    rand_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=10,
        scoring='neg_mean_squared_error',
        cv=3,
        random_state=42,
        n_jobs=-1
    )

    rand_search.fit(X_train, y_train)
    best_model = rand_search.best_estimator_

    # Evaluate
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("[INFO] Best Params:", rand_search.best_params_)
    print(f"[INFO] Test RMSE: {rmse:.3f}")
    print(f"[INFO] Test R^2:  {r2:.3f}")

    return best_model

if __name__ == "__main__":
    # Define paths
    input_csv = "data/IMDb_Movies_Cleaned.csv"
    model_path = "model/xgb_best_model.pkl"

    # Create 'model' folder if not exists
    os.makedirs("model", exist_ok=True)

    # 1. Load preprocessed data
    df = load_preprocessed_data(input_csv)

    # 2. Create genre dummies
    df = create_genre_dummies(df)

    # 3. Train & tune XGBoost, then evaluate
    best_xgb_model = train_and_tune_model(df)

    # 4. Save the best model to disk
    joblib.dump(best_xgb_model, model_path)
    print(f"[INFO] Saved XGBoost model to {model_path}")
