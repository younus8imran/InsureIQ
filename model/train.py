import os


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

import mlflow
import mlflow.sklearn

import joblib

DATA_PATH = "data/insurance_10k.csv"
MODEL_DIR = "artifacts/best_model"

def load_data(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=["charges"])
    y = df["charges"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_preprocessor(X):
    numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipe = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical)
        ]
    )

def train():
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)
    preprocessor = build_preprocessor(X_train)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "RandomForest": RandomForestRegressor(random_state=42),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    }

    # Hyper-parameter grids for some models
    param_grids = {
        "Ridge": {"model__alpha": [0.1, 1, 10]},
        "Lasso": {"model__alpha": [0.01, 0.1, 1]},
        "ElasticNet": {"model__alpha": [0.01, 0.1, 1],
                       "model__l1_ratio": [0.2, 0.5, 0.8]},
        "RandomForest": {"model__n_estimators": [100, 200],
                         "model__max_depth": [None, 5, 10]},
        "XGBoost": {"model__n_estimators": [100, 200],
                    "model__max_depth": [3, 5],
                    "model__learning_rate": [0.05, 0.1]}
    }

    mlflow.set_tracking_uri("file:./artifacts/mlruns")
    mlflow.set_experiment("medical_charges")

    best_run = None
    best_rmse = float("inf")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            pipe = Pipeline(steps=[
                ("prep", preprocessor),
                ("model", model)
            ])

            if name in param_grids:
                grid = GridSearchCV(
                    pipe,
                    param_grids[name],
                    cv=5,
                    scoring="neg_root_mean_squared_error",
                    n_jobs=-1
                )
                grid.fit(X_train, y_train)
                pipe = grid.best_estimator_
                for k, v in grid.best_params_.items():
                    mlflow.log_param(k, v)
            else:
                pipe.fit(X_train, y_train)

            preds = pipe.predict(X_test)
            rmse = root_mean_squared_error(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            mlflow.sklearn.log_model(
                pipe,
                name="model"
            )

            if rmse < best_rmse:
                best_rmse = rmse
                best_run = mlflow.active_run().info.run_id
                best_model_name = name

    # Save best model locally
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_uri = f"runs:/{best_run}/model"
    best_model = mlflow.sklearn.load_model(best_uri)
    joblib.dump(best_model, f"{MODEL_DIR}/{best_model_name}.joblib")
    print("Training complete. Best model saved at:", MODEL_DIR)
    

if __name__ == "__main__":
    train()