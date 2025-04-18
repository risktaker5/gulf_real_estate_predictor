import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json

# Constants
DATASET_PATH = "data/real_estate_dataset.csv"
MODEL_DIR = "model"
RANDOM_STATE = 42

def create_model_directory():
    os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_prepare_data():
    df = pd.read_csv(DATASET_PATH)
    X = df[['Area', 'Bedrooms', 'Bathrooms', 'Country', 'City']]
    y = df['Price']
    return X, y

def get_preprocessor():
    numerical_features = ['Area', 'Bedrooms', 'Bathrooms']
    categorical_features = ['Country', 'City']

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = get_preprocessor()

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }

    return model, metrics

def save_model_artifacts(model, metrics):
    joblib.dump(model, os.path.join(MODEL_DIR, 'real_estate_model.pkl'))

    with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    with open(os.path.join(MODEL_DIR, 'feature_names.json'), 'w') as f:
        json.dump(list(feature_names), f, indent=4)

def main():
    print("Starting model training...")
    create_model_directory()
    print("Loading and preparing data...")
    X, y = load_and_prepare_data()
    print("Training model...")
    model, metrics = train_model(X, y)
    print("Saving model artifacts...")
    save_model_artifacts(model, metrics)
    print("\nTraining completed!")
    print("Model saved to:", os.path.join(MODEL_DIR, 'real_estate_model.pkl'))
    print("\nModel Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
