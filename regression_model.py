import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor


# input/output paths
INPUT_PATH = ""
OUTPUT_PATH = ""


# load prepared analysis dataset
def load_data():
    df = pd.read_csv(INPUT_PATH)
    return df


# define target and feature groups
def define_features():
    target = "price"

    numeric_features = [
        "car_age",
        "hand_num",
        "horse_power",
        "engine_volume",
        "4x4",
        "valid_test",
        "magnesium_wheels",
        "distance_control",
        "economical",
        "adaptive_cruise_control",
        "cruise_control"
    ]

    categorical_features = [
        "brand",
        "model",
        "fuel_type",
        "brand_group"
    ]

    return target, numeric_features, categorical_features


# train/test split
def split_data(df, target, numeric_features, categorical_features):
    feature_columns = numeric_features + categorical_features

    X = df[feature_columns]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


# preprocessing for numeric and categorical columns
def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


# train and evaluate one model
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(name)
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²: {r2:.4f}")
    print("----------------------")

    return rmse, r2


# final pipeline for prediction
def build_xgboost_pipeline(preprocessor):
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ))
        ]
    )
    return model


# add prediction and pricing gap columns
def generate_predictions(df, model, numeric_features, categorical_features):

    feature_columns = numeric_features + categorical_features
    X = df[feature_columns]

    predicted_price = model.predict(X)

    df["predicted_price"] = predicted_price

    df["price_gap"] = df["price"] - df["predicted_price"]

    df["price_gap_percent"] = (df["price_gap"] / df["predicted_price"]) * 100

    # absolute gap for ranking
    df["abs_price_gap"] = df["price_gap"].abs()

    # deal classification
    conditions = [
        df["price_gap_percent"] <= -10,
        df["price_gap_percent"].between(-10, 10),
        df["price_gap_percent"] >= 10
    ]

    choices = [
        "Good Deal",
        "Fair Price",
        "Overpriced"
    ]

    df["deal_label"] = np.select(conditions, choices, default="Unknown")

    return df


# full training and comparison flow
def main():
    df = load_data()

    target, numeric_features, categorical_features = define_features()

    X_train, X_test, y_train, y_test = split_data(
        df,
        target,
        numeric_features,
        categorical_features
    )

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    }

    results = []

    # compare candidate models
    for name, regressor in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", regressor)
            ]
        )

        rmse, r2 = evaluate_model(
            name,
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results.append((name, rmse, r2))

    print("\nFinal Comparison")
    print("================")

    for name, rmse, r2 in results:
        print(f"{name:20} RMSE: {rmse:,.0f}   R²: {r2:.3f}")

    print("\nTraining final XGBoost model on full dataset...")

    # fit final model and export predictions
    best_model = build_xgboost_pipeline(preprocessor)
    best_model.fit(X_train, y_train)

    df_with_predictions = generate_predictions(
        df,
        best_model,
        numeric_features,
        categorical_features
    )

    df_with_predictions.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved dataset with predictions:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
