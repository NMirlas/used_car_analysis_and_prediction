import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor


# input/output paths
INPUT_PATH = ""
OUTPUT_PATH = ""


# load source data
def load_data():
    return pd.read_csv(INPUT_PATH)


df = load_data()

# features used by the model
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

target = "price"

# base feature matrix
feature_columns = numeric_features + categorical_features

X = df[feature_columns]
y = df[target]


# preprocessing pipeline
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

# train model on current market data
print("Training model...")
model.fit(X, y)

print("Generating future age simulations...")

simulation_rows = []

MAX_YEARS = 10

# generate age scenarios per car
for _, row in df.iterrows():

    current_age = row["car_age"]

    for i in range(MAX_YEARS + 1):

        simulated_age = current_age + i

        new_row = row.copy()
        new_row["car_age"] = simulated_age
        new_row["simulated_age"] = simulated_age

        simulation_rows.append(new_row)


simulation_df = pd.DataFrame(simulation_rows)

print("Simulation rows:", simulation_df.shape)


X_sim = simulation_df[feature_columns]

# predict simulated future prices
simulation_df["future_predicted_price"] = model.predict(X_sim)


# save simulation output
simulation_df.to_csv(OUTPUT_PATH, index=False)

print("Saved simulation dataset:")
print(OUTPUT_PATH)
