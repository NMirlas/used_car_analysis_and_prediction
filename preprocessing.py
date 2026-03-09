import os
import pandas as pd
import numpy as np

# file paths
ORIGINAL_DATA_PATH = ""

PAIRS_DIR = ""
PAIRS_PATH = ""
CORRECTIONS_PATH = ""

PROCESSED_DIR = ""
CLEAN_PATH = ""
ANALYSIS_PATH = ""

CURRENT_YEAR = 2025


# load source dataset
def load_data() -> pd.DataFrame:
    return pd.read_csv(ORIGINAL_DATA_PATH)


# light text cleanup for brand/model
def light_clean_brand_model(df: pd.DataFrame) -> pd.DataFrame:
    df["brand_raw"] = df["brand"]
    df["model_raw"] = df["model"]

    df["brand"] = df["brand"].astype(str).str.strip().str.lower()
    df["model"] = df["model"].astype(str).str.strip().str.lower()

    df.loc[df["brand"].isin(["nan", "none", ""]), "brand"] = pd.NA
    df.loc[df["model"].isin(["nan", "none", ""]), "model"] = pd.NA

    return df


# export unique pairs for validation
def export_brand_model_pairs(df: pd.DataFrame) -> None:
    os.makedirs(PAIRS_DIR, exist_ok=True)

    pairs = (
        df[["brand", "model"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["brand", "model"])
    )

    pairs.to_csv(PAIRS_PATH, index=False)
    print(f"Saved {len(pairs)} brand-model pairs to:\n{PAIRS_PATH}")


# apply corrected brand/model mapping
def apply_corrections(df: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(CORRECTIONS_PATH):
        print("No corrections file found yet, skipping apply_corrections()")
        return df

    corrections = pd.read_csv(CORRECTIONS_PATH)

    corrections["brand"] = corrections["brand"].astype(str).str.strip().str.lower()
    corrections["model"] = corrections["model"].astype(str).str.strip().str.lower()
    corrections["corrected_brand"] = corrections["corrected_brand"].astype(str).str.strip().str.lower()
    corrections["corrected_model"] = corrections["corrected_model"].astype(str).str.strip().str.lower()

    df = df.merge(corrections, on=["brand", "model"], how="left")

    df["brand"] = df["corrected_brand"].fillna(df["brand"])
    df["model"] = df["corrected_model"].fillna(df["model"])

    df = df.drop(columns=["corrected_brand", "corrected_model"])

    print(f"Applied corrections from:\n{CORRECTIONS_PATH}")
    return df


# save cleaned raw-like dataset
def save_clean_dataset(df: pd.DataFrame) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Saved clean dataset to:\n{CLEAN_PATH}")


# remove temporary/helper columns
def drop_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        "brand_normalized",
        "brand_raw",
        "model_raw"
    ]

    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_columns_to_drop)

    return df


# add model-ready engineered features
def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df["car_age"] = CURRENT_YEAR - df["year"]

    df["price_per_hp"] = np.where(
        df["horse_power"].notna() & (df["horse_power"] > 0),
        df["price"] / df["horse_power"],
        np.nan
    )

    df["price_per_liter"] = np.where(
        df["engine_volume"].notna() & (df["engine_volume"] > 0),
        df["price"] / df["engine_volume"],
        np.nan
    )

    df["age_bucket"] = pd.cut(
        df["car_age"],
        bins=[-1, 3, 6, 10, 15, 100],
        labels=["0-3", "4-6", "7-10", "11-15", "15+"]
    )

    df["price_bucket"] = pd.cut(
        df["price"],
        bins=[0, 50000, 100000, 200000, 400000, 10000000],
        labels=["0-50k", "50-100k", "100-200k", "200-400k", "400k+"]
    )

    return df


# filter invalid/outlier rows
def filter_bad_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["brand"].notna()]
    df = df[df["model"].notna()]
    df = df[df["price"].notna()]
    df = df[df["year"].notna()]

    df = df[df["price"] > 1000]
    df = df[df["year"].between(1990, CURRENT_YEAR)]
    df = df[(df["horse_power"].isna()) | (df["horse_power"] > 30)]
    df = df[(df["engine_volume"].isna()) | (df["engine_volume"] > 0)]
    df = df[df["car_age"] >= 0]

    return df


# save final analysis dataset
def save_analysis_dataset(df: pd.DataFrame) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(ANALYSIS_PATH, index=False)
    print(f"Saved analysis dataset to:\n{ANALYSIS_PATH}")


# full preprocessing flow
def main():
    df = load_data()

    df = light_clean_brand_model(df)

    export_brand_model_pairs(df)

    df = apply_corrections(df)

    save_clean_dataset(df)

    df = drop_helper_columns(df)

    df = add_feature_engineering(df)

    df = filter_bad_rows(df)

    save_analysis_dataset(df)


if __name__ == "__main__":
    main()
