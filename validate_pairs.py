import os
import time
import pandas as pd

from brand_model_validator import BrandModelValidator

# insert your openrouter api key
OPENROUTER_API_KEY = ""

# insert your i/o paths
INPUT_PATH = ""
OUTPUT_PATH = ""

BATCH_SIZE = 20
SLEEP_BETWEEN_BATCHES = 1.0


# full validation flow
def main():
    # init key and validator
    api_key = (OPENROUTER_API_KEY or "").strip()

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in code")

    print("KEY PREFIX:", api_key[:10], "LEN:", len(api_key))

    validator = BrandModelValidator(api_key=api_key, model="openai/gpt-4o-mini")

    # load input pairs
    df = pd.read_csv(INPUT_PATH)
    pairs = df[["brand", "model"]].dropna().to_dict("records")

    already_done = set()
    results = []

    # resume if output already exists
    if os.path.exists(OUTPUT_PATH):
        done_df = pd.read_csv(OUTPUT_PATH)
        results = done_df.to_dict("records")
        already_done = set(zip(done_df["brand"], done_df["model"]))
        print(f"Resume: found {len(already_done)} completed rows in output file.")

    remaining = [p for p in pairs if (p["brand"], p["model"]) not in already_done]

    total = len(pairs)
    print(f"Total pairs: {total}")
    print(f"Remaining pairs to process: {len(remaining)}")

    # process in batches
    for start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[start:start + BATCH_SIZE]

        try:
            corrected_rows = validator.validate_batch(batch)

        # fallback to single requests on batch failure
        except Exception as e:
            print(f"Batch failed ({start}-{start + len(batch)}): {e}")
            print("Falling back to single validation for this batch...")

            corrected_rows = []
            for row in batch:
                correction = validator.validate(row["brand"], row["model"])
                corrected_rows.append({
                    "brand": str(row["brand"]).strip().lower(),
                    "model": str(row["model"]).strip().lower(),
                    "corrected_brand": correction["corrected_brand"],
                    "corrected_model": correction["corrected_model"],
                })

        results.extend(corrected_rows)

        # save progress after each batch
        output_df = pd.DataFrame(results).drop_duplicates(
            subset=["brand", "model"],
            keep="last"
        )
        output_df.to_csv(OUTPUT_PATH, index=False)

        processed_now = min(start + len(batch), len(remaining))
        print(
            f"Saved batch. Progress: {processed_now}/{len(remaining)} "
            f"| total saved: {len(output_df)}/{total}"
        )

        time.sleep(SLEEP_BETWEEN_BATCHES)

    print("Done. Saved corrections to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
