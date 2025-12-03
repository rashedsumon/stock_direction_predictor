## **5️⃣ data_loader.py** (Load & preprocess CSVs from repo)

import os
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = "data"  # Make sure your CSV files are committed here

# Optional pre-defined tickers file (if you want to filter specific tickers)
TICKERS_FILE = os.path.join(DATA_DIR, "tickers.csv")

# -----------------------------
# Load & preprocess data
# -----------------------------
def load_data(tickers=None):
    """
    Load historical price data for the selected tickers from CSV files in DATA_DIR.
    If tickers=None, load all tickers in the dataset.
    """
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory '{DATA_DIR}' not found. "
                                "Please add CSV files to this folder.")

    # Collect all CSV files
    all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in '{DATA_DIR}'. Please add dataset files.")

    df_list = []
    for file in all_files:
        df = pd.read_csv(file)

        # Filter by tickers if provided
        if tickers:
            df = df[df["ticker"].isin(tickers)]

        df_list.append(df)

    # Combine all files
    combined_df = pd.concat(df_list, ignore_index=True)

    # Preprocess dates
    combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
    combined_df = combined_df.dropna(subset=['date'])

    # Sort by ticker & date
    combined_df = combined_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    return combined_df
