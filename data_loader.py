## **5️⃣ data_loader.py** (Download & preprocess)

import os
import pandas as pd
from datetime import datetime
import kagglehub

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = "data"
KAGGLE_DATASET = "jakewright/9000-tickers-of-stock-market-data-full-history"

# Optional pre-defined tickers file (if you want to filter specific tickers)
TICKERS_FILE = os.path.join(DATA_DIR, "tickers.csv")

# -----------------------------
# Data download
# -----------------------------
def download_data():
    """
    Download dataset from KaggleHub if not already present.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # Check if CSVs already exist
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if csv_files:
        print("CSV files already exist. Skipping download.")
        return [os.path.join(DATA_DIR, f) for f in csv_files]

    print("Downloading dataset from KaggleHub...")
    path = kagglehub.dataset_download(KAGGLE_DATASET, path=DATA_DIR)
    print("Dataset downloaded to:", path)
    return path

# -----------------------------
# Load & preprocess data
# -----------------------------
def load_data(tickers=None):
    """
    Load historical price data for the selected tickers.
    If tickers=None, load all tickers in dataset.
    """
    # Ensure data exists
    download_data()

    # Collect all CSV files
    all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}. Please check your dataset.")

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
