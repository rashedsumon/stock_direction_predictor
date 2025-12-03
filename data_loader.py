

## **5️⃣ data_loader.py** (Download & preprocess)


import os
import pandas as pd
import kagglehub
from datetime import datetime

DATA_DIR = "data"
TICKERS_FILE = os.path.join(DATA_DIR, "tickers.csv")  # Optional pre-defined tickers

def download_data():
    """Download dataset from KaggleHub."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    path = kagglehub.dataset_download("jakewright/9000-tickers-of-stock-market-data-full-history", path=DATA_DIR)
    print("Dataset downloaded to:", path)
    return path

def load_data(tickers=None):
    """
    Load historical price data for the selected tickers.
    If tickers=None, load all tickers in dataset.
    """
    all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    df_list = []
    for file in all_files:
        df = pd.read_csv(file)
        if tickers:
            df = df[df["ticker"].isin(tickers)]
        df_list.append(df)
    combined_df = pd.concat(df_list)
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df = combined_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return combined_df
