import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="Stock Direction Predictor", layout="wide")
st.title("📈 Stock Direction Predictor")

# -----------------------------
# Generate sample stock data
# -----------------------------
def generate_sample_data(tickers, days=365):
    """
    Generate synthetic stock data for the last `days` days for given tickers.
    """
    all_data = []
    for ticker in tickers:
        np.random.seed(hash(ticker) % 2**32)  # deterministic per ticker
        dates = [datetime.today() - timedelta(days=i) for i in range(days)]
        dates.sort()
        prices = np.cumsum(np.random.randn(days) * 2 + 0.5) + 100  # random walk
        volumes = np.random.randint(1_000_000, 5_000_000, size=days)
        df = pd.DataFrame({
            "date": dates,
            "ticker": ticker,
            "open": prices + np.random.randn(days),
            "high": prices + np.random.rand(days)*2,
            "low": prices - np.random.rand(days)*2,
            "close": prices,
            "volume": volumes
        })
        all_data.append(df)
    return pd.concat(all_data).reset_index(drop=True)

# -----------------------------
# Technical indicators
# -----------------------------
def add_technical_indicators(df):
    """
    Add simple moving average as a technical indicator.
    """
    df = df.copy()
    df['sma_5'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(5).mean())
    df['sma_10'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(10).mean())
    df = df.dropna()
    return df

# -----------------------------
# Target variable
# -----------------------------
def add_target(df):
    """
    Create binary target: 1 if next day's close is higher, else 0.
    """
    df = df.copy()
    df['target'] = df.groupby('ticker')['close'].shift(-1) > df['close']
    df['target'] = df['target'].astype(int)
    df = df.dropna()
    return df

# -----------------------------
# ML model training
# -----------------------------
def train_model(df, features):
    X = df[features]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    joblib.dump(model, "models/random_forest_model.pkl")
    return model, acc

# -----------------------------
# User input
# -----------------------------
tickers_input = st.text_input(
    "Enter tickers (comma-separated, e.g., AAPL,MSFT,GOOGL):",
    value="AAPL,MSFT"
)
tickers = [t.strip().upper() for t in tickers_input.split(",")] if tickers_input else ["AAPL", "MSFT"]

# -----------------------------
# Generate & display data
# -----------------------------
df = generate_sample_data(tickers)
st.write("Sample Data", df.head())

# Feature engineering
df = add_technical_indicators(df)
df = add_target(df)
features = ['open','high','low','close','volume','sma_5','sma_10']

# -----------------------------
# Train model
# -----------------------------
if st.button("Train RandomForest Model"):
    with st.spinner("Training model..."):
        model, acc = train_model(df, features)
    st.success(f"Model trained! Accuracy on test set: {acc:.2%}")
