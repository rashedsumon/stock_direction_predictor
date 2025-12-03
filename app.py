import streamlit as st
from data_loader import download_data, load_data
from utils import add_technical_indicators, add_target
from model import train_model

st.set_page_config(page_title="Stock Direction Predictor", layout="wide")
st.title("📈 Stock Direction Predictor")

# Download and load data
if st.button("Download Latest Data"):
    download_data()
    st.success("Data downloaded!")

tickers_input = st.text_input("Enter tickers (comma-separated, e.g., AAPL,MSFT,GOOGL):")
tickers = [t.strip().upper() for t in tickers_input.split(",")] if tickers_input else None
df = load_data(tickers)

st.write("Sample Data", df.head())

# Feature engineering
df = add_technical_indicators(df)
df = add_target(df)
features = [col for col in df.columns if col not in ['ticker','date','target']]

# Train model
if st.button("Train RandomForest Model"):
    model = train_model(df, features, model_type="RandomForest")
    st.success("Model trained and saved!")
