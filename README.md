# Stock Direction Predictor

A machine learning pipeline that predicts the next-period stock price direction ("up" or "down") using historical price data and classic technical indicators. 

## Features
- Automatic data download from KaggleHub
- Feature engineering: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, etc.
- Multiple ML algorithms: Random Forest, XGBoost, LightGBM, Logistic Regression
- Streamlit web interface for interactive predictions
- Outperforms naïve buy-and-hold baseline in directional accuracy

## Installation
```bash
git clone <repo_url>
cd stock_direction_predictor
pip install -r requirements.txt
streamlit run app.py
