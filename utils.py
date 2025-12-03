import pandas as pd
import ta

def add_technical_indicators(df):
    """Add classic technical indicators to the dataframe."""
    df = df.copy()
    # Example indicators
    df['SMA_10'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(10).mean())
    df['EMA_10'] = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=10, adjust=False).mean())
    df['RSI_14'] = df.groupby('ticker')['close'].transform(lambda x: ta.momentum.RSIIndicator(x, window=14).rsi())
    df['MACD'] = df.groupby('ticker')['close'].transform(lambda x: ta.trend.MACD(x).macd())
    df['Bollinger_H'] = df.groupby('ticker')['close'].transform(lambda x: ta.volatility.BollingerBands(x).bollinger_hband())
    df['Bollinger_L'] = df.groupby('ticker')['close'].transform(lambda x: ta.volatility.BollingerBands(x).bollinger_lband())
    df['ATR_14'] = df.groupby('ticker')[['high', 'low', 'close']].transform(lambda x: ta.volatility.AverageTrueRange(x['high'], x['low'], x['close'], window=14).average_true_range())
    
    # Lagged returns and momentum
    df['return_1'] = df.groupby('ticker')['close'].pct_change()
    df['momentum_5'] = df.groupby('ticker')['close'].transform(lambda x: x.diff(5))
    
    df = df.dropna()
    return df

def add_target(df):
    """Create up/down target for next day."""
    df['target'] = df.groupby('ticker')['close'].shift(-1) > df['close']
    df['target'] = df['target'].astype(int)
    df = df.dropna()
    return df
