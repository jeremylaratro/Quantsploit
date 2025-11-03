"""
Technical Analysis Indicators - Pandas TA Compatibility Layer
Since pandas-ta isn't available, we implement the basic indicators
"""

import pandas as pd
import numpy as np


def sma(close: pd.Series, length: int = 20) -> pd.Series:
    """Simple Moving Average"""
    return close.rolling(window=length).mean()


def ema(close: pd.Series, length: int = 20) -> pd.Series:
    """Exponential Moving Average"""
    return close.ewm(span=length, adjust=False).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD Indicator"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({
        f'MACD_{fast}_{slow}_{signal}': macd_line,
        f'MACDs_{fast}_{slow}_{signal}': signal_line,
        f'MACDh_{fast}_{slow}_{signal}': histogram
    })


def bbands(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands"""
    middle = close.rolling(window=length).mean()
    std_dev = close.rolling(window=length).std()
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)

    # Format std properly for column names
    std_str = f"{std:.1f}"

    return pd.DataFrame({
        f'BBL_{length}_{std_str}': lower,
        f'BBM_{length}_{std_str}': middle,
        f'BBU_{length}_{std_str}': upper
    })


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=length).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.DataFrame:
    """Average Directional Index"""
    # Calculate +DM and -DM
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=low.index)

    # Calculate ATR
    tr = atr(high, low, close, length)

    # Calculate smoothed +DI and -DI
    plus_di = 100 * (plus_dm.rolling(window=length).mean() / tr)
    minus_di = 100 * (minus_dm.rolling(window=length).mean() / tr)

    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_line = dx.rolling(window=length).mean()

    return pd.DataFrame({
        f'ADX_{length}': adx_line,
        f'DMP_{length}': plus_di,
        f'DMN_{length}': minus_di
    })


def roc(close: pd.Series, length: int = 12) -> pd.Series:
    """Rate of Change"""
    return ((close - close.shift(length)) / close.shift(length)) * 100


def stoch(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3) -> pd.DataFrame:
    """Stochastic Oscillator"""
    lowest_low = low.rolling(window=k).min()
    highest_high = high.rolling(window=k).max()

    k_line = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_line = k_line.rolling(window=d).mean()

    return pd.DataFrame({
        f'STOCHk_{k}_{d}_3': k_line,
        f'STOCHd_{k}_{d}_3': d_line
    })


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume"""
    obv_values = []
    obv_val = 0

    for i in range(len(close)):
        if i == 0:
            obv_val = volume.iloc[i]
        else:
            if close.iloc[i] > close.iloc[i-1]:
                obv_val += volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv_val -= volume.iloc[i]
        obv_values.append(obv_val)

    return pd.Series(obv_values, index=close.index)


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume Weighted Average Price"""
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()


# Create a ta object that mimics pandas_ta interface
class TA:
    """Mimics pandas_ta interface"""

    @staticmethod
    def sma(close, length=20):
        return sma(close, length)

    @staticmethod
    def ema(close, length=20):
        return ema(close, length)

    @staticmethod
    def rsi(close, length=14):
        return rsi(close, length)

    @staticmethod
    def macd(close, fast=12, slow=26, signal=9):
        return macd(close, fast, slow, signal)

    @staticmethod
    def bbands(close, length=20, std=2.0):
        return bbands(close, length, std)

    @staticmethod
    def atr(high, low, close, length=14):
        return atr(high, low, close, length)

    @staticmethod
    def adx(high, low, close, length=14):
        return adx(high, low, close, length)

    @staticmethod
    def roc(close, length=12):
        return roc(close, length)

    @staticmethod
    def stoch(high, low, close, k=14, d=3):
        return stoch(high, low, close, k, d)

    @staticmethod
    def obv(close, volume):
        return obv(close, volume)

    @staticmethod
    def vwap(high, low, close, volume):
        return vwap(high, low, close, volume)


ta = TA()
