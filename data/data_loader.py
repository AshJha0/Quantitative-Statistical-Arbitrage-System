"""
Data Loader Module
Handles market data ingestion, preprocessing, and feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json


class DataLoader:
    """
    Responsible for loading, cleaning, and preprocessing market data.
    Supports multiple data sources and provides standardized OHLCV format.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, pd.DataFrame] = {}

    def load_csv(self, symbol: str, filepath: Optional[str] = None) -> pd.DataFrame:
        """Load price data from CSV file."""
        if filepath is None:
            filepath = self.data_dir / f"{symbol}.csv"

        df = pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')
        return self._standardize(df, symbol)

    def load_from_dict(self, symbol: str, data: Dict) -> pd.DataFrame:
        """Load price data from dictionary format."""
        df = pd.DataFrame(data)
        return self._standardize(df, symbol)

    def _standardize(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Standardize dataframe to expected format."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        # Handle common column name variations
        column_mapping = {
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume',
            'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low',
            'CLOSE': 'close', 'VOLUME': 'volume',
            'price': 'close', 'Price': 'close'
        }

        df = df.rename(columns=column_mapping)

        # Ensure required columns exist
        for col in required_columns:
            if col not in df.columns:
                if col == 'volume':
                    df[col] = 0  # Default volume if not available
                else:
                    raise ValueError(f"Missing required column: {col}")

        df['symbol'] = symbol
        return df[required_columns + ['symbol']]

    def generate_synthetic_data(
        self,
        symbol: str,
        n_points: int = 10000,
        base_price: float = 100.0,
        volatility: float = 0.02,
        drift: float = 0.0001,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data for testing.
        Uses geometric Brownian motion for realistic price paths.
        """
        if seed is not None:
            np.random.seed(seed)

        # Generate returns using GBM
        returns = np.random.normal(drift, volatility, n_points)
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLCV from prices
        df = pd.DataFrame({'close': prices})
        df['open'] = df['close'].shift(1).fillna(base_price)
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.005, n_points)))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.005, n_points)))
        df['volume'] = np.random.uniform(1e5, 1e6, n_points)

        # Create datetime index
        df.index = pd.date_range(start='2024-01-01', periods=n_points, freq='1min')
        df.index.name = 'timestamp'
        df['symbol'] = symbol

        return df[['open', 'high', 'low', 'close', 'volume', 'symbol']]

    def generate_correlated_pair(
        self,
        symbol_a: str,
        symbol_b: str,
        n_points: int = 10000,
        base_price_a: float = 2000.0,  # e.g., XAUUSD
        base_price_b: float = 25.0,     # e.g., XAGUSD
        volatility: float = 0.02,
        correlation: float = 0.85,
        cointegration_beta: float = 80.0,  # Gold/Silver ratio
        seed: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate two correlated price series with controlled cointegration.
        Useful for testing pairs trading strategies.
        """
        if seed is not None:
            np.random.seed(seed)

        # Common factor
        common = np.random.normal(0, 1, n_points)

        # Idiosyncratic components
        idio_a = np.random.normal(0, 1, n_points)
        idio_b = np.random.normal(0, 1, n_points)

        # Create correlated returns
        corr_factor = np.sqrt(1 - correlation**2)
        returns_a = volatility * (correlation * common + corr_factor * idio_a)
        returns_b = volatility * (correlation * common + corr_factor * idio_b)

        # Convert to prices
        prices_a = base_price_a * np.exp(np.cumsum(returns_a))
        prices_b = base_price_b * np.exp(np.cumsum(returns_b))

        def make_ohlcv(prices, symbol):
            df = pd.DataFrame({'close': prices})
            df['open'] = df['close'].shift(1).fillna(prices[0])
            df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.003, n_points)))
            df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.003, n_points)))
            df['volume'] = np.random.uniform(1e5, 1e6, n_points)
            df.index = pd.date_range(start='2024-01-01', periods=n_points, freq='1min')
            df.index.name = 'timestamp'
            df['symbol'] = symbol
            return df[['open', 'high', 'low', 'close', 'volume', 'symbol']]

        return make_ohlcv(prices_a, symbol_a), make_ohlcv(prices_b, symbol_b)

    def resample(self, df: pd.DataFrame, freq: str = '5min') -> pd.DataFrame:
        """Resample data to different timeframes."""
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        return df.resample(freq).agg(ohlc_dict).dropna()

    def save(self, df: pd.DataFrame, symbol: str) -> Path:
        """Save dataframe to CSV."""
        filepath = self.data_dir / f"{symbol}.csv"
        df.to_csv(filepath)
        return filepath

    def load_multiple(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Load multiple symbols into a dictionary."""
        return {symbol: self.load_csv(symbol) for symbol in symbols}


class FeatureEngineer:
    """
    Computes technical indicators and features for modeling.
    All features are computed in a point-in-time manner to avoid lookahead bias.
    """

    @staticmethod
    def compute_returns(df: pd.DataFrame, periods: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """Compute log returns for multiple periods."""
        df = df.copy()
        for period in periods:
            df[f'return_{period}'] = np.log(df['close'] / df['close'].shift(period))
        return df

    @staticmethod
    def compute_volatility(df: pd.DataFrame, windows: List[int] = [10, 50, 100]) -> pd.DataFrame:
        """Compute rolling volatility measures."""
        df = df.copy()
        returns = np.log(df['close'] / df['close'].shift(1))
        for window in windows:
            df[f'volatility_{window}'] = returns.rolling(window).std() * np.sqrt(252)
        return df

    @staticmethod
    def compute_momentum(df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Compute momentum indicators."""
        df = df.copy()
        for window in windows:
            df[f'momentum_{window}'] = df['close'] - df['close'].shift(window)
            df[f'momentum_pct_{window}'] = df[f'momentum_{window}'] / df['close'].shift(window)
        return df

    @staticmethod
    def compute_ma(df: pd.DataFrame, windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """Compute moving averages and crosses."""
        df = df.copy()
        for window in windows:
            df[f'ma_{window}'] = df['close'].rolling(window).mean()

        # Golden/death cross signals
        df['ma_ratio_5_20'] = df['ma_5'] / df['ma_20']
        df['ma_ratio_10_50'] = df['ma_10'] / df['ma_50']

        return df

    @staticmethod
    def compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Compute Relative Strength Index."""
        df = df.copy()
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def compute_bollinger_bands(df: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
        """Compute Bollinger Bands."""
        df = df.copy()
        df['bb_middle'] = df['close'].rolling(window).mean()
        bb_std = df['close'].rolling(window).std()
        df['bb_upper'] = df['bb_middle'] + n_std * bb_std
        df['bb_lower'] = df['bb_middle'] - n_std * bb_std
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        return df

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features for a given dataframe."""
        df = self.compute_returns(df)
        df = self.compute_volatility(df)
        df = self.compute_momentum(df)
        df = self.compute_ma(df)
        df = self.compute_rsi(df)
        df = self.compute_bollinger_bands(df)
        return df
