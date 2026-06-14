"""
Spread Model Module
Handles spread construction, cointegration testing, and normalization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.api import VAR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SpreadConfig:
    """Configuration for spread modeling."""
    method: str = 'rolling_regression'  # 'static', 'rolling', 'ewm'
    hedge_ratio_window: int = 60  # Window for rolling hedge ratio
    lookback_period: int = 252  # Period for mean/std calculation
    update_frequency: int = 1  # How often to recalculate parameters
    min_correlation: float = 0.5  # Minimum correlation to consider pair
    min_cointegration_pvalue: float = 0.05  # Maximum p-value for cointegration


@dataclass
class SpreadState:
    """Current state of a spread model."""
    asset_a: str
    asset_b: str
    hedge_ratio: float
    spread_mean: float
    spread_std: float
    current_spread: float
    current_zscore: float
    correlation: float
    cointegration_pvalue: Optional[float]
    last_update: int
    is_valid: bool


class SpreadModel:
    """
    Models the relationship between two assets and constructs tradable spreads.
    Supports multiple methods for hedge ratio estimation.
    """

    def __init__(self, config: Optional[SpreadConfig] = None):
        self.config = config or SpreadConfig()
        self._states: Dict[Tuple[str, str], SpreadState] = {}
        self._history: Dict[Tuple[str, str], List[float]] = {}

    def compute_hedge_ratio_static(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series
    ) -> float:
        """
        Compute static hedge ratio using OLS regression.
        Spread = Price_A - β * Price_B
        """
        # Use log prices for better statistical properties
        log_a = np.log(prices_a)
        log_b = np.log(prices_b)

        # Remove NaN
        mask = ~(np.isnan(log_a) | np.isnan(log_b))
        if mask.sum() < 30:
            return np.nan

        model = LinearRegression()
        model.fit(log_b[mask].values.reshape(-1, 1), log_a[mask].values)
        return model.coef_[0]

    def compute_hedge_ratio_rolling(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        window: int = 60
    ) -> pd.Series:
        """Compute rolling hedge ratio over time."""
        log_a = np.log(prices_a)
        log_b = np.log(prices_b)

        hedge_ratios = []
        for i in range(len(prices_a)):
            if i < window:
                hedge_ratios.append(np.nan)
            else:
                slice_a = log_a.iloc[i-window:i].values
                slice_b = log_b.iloc[i-window:i].values
                mask = ~(np.isnan(slice_a) | np.isnan(slice_b))
                if mask.sum() < 10:
                    hedge_ratios.append(np.nan)
                else:
                    model = LinearRegression()
                    model.fit(slice_b[mask].reshape(-1, 1), slice_a[mask].reshape(-1, 1))
                    hedge_ratios.append(model.coef_[0][0])

        return pd.Series(hedge_ratios, index=prices_a.index)

    def compute_hedge_ratio_ewm(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        span: int = 60
    ) -> float:
        """
        Compute exponentially weighted hedge ratio.
        Gives more weight to recent observations.
        """
        log_a = np.log(prices_a)
        log_b = np.log(prices_b)

        # Use expanding window with exponential weights
        returns_a = log_a.diff()
        returns_b = log_b.diff()

        # Exponentially weighted covariance
        cov = returns_a.ewm(span=span).cov(returns_b)
        var = returns_b.ewm(span=span).var()

        hedge_ratio = cov / var
        return hedge_ratio.iloc[-1] if not np.isnan(hedge_ratio.iloc[-1]) else hedge_ratio.mean()

    def test_cointegration(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        hedge_ratio: float
    ) -> Tuple[float, float, bool]:
        """
        Test for cointegration using Engle-Granger two-step method.
        Returns: (test_statistic, p_value, is_cointegrated)
        """
        log_a = np.log(prices_a)
        log_b = np.log(prices_b)

        # Construct spread
        spread = log_a - hedge_ratio * log_b

        # ADF test on spread
        try:
            adf_result = adfuller(spread.dropna(), maxlag=10, autolag='AIC')
            test_stat = adf_result[0]
            p_value = adf_result[1]
            is_cointegrated = p_value < self.config.min_cointegration_pvalue
            return test_stat, p_value, is_cointegrated
        except Exception:
            return np.nan, 1.0, False

    def test_cointegration_johansen(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        constant: int = 0
    ) -> Tuple[float, float, bool]:
        """
        Johansen cointegration test for multiple assets.
        More robust than Engle-Granger for multiple cointegrating vectors.
        Note: Requires arch package, falls back to Engle-Granger if not available.
        """
        try:
            from arch.unitroot.cointegration import johansen_test
            log_a = np.log(prices_a.dropna())
            log_b = np.log(prices_b.dropna())

            # Align series
            common_idx = log_a.index.intersection(log_b.index)
            data = pd.DataFrame({
                'asset_a': log_a.loc[common_idx],
                'asset_b': log_b.loc[common_idx]
            })

            result = johansen_test(data, constant=constant)
            # Trace statistic for r=0 vs r<=1
            trace_stat = result.trace_stat[0]
            critical_value = result.crit_vals[0]  # 90% critical value
            is_cointegrated = trace_stat > critical_value

            return trace_stat, 0.05 if is_cointegrated else 0.5, is_cointegrated

        except ImportError:
            # Fall back to Engle-Granger
            hedge_ratio = self.compute_hedge_ratio_static(prices_a, prices_b)
            return self.test_cointegration(prices_a, prices_b, hedge_ratio)[:2] + (True,)

    def construct_spread(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        hedge_ratio: float
    ) -> pd.Series:
        """
        Construct the spread series.
        Spread = log(Price_A) - β * log(Price_B)
        Using log prices ensures the spread is returns-like and stationary.
        """
        log_a = np.log(prices_a)
        log_b = np.log(prices_b)
        return log_a - hedge_ratio * log_b

    def normalize_spread(
        self,
        spread: pd.Series,
        window: Optional[int] = None
    ) -> Tuple[pd.Series, float, float]:
        """
        Normalize spread to Z-score using rolling statistics.
        Z = (Spread - Rolling_Mean) / Rolling_Std

        This is critical for avoiding lookahead bias - at each point in time,
        we only use information available up to that point.
        """
        window = window or self.config.lookback_period

        rolling_mean = spread.rolling(window=window, min_periods=10).mean()
        rolling_std = spread.rolling(window=window, min_periods=10).std()

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan).fillna(spread.std())

        z_score = (spread - rolling_mean) / rolling_std

        # Get current parameters
        current_mean = rolling_mean.iloc[-1] if not np.isnan(rolling_mean.iloc[-1]) else spread.mean()
        current_std = rolling_std.iloc[-1] if not np.isnan(rolling_std.iloc[-1]) else spread.std()

        return z_score, current_mean, current_std

    def compute_correlation(
        self,
        returns_a: pd.Series,
        returns_b: pd.Series,
        window: Optional[int] = None
    ) -> float:
        """Compute rolling correlation between returns."""
        if window:
            corr = returns_a.rolling(window).corr(returns_b)
            return corr.iloc[-1] if not np.isnan(corr.iloc[-1]) else corr.mean()
        return returns_a.corr(returns_b)

    def update_state(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        asset_a: str,
        asset_b: str,
        current_idx: int
    ) -> SpreadState:
        """
        Update the spread model state with latest data.
        Called periodically to refresh model parameters.
        """
        # Check correlation filter
        returns_a = np.log(prices_a).diff()
        returns_b = np.log(prices_b).diff()
        correlation = self.compute_correlation(returns_a, returns_b, window=60)

        if abs(correlation) < self.config.min_correlation:
            return SpreadState(
                asset_a=asset_a,
                asset_b=asset_b,
                hedge_ratio=0,
                spread_mean=0,
                spread_std=0,
                current_spread=0,
                current_zscore=0,
                correlation=correlation,
                cointegration_pvalue=None,
                last_update=current_idx,
                is_valid=False
            )

        # Compute hedge ratio based on method
        if self.config.method == 'static':
            hedge_ratio = self.compute_hedge_ratio_static(prices_a, prices_b)
        elif self.config.method == 'rolling':
            hedge_ratios = self.compute_hedge_ratio_rolling(prices_a, prices_b, self.config.hedge_ratio_window)
            hedge_ratio = hedge_ratios.iloc[-1] if not np.isnan(hedge_ratios.iloc[-1]) else hedge_ratios.mean()
        elif self.config.method == 'ewm':
            hedge_ratio = self.compute_hedge_ratio_ewm(prices_a, prices_b)
        else:
            hedge_ratio = self.compute_hedge_ratio_static(prices_a, prices_b)

        # Test cointegration
        _, p_value, is_cointegrated = self.test_cointegration(prices_a, prices_b, hedge_ratio)

        # Construct and normalize spread
        spread = self.construct_spread(prices_a, prices_b, hedge_ratio)
        z_score, spread_mean, spread_std = self.normalize_spread(spread)

        state = SpreadState(
            asset_a=asset_a,
            asset_b=asset_b,
            hedge_ratio=hedge_ratio if not np.isnan(hedge_ratio) else 0,
            spread_mean=spread_mean if not np.isnan(spread_mean) else 0,
            spread_std=spread_std if not np.isnan(spread_std) else 1,
            current_spread=spread.iloc[-1] if len(spread) > 0 else 0,
            current_zscore=z_score.iloc[-1] if len(z_score) > 0 else 0,
            correlation=correlation,
            cointegration_pvalue=p_value,
            last_update=current_idx,
            is_valid=is_cointegrated and not np.isnan(hedge_ratio)
        )

        key = (asset_a, asset_b)
        self._states[key] = state

        return state

    def get_zscore(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        hedge_ratio: float,
        window: int = 252
    ) -> Tuple[float, float, float]:
        """
        Compute current Z-score for a pair given historical data.
        Returns: (z_score, mean, std)
        """
        spread = self.construct_spread(prices_a, prices_b, hedge_ratio)
        z_score, spread_mean, spread_std = self.normalize_spread(spread, window)

        return (
            z_score.iloc[-1] if not np.isnan(z_score.iloc[-1]) else 0,
            spread_mean,
            spread_std
        )

    def get_state(self, asset_a: str, asset_b: str) -> Optional[SpreadState]:
        """Get current state for a pair."""
        return self._states.get((asset_a, asset_b))

    def get_all_states(self) -> Dict[Tuple[str, str], SpreadState]:
        """Get all spread states."""
        return self._states.copy()


class BasketSpreadModel(SpreadModel):
    """
    Extends spread modeling to multi-asset baskets.
    Models spreads like: Asset_A - (β1*Asset_B + β2*Asset_C + ...)
    """

    def compute_basket_hedge_ratios(
        self,
        target: pd.Series,
        basket: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute hedge ratios for a basket of assets against a target.
        Uses multivariate regression.
        """
        log_target = np.log(target)
        log_basket = np.log(basket)

        # Remove NaN
        combined = pd.concat([log_target, log_basket], axis=1).dropna()
        if len(combined) < 50:
            return {}

        y = combined.iloc[:, 0].values
        X = combined.iloc[:, 1:].values

        # Use Ridge regression for stability
        model = Ridge(alpha=1.0)
        model.fit(X, y)

        return {col: coef for col, coef in zip(basket.columns, model.coef_)}

    def construct_basket_spread(
        self,
        target: pd.Series,
        basket: pd.DataFrame,
        hedge_ratios: Dict[str, float]
    ) -> pd.Series:
        """Construct spread between target and weighted basket."""
        log_target = np.log(target)
        basket_component = sum(
            hedge_ratios[col] * np.log(basket[col])
            for col in basket.columns
        )
        return log_target - basket_component
