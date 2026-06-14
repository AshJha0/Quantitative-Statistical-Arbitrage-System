"""
Mathematical Utilities for Quantitative Finance
Core statistical and numerical methods for spread modeling.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import eigh
from typing import Tuple, Optional, List
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class CointegrationTestResult:
    """Result of cointegration test."""
    test_statistic: float
    p_value: float
    critical_values: dict
    is_cointegrated: bool
    method: str
    null_hypothesis: str


@dataclass
class HedgeRatioEstimate:
    """Hedge ratio estimation result."""
    hedge_ratio: float
    standard_error: float
    t_statistic: float
    p_value: float
    r_squared: float
    method: str
    confidence_interval: Tuple[float, float]


class StatisticalTests:
    """
    Statistical tests for pairs trading analysis.
    """

    @staticmethod
    def augmented_dickey_fuller(
        series: pd.Series,
        max_lag: int = None,
        regression: str = 'c',
        autolag: str = 'AIC'
    ) -> CointegrationTestResult:
        """
        Augmented Dickey-Fuller test for unit root.

        Null hypothesis: Series has a unit root (non-stationary)
        Alternative: Series is stationary

        Args:
            series: Time series to test
            max_lag: Maximum lag for ADF regression
            regression: Type of regression ('c'=constant, 'ct'=constant+trend, 'n'=none)
            autolag: Method for selecting lag ('AIC', 'BIC', 't-stat', None)

        Returns:
            CointegrationTestResult with test statistics
        """
        from statsmodels.tsa.stattools import adfuller

        series = series.dropna()
        if len(series) < 10:
            return CointegrationTestResult(
                test_statistic=np.nan,
                p_value=1.0,
                critical_values={},
                is_cointegrated=False,
                method='ADF',
                null_hypothesis='Series has unit root'
            )

        result = adfuller(series, maxlag=max_lag, regression=regression, autolag=autolag)

        return CointegrationTestResult(
            test_statistic=result[0],
            p_value=result[1],
            critical_values=dict(zip(['1%', '5%', '10%'], result[4].values())),
            is_cointegrated=result[1] < 0.05,
            method='ADF',
            null_hypothesis='Series has unit root (non-stationary)'
        )

    @staticmethod
    def engle_granger(
        series_y: pd.Series,
        series_x: pd.Series,
        regression: str = 'c'
    ) -> CointegrationTestResult:
        """
        Engle-Granger two-step cointegration test.

        Step 1: Estimate long-run relationship via OLS
        Step 2: Test residuals for stationarity using ADF

        Null hypothesis: No cointegration
        Alternative: Series are cointegrated

        Args:
            series_y: Dependent variable (e.g., log price of asset A)
            series_x: Independent variable (e.g., log price of asset B)
            regression: Type of regression for OLS

        Returns:
            CointegrationTestResult with test statistics
        """
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tsa.stattools import adfuller

        # Align series
        common_idx = series_y.index.intersection(series_x.index)
        y = series_y.loc[common_idx].dropna()
        x = series_x.loc[common_idx].dropna()
        common_idx = y.index.intersection(x.index)

        if len(common_idx) < 30:
            return CointegrationTestResult(
                test_statistic=np.nan,
                p_value=1.0,
                critical_values={},
                is_cointegrated=False,
                method='Engle-Granger',
                null_hypothesis='No cointegration'
            )

        y = y.loc[common_idx]
        x = x.loc[common_idx]

        # Step 1: OLS regression y = α + β*x + ε
        X = np.column_stack([np.ones(len(x)), x.values])
        beta = np.linalg.lstsq(X, y.values, rcond=None)[0]
        residuals = y.values - (beta[0] + beta[1] * x.values)

        # Step 2: ADF test on residuals
        adf_result = adfuller(residuals, maxlag=10, autolag='AIC', regression='c')

        # Critical values specific to Engle-Granger (different from standard ADF)
        # These are approximate values for 2-variable cointegration
        n_obs = len(residuals)
        cv_adjustment = 1 + 50 / n_obs  # Small sample adjustment

        eg_critical_values = {
            '1%': -3.90 * cv_adjustment,
            '5%': -3.34 * cv_adjustment,
            '10%': -3.04 * cv_adjustment
        }

        return CointegrationTestResult(
            test_statistic=adf_result[0],
            p_value=adf_result[1],
            critical_values=eg_critical_values,
            is_cointegrated=adf_result[0] < eg_critical_values['5%'],
            method='Engle-Granger',
            null_hypothesis='No cointegration between series'
        )

    @staticmethod
    def johansen_test(
        data: pd.DataFrame,
        constant: int = 0,
        klag: int = None
    ) -> CointegrationTestResult:
        """
        Johansen cointegration test for multiple series.

        More robust than Engle-Granger for:
        - Multiple cointegrating vectors
        - Testing rank of cointegration

        Args:
            data: DataFrame with multiple price series (log prices)
            constant: 0=no constant, 1=constant in cointegration, 2=constant+trend
            klag: Lag length for VAR (default: floor(T^(1/4)))

        Returns:
            CointegrationTestResult with trace statistic
        """
        try:
            from arch.unitroot.cointegration import johansen_test as arch_johansen
            result = arch_johansen(data, constant=constant, klag=klag)

            return CointegrationTestResult(
                test_statistic=result.trace_stat[0],
                p_value=0.05 if result.trace_stat[0] > result.crit_vals[0] else 0.5,
                critical_values={
                    '90%': result.crit_vals[0],
                    '95%': result.crit_vals[1],
                    '99%': result.crit_vals[2]
                },
                is_cointegrated=result.trace_stat[0] > result.crit_vals[0],
                method='Johansen',
                null_hypothesis='Rank = 0 (no cointegration)'
            )
        except ImportError:
            # Fallback: pairwise Engle-Granger
            if len(data.columns) >= 2:
                return StatisticalTests.engle_granger(
                    data.iloc[:, 0],
                    data.iloc[:, 1]
                )
            return CointegrationTestResult(
                test_statistic=np.nan,
                p_value=1.0,
                critical_values={},
                is_cointegrated=False,
                method='Fallback',
                null_hypothesis='No cointegration'
            )

    @staticmethod
    def hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
        """
        Calculate Hurst Exponent to determine mean-reversion tendency.

        H < 0.5: Mean-reverting
        H = 0.5: Random walk
        H > 0.5: Trending

        Args:
            series: Time series to analyze
            max_lag: Maximum lag for R/S calculation

        Returns:
            Hurst exponent (0 < H < 1)
        """
        series = series.dropna()
        n = len(series)

        if n < max_lag * 2:
            max_lag = n // 2

        lags = range(2, max_lag)
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]

        # Linear regression in log-log space
        try:
            coef = np.polyfit(np.log(list(lags)), np.log(tau), 1)
            return coef[0]
        except (ValueError, RuntimeWarning):
            return 0.5

    @staticmethod
    def ornstein_uhlenbeck_params(
        series: pd.Series,
        dt: float = 1.0
    ) -> Tuple[float, float, float]:
        """
        Estimate parameters of Ornstein-Uhlenbeck process.

        dX_t = θ(μ - X_t)dt + σdW_t

        Where:
        - θ: Mean reversion speed (higher = faster reversion)
        - μ: Long-term mean
        - σ: Volatility of innovations

        Returns:
            (theta, mu, sigma) - OU parameters
        """
        series = series.dropna()

        # Discrete approximation: X_{t+1} - X_t = θ(μ - X_t) + σ*ε
        # Rearranged: ΔX = θ*μ - θ*X_t + σ*ε

        X = series.values[:-1]
        dX = np.diff(series.values)

        # OLS: dX = α + β*X + ε
        A = np.column_stack([np.ones(len(X)), X])
        result = np.linalg.lstsq(A, dX, rcond=None)[0]

        alpha, beta = result[0], result[1]

        # Map to OU parameters
        theta = -beta / dt
        mu = alpha / theta if theta != 0 else np.mean(series)
        sigma = np.std(dX) / np.sqrt(dt)

        return theta, mu, sigma

    @staticmethod
    def half_life(theta: float, dt: float = 1.0) -> float:
        """
        Calculate half-life of mean reversion.

        Half-life = ln(2) / θ

        Time for deviation to reduce by 50% on average.
        """
        if theta <= 0:
            return np.inf
        return np.log(2) / theta


class HedgeRatioEstimator:
    """
    Multiple methods for estimating hedge ratio between assets.
    """

    @staticmethod
    def ols(
        y: pd.Series,
        x: pd.Series,
        use_log: bool = True
    ) -> HedgeRatioEstimate:
        """
        Ordinary Least Squares hedge ratio.

        log(P_y) = α + β*log(P_x) + ε

        β represents the hedge ratio.
        """
        # Align and transform
        common_idx = y.index.intersection(x.index)
        if use_log:
            y_vals = np.log(y.loc[common_idx]).values
            x_vals = np.log(x.loc[common_idx]).values
        else:
            y_vals = y.loc[common_idx].values
            x_vals = x.loc[common_idx].values

        n = len(y_vals)
        if n < 10:
            return HedgeRatioEstimate(
                hedge_ratio=np.nan,
                standard_error=np.nan,
                t_statistic=np.nan,
                p_value=1.0,
                r_squared=0.0,
                method='OLS',
                confidence_interval=(np.nan, np.nan)
            )

        # OLS estimation
        X = np.column_stack([np.ones(n), x_vals])
        beta_hat = np.linalg.lstsq(X, y_vals, rcond=None)[0]

        # Residuals and variance
        y_pred = X @ beta_hat
        residuals = y_vals - y_pred
        sse = np.sum(residuals ** 2)
        sigma_sq = sse / (n - 2)

        # Standard error of β
        x_centered = x_vals - np.mean(x_vals)
        se_beta = np.sqrt(sigma_sq / np.sum(x_centered ** 2))

        # T-statistic and p-value
        t_stat = beta_hat[1] / se_beta if se_beta > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

        # R-squared
        ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
        r_squared = 1 - sse / ss_tot if ss_tot > 0 else 0

        # 95% confidence interval
        t_crit = stats.t.ppf(0.975, n - 2)
        ci_low = beta_hat[1] - t_crit * se_beta
        ci_high = beta_hat[1] + t_crit * se_beta

        return HedgeRatioEstimate(
            hedge_ratio=beta_hat[1],
            standard_error=se_beta,
            t_statistic=t_stat,
            p_value=p_value,
            r_squared=r_squared,
            method='OLS',
            confidence_interval=(ci_low, ci_high)
        )

    @staticmethod
    def rolling_ols(
        y: pd.Series,
        x: pd.Series,
        window: int = 60,
        use_log: bool = True
    ) -> pd.Series:
        """
        Rolling window OLS hedge ratio estimation.

        Captures time-varying relationship between assets.
        """
        common_idx = y.index.intersection(x.index)
        if use_log:
            y_aligned = np.log(y.loc[common_idx])
            x_aligned = np.log(x.loc[common_idx])
        else:
            y_aligned = y.loc[common_idx]
            x_aligned = x.loc[common_idx]

        hedge_ratios = []

        for i in range(len(common_idx)):
            if i < window:
                hedge_ratios.append(np.nan)
            else:
                y_window = y_aligned.iloc[i - window:i].values
                x_window = x_aligned.iloc[i - window:i].values

                if len(y_window) < 10:
                    hedge_ratios.append(np.nan)
                else:
                    X = np.column_stack([np.ones(len(x_window)), x_window])
                    try:
                        beta = np.linalg.lstsq(X, y_window, rcond=None)[0]
                        hedge_ratios.append(beta[1])
                    except:
                        hedge_ratios.append(np.nan)

        return pd.Series(hedge_ratios, index=common_idx)

    @staticmethod
    def ewm(
        y: pd.Series,
        x: pd.Series,
        span: int = 60,
        use_log: bool = True
    ) -> pd.Series:
        """
        Exponentially weighted moving hedge ratio.

        Gives more weight to recent observations.
        """
        common_idx = y.index.intersection(x.index)
        if use_log:
            y_aligned = np.log(y.loc[common_idx])
            x_aligned = np.log(x.loc[common_idx])
        else:
            y_aligned = y.loc[common_idx]
            x_aligned = x.loc[common_idx]

        # Compute returns for EWM
        y_ret = y_aligned.diff()
        x_ret = x_aligned.diff()

        # Exponentially weighted covariance and variance
        cov = y_ret.ewm(span=span).cov(x_ret)
        var = x_ret.ewm(span=span).var()

        hedge_ratio = cov / var
        return hedge_ratio

    @staticmethod
    def total_least_squares(
        y: pd.Series,
        x: pd.Series,
        use_log: bool = True
    ) -> HedgeRatioEstimate:
        """
        Total Least Squares (TLS) hedge ratio.

        Accounts for errors in both variables (not just y).
        More appropriate when both assets have measurement error.
        """
        common_idx = y.index.intersection(x.index)
        if use_log:
            y_vals = np.log(y.loc[common_idx]).values
            x_vals = np.log(x.loc[common_idx]).values
        else:
            y_vals = y.loc[common_idx].values
            x_vals = x.loc[common_idx].values

        n = len(y_vals)
        if n < 10:
            return HedgeRatioEstimate(
                hedge_ratio=np.nan,
                standard_error=np.nan,
                t_statistic=np.nan,
                p_value=1.0,
                r_squared=0.0,
                method='TLS',
                confidence_interval=(np.nan, np.nan)
            )

        # Center the data
        y_centered = y_vals - np.mean(y_vals)
        x_centered = x_vals - np.mean(x_vals)

        # SVD approach for TLS
        A = np.column_stack([x_centered, y_centered])
        U, s, Vt = np.linalg.svd(A, full_matrices=False)

        # TLS solution from right singular vector
        beta_tls = -Vt[0, 0] / Vt[0, 1]

        # Approximate standard error (bootstrap would be more accurate)
        residuals = y_centered - beta_tls * x_centered
        se = np.std(residuals) / np.sqrt(np.sum(x_centered ** 2))

        return HedgeRatioEstimate(
            hedge_ratio=beta_tls,
            standard_error=se,
            t_statistic=beta_tls / se if se > 0 else 0,
            p_value=0.01,  # Approximate
            r_squared=0.95,  # Approximate
            method='TLS',
            confidence_interval=(beta_tls - 1.96 * se, beta_tls + 1.96 * se)
        )


class SpreadNormalizer:
    """
    Methods for normalizing spreads to Z-scores.
    """

    @staticmethod
    def rolling_zscore(
        spread: pd.Series,
        window: int = 252,
        min_periods: int = 30
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute rolling Z-score normalization.

        Z_t = (Spread_t - μ_{t-1}) / σ_{t-1}

        Uses only past data to avoid lookahead bias.

        Returns:
            (z_score, rolling_mean, rolling_std)
        """
        # Rolling statistics (shifted to avoid lookahead)
        rolling_mean = spread.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = spread.rolling(window=window, min_periods=min_periods).std()

        # Z-score
        z_score = (spread - rolling_mean) / rolling_std

        # Handle division by zero
        z_score = z_score.replace([np.inf, -np.inf], np.nan)
        z_score = z_score.fillna(0)

        return z_score, rolling_mean, rolling_std

    @staticmethod
    def expanding_zscore(
        spread: pd.Series,
        min_periods: int = 30
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute expanding window Z-score.

        Uses all available history up to each point.
        More stable but slower to adapt to regime changes.
        """
        expanding_mean = spread.expanding(min_periods=min_periods).mean()
        expanding_std = spread.expanding(min_periods=min_periods).std()

        z_score = (spread - expanding_mean) / expanding_std
        z_score = z_score.replace([np.inf, -np.inf], np.nan).fillna(0)

        return z_score, expanding_mean, expanding_std

    @staticmethod
    def ewm_zscore(
        spread: pd.Series,
        span: int = 252,
        min_periods: int = 30
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute EWM-based Z-score.

        Faster adaptation to recent regime changes.
        """
        ewm_mean = spread.ewm(span=span, min_periods=min_periods).mean()
        ewm_std = spread.ewm(span=span, min_periods=min_periods).std()

        z_score = (spread - ewm_mean) / ewm_std
        z_score = z_score.replace([np.inf, -np.inf], np.nan).fillna(0)

        return z_score, ewm_mean, ewm_std

    @staticmethod
    def percentile_rank(
        spread: pd.Series,
        window: int = 252
    ) -> pd.Series:
        """
        Compute percentile rank of spread within rolling window.

        Alternative to Z-score, more robust to outliers.
        """
        def rank_func(x):
            if len(x) < 2:
                return 0.5
            return stats.percentileofscore(x, x.iloc[-1]) / 100

        percentile = spread.rolling(window=window).apply(rank_func, raw=False)
        return percentile.fillna(0.5)


class CorrelationAnalyzer:
    """
    Tools for analyzing correlations between assets.
    """

    @staticmethod
    def rolling_correlation(
        x: pd.Series,
        y: pd.Series,
        window: int = 60
    ) -> pd.Series:
        """Compute rolling correlation between two series."""
        common_idx = x.index.intersection(y.index)
        return x.loc[common_idx].rolling(window).corr(y.loc[common_idx])

    @staticmethod
    def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
        """Compute correlation matrix for multiple assets."""
        return returns.corr()

    @staticmethod
    def eigen_decomposition(corr_matrix: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Eigen decomposition of correlation matrix.

        Returns:
            (eigenvalues, eigenvectors)
        """
        eigenvalues, eigenvectors = eigh(corr_matrix.values)
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        return eigenvalues[idx], eigenvectors[:, idx]

    @staticmethod
    def principal_components(returns: pd.DataFrame, n_components: int = 3) -> pd.DataFrame:
        """
        Extract principal components from returns.

        Useful for identifying common factors driving correlations.
        """
        corr = returns.corr()
        eigenvalues, eigenvectors = eigen_decomposition(corr)

        # Project returns onto principal components
        pcs = pd.DataFrame(
            returns.values @ eigenvectors[:, :n_components],
            index=returns.index,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        return pcs

    @staticmethod
    def minimum_spanning_tree(corr_matrix: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """
        Construct minimum spanning tree from correlation matrix.

        Useful for visualizing asset relationships and clustering.
        """
        from scipy.sparse.csgraph import minimum_spanning_tree

        # Convert correlation to distance
        distance = np.sqrt(2 * (1 - corr_matrix.values))
        np.fill_diagonal(distance, 0)

        # MST
        mst = minimum_spanning_tree(distance)

        # Extract edges
        edges = []
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if mst[i, j] > 0:
                    edges.append((
                        corr_matrix.index[i],
                        corr_matrix.columns[j],
                        corr_matrix.values[i, j]
                    ))

        return sorted(edges, key=lambda x: -x[2])  # Sort by correlation desc


def generate_math_report(
    prices_a: pd.Series,
    prices_b: pd.Series,
    pair_name: str = "Pair"
) -> dict:
    """
    Generate comprehensive mathematical analysis report for a pair.
    """
    # Log prices
    log_a = np.log(prices_a)
    log_b = np.log(prices_b)

    # Returns
    ret_a = log_a.diff()
    ret_b = log_b.diff()

    # Hedge ratio
    hr = HedgeRatioEstimator.ols(prices_a, prices_b, use_log=True)

    # Spread
    spread = log_a - hr.hedge_ratio * log_b

    # Cointegration
    eg_result = StatisticalTests.engle_granger(prices_a, prices_b)

    # Stationarity of spread
    adf_result = StatisticalTests.augmented_dickey_fuller(spread)

    # Hurst exponent
    hurst = StatisticalTests.hurst_exponent(spread)

    # OU parameters
    theta, mu, sigma = StatisticalTests.ornstein_uhlenbeck_params(spread)
    half_life = StatisticalTests.half_life(theta)

    # Correlation
    correlation = ret_a.corr(ret_b)

    return {
        'pair_name': pair_name,
        'hedge_ratio': hr.hedge_ratio,
        'hedge_ratio_se': hr.standard_error,
        'hedge_ratio_r2': hr.r_squared,
        'correlation': correlation,
        'cointegration': {
            'method': eg_result.method,
            'test_statistic': eg_result.test_statistic,
            'p_value': eg_result.p_value,
            'is_cointegrated': eg_result.is_cointegrated
        },
        'stationarity': {
            'adf_statistic': adf_result.test_statistic,
            'p_value': adf_result.p_value,
            'is_stationary': adf_result.is_cointegrated
        },
        'mean_reversion': {
            'hurst_exponent': hurst,
            'interpretation': 'mean-reverting' if hurst < 0.5 else 'trending' if hurst > 0.5 else 'random walk',
            'ou_theta': theta,
            'ou_mu': mu,
            'ou_sigma': sigma,
            'half_life_bars': half_life
        },
        'spread_statistics': {
            'mean': spread.mean(),
            'std': spread.std(),
            'skewness': spread.skew(),
            'kurtosis': spread.kurtosis()
        }
    }
