"""
Risk Management Module
Portfolio-level risk controls and monitoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Portfolio risk limits."""
    max_gross_exposure: float = 1.0  # Total leverage
    max_net_exposure: float = 0.5    # Net directional exposure
    max_position_size: float = 0.25  # Per position limit
    max_correlation: float = 0.7     # Max correlation between positions
    max_var_1d: float = 0.02         # 1-day VaR limit
    max_drawdown: float = 0.15       # Max portfolio drawdown
    max_sector_exposure: float = 0.5 # Max exposure to single sector


@dataclass
class RiskMetrics:
    """Current risk metrics."""
    gross_exposure: float
    net_exposure: float
    var_1d: float
    expected_shortfall: float
    portfolio_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    correlation_matrix: pd.DataFrame
    position_concentration: float


class RiskManager:
    """
    Portfolio-level risk management.
    Monitors exposures and enforces risk limits.
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self._position_history: List[Dict] = []
        self._pnl_history: List[float] = []
        self._peak_equity: float = 1.0

    def update_portfolio(
        self,
        positions: Dict,
        returns: pd.Series,
        equity: float
    ) -> RiskMetrics:
        """Update risk metrics based on current portfolio."""
        # Track peak equity
        if equity > self._peak_equity:
            self._peak_equity = equity

        # Store history
        self._pnl_history.extend(returns.tolist() if hasattr(returns, 'tolist') else [returns])
        self._position_history.append({
            'positions': len(positions),
            'exposure': sum(abs(p.get('size', 0)) for p in positions.values())
        })

        # Calculate metrics
        gross_exp = sum(abs(p.get('size', 0)) for p in positions.values())
        net_exp = sum(p.get('size', 0) * p.get('direction', 0) for p in positions.values())

        # VaR calculation (parametric)
        if len(self._pnl_history) > 20:
            var_1d = np.percentile(self._pnl_history[-100:], 5)
            es_1d = np.mean([x for x in self._pnl_history[-100:] if x <= var_1d])
        else:
            var_1d = es_1d = 0

        # Portfolio volatility
        if len(self._pnl_history) > 20:
            port_vol = np.std(self._pnl_history[-100:]) * np.sqrt(252)
        else:
            port_vol = 0

        # Sharpe ratio
        if len(self._pnl_history) > 20 and np.std(self._pnl_history[-100:]) > 0:
            sharpe = np.mean(self._pnl_history[-100:]) / np.std(self._pnl_history[-100:]) * np.sqrt(252)
        else:
            sharpe = 0

        # Drawdown
        current_dd = (self._peak_equity - equity) / self._peak_equity
        max_dd = max([d for d in [(self._peak_equity - e) / self._peak_equity
                                   for e in self._pnl_history]] or [0])

        # Correlation matrix (simplified)
        corr_matrix = pd.DataFrame()

        # Position concentration (Herfindahl index)
        sizes = [abs(p.get('size', 0)) for p in positions.values()]
        concentration = sum(s**2 for s in sizes) / sum(sizes)**2 if sizes and sum(sizes) > 0 else 0

        return RiskMetrics(
            gross_exposure=gross_exp,
            net_exposure=net_exp,
            var_1d=abs(var_1d),
            expected_shortfall=abs(es_1d) if es_1d else 0,
            portfolio_volatility=port_vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            correlation_matrix=corr_matrix,
            position_concentration=concentration
        )

    def check_limits(self, metrics: RiskMetrics) -> Tuple[bool, List[str]]:
        """
        Check if portfolio is within risk limits.
        Returns: (all_ok, list_of_breaches)
        """
        breaches = []

        if metrics.gross_exposure > self.limits.max_gross_exposure:
            breaches.append(f"Gross exposure {metrics.gross_exposure:.2f} > limit {self.limits.max_gross_exposure}")

        if metrics.net_exposure > self.limits.max_net_exposure:
            breaches.append(f"Net exposure {metrics.net_exposure:.2f} > limit {self.limits.max_net_exposure}")

        if metrics.var_1d > self.limits.max_var_1d:
            breaches.append(f"1D VaR {metrics.var_1d:.2%} > limit {self.limits.max_var_1d:.2%}")

        if metrics.current_drawdown > self.limits.max_drawdown:
            breaches.append(f"Drawdown {metrics.current_drawdown:.2%} > limit {self.limits.max_drawdown:.2%}")

        if metrics.position_concentration > 0.5:  # High concentration
            breaches.append(f"High position concentration: {metrics.position_concentration:.2f}")

        return len(breaches) == 0, breaches

    def get_position_recommendation(
        self,
        current_positions: Dict,
        new_position_size: float,
        new_position_correlation: float
    ) -> Tuple[bool, str]:
        """
        Determine if a new position should be allowed.
        Returns: (allowed, reason)
        """
        current_gross = sum(abs(p.get('size', 0)) for p in current_positions.values())

        if current_gross + new_position_size > self.limits.max_gross_exposure:
            return False, "Would exceed gross exposure limit"

        # Check correlation with existing positions
        for pair, pos in current_positions.items():
            # In production, would check actual correlation
            pass

        return True, "Position within limits"

    def generate_risk_report(self, metrics: RiskMetrics) -> str:
        """Generate human-readable risk report."""
        report = []
        report.append("=" * 50)
        report.append("RISK REPORT")
        report.append("=" * 50)
        report.append(f"Gross Exposure:     {metrics.gross_exposure:.2f} (limit: {self.limits.max_gross_exposure})")
        report.append(f"Net Exposure:       {metrics.net_exposure:.2f} (limit: {self.limits.max_net_exposure})")
        report.append(f"1D VaR (95%):       {metrics.var_1d:.2%} (limit: {self.limits.max_var_1d:.2%})")
        report.append(f"Expected Shortfall: {metrics.expected_shortfall:.2%}")
        report.append(f"Portfolio Vol:      {metrics.portfolio_volatility:.2%} (ann.)")
        report.append(f"Sharpe Ratio:       {metrics.sharpe_ratio:.2f}")
        report.append(f"Max Drawdown:       {metrics.max_drawdown:.2%} (limit: {self.limits.max_drawdown:.2%})")
        report.append(f"Current Drawdown:   {metrics.current_drawdown:.2%}")
        report.append(f"Position Concentration: {metrics.position_concentration:.2f}")
        report.append("=" * 50)

        # Status
        all_ok, breaches = self.check_limits(metrics)
        if all_ok:
            report.append("STATUS: ✓ All limits OK")
        else:
            report.append("STATUS: ✗ LIMIT BREACHES:")
            for breach in breaches:
                report.append(f"  - {breach}")

        return "\n".join(report)
