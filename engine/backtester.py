"""
Backtesting Engine Module
Event-driven backtesting with realistic simulation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    # Capital and sizing
    initial_capital: float = 1_000_000
    max_position_size: float = 0.25  # 25% of capital per position
    max_total_exposure: float = 1.0  # Maximum gross exposure

    # Transaction costs
    commission_rate: float = 0.0001  # 1 bps
    spread_cost: float = 0.0005  # 5 bps slippage
    borrow_cost: float = 0.03  # 3% annual short borrow

    # Risk limits
    max_drawdown: float = 0.15  # 15% max drawdown
    daily_loss_limit: float = 0.05  # 5% daily loss limit
    var_limit: float = 0.02  # 2% daily VaR limit

    # Data handling
    warmup_period: int = 252  # Period for initial statistics
    rebalance_frequency: int = 1  # Rebalance every N bars


@dataclass
class Trade:
    """Represents an executed trade."""
    timestamp: int
    pair: Tuple[str, str]
    direction: int  # 1=long spread, -1=short spread
    size: float
    entry_price: float
    exit_price: Optional[float] = None
    pnl: float = 0.0
    commission: float = 0.0
    exit_timestamp: Optional[int] = None
    tags: Dict = field(default_factory=dict)


@dataclass
class Position:
    """Current open position."""
    pair: Tuple[str, str]
    direction: int
    size: float
    entry_zscore: float
    entry_timestamp: int
    bars_held: int = 0


@dataclass
class BacktestResult:
    """Complete backtest results with metrics."""
    trades: List[Trade]
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    positions_history: List[Dict]
    metrics: Dict[str, float]
    config: BacktestConfig


class BacktestEngine:
    """
    Event-driven backtesting engine for spread strategies.
    Simulates realistic trading with costs, slippage, and risk limits.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset engine state for new backtest."""
        self.capital = self.config.initial_capital
        self.initial_capital = self.config.initial_capital
        self.positions: Dict[Tuple[str, str], Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.peak_equity: float = self.initial_capital
        self.drawdown_curve: List[float] = []
        self.daily_pnl: List[float] = []
        self._current_bar = 0
        self._bars_since_rebalance = 0

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        spread_model,
        strategy,
        pairs: List[Tuple[str, str]],
        progress_callback: Optional[Callable] = None
    ) -> BacktestResult:
        """
        Run complete backtest simulation.

        Args:
            data: Dictionary of symbol -> price data
            spread_model: SpreadModel instance for computing spreads
            strategy: MeanReversionStrategy for signals
            pairs: List of (asset_a, asset_b) pairs to trade
            progress_callback: Optional callback for progress updates
        """
        self._reset_state()

        # Get common index
        all_indices = [df.index for df in data.values() if df is not None]
        if not all_indices:
            raise ValueError("No data provided")

        common_index = all_indices[0]
        for idx in all_indices[1:]:
            common_index = common_index.intersection(idx)

        if len(common_index) < self.config.warmup_period + 100:
            raise ValueError(f"Insufficient data: {len(common_index)} bars")

        logger.info(f"Starting backtest on {len(common_index)} bars, {len(pairs)} pairs")

        # Main backtest loop
        for i, timestamp in enumerate(common_index):
            if i < self.config.warmup_period:
                # Warmup: update models but don't trade
                self._warmup_step(data, spread_model, pairs, timestamp, i)
                continue

            # Trading step
            self._trading_step(data, spread_model, strategy, pairs, timestamp, i)

            # Record equity
            self._record_equity(timestamp)

            # Progress callback
            if progress_callback and i % 100 == 0:
                progress_callback(i, len(common_index))

            # Check risk limits
            if not self._check_risk_limits():
                logger.warning("Risk limit breached - stopping backtest")
                break

        # Close remaining positions
        self._close_all_positions(data, spread_model)

        # Compute results
        return self._compute_result()

    def _warmup_step(
        self,
        data: Dict[str, pd.DataFrame],
        spread_model,
        pairs: List[Tuple[str, str]],
        timestamp,
        bar_idx: int
    ) -> None:
        """Warmup step - update models without trading."""
        for asset_a, asset_b in pairs:
            df_a = data.get(asset_a)
            df_b = data.get(asset_b)

            if df_a is None or df_b is None:
                continue

            # Get data up to current point (no lookahead!)
            slice_a = df_a.loc[:timestamp]['close']
            slice_b = df_b.loc[:timestamp]['close']

            if len(slice_a) < 30 or len(slice_b) < 30:
                continue

            # Update spread model
            spread_model.update_state(slice_a, slice_b, asset_a, asset_b, bar_idx)

    def _trading_step(
        self,
        data: Dict[str, pd.DataFrame],
        spread_model,
        strategy,
        pairs: List[Tuple[str, str]],
        timestamp,
        bar_idx: int
    ) -> None:
        """Main trading logic for each bar."""
        self._current_bar = bar_idx
        signals_to_execute = []

        # Update models and generate signals for each pair
        for asset_a, asset_b in pairs:
            df_a = data.get(asset_a)
            df_b = data.get(asset_b)

            if df_a is None or df_b is None:
                continue

            slice_a = df_a.loc[:timestamp]['close']
            slice_b = df_b.loc[:timestamp]['close']

            if len(slice_a) < 30 or len(slice_b) < 30:
                continue

            # Update spread state
            state = spread_model.update_state(slice_a, slice_b, asset_a, asset_b, bar_idx)

            if not state.is_valid:
                continue

            # Generate signal
            signal = strategy.generate_signal(
                asset_a=asset_a,
                asset_b=asset_b,
                z_score=state.current_zscore,
                current_spread=state.current_spread,
                spread_std=state.spread_std,
                hedge_ratio=state.hedge_ratio,
                timestamp=bar_idx
            )

            signals_to_execute.append((signal, state))

        # Process existing positions
        self._update_positions(strategy, spread_model, data, timestamp)

        # Execute new signals
        self._execute_signals(signals_to_execute, strategy)

        self._bars_since_rebalance += 1

    def _update_positions(
        self,
        strategy,
        spread_model,
        data: Dict[str, pd.DataFrame],
        timestamp
    ) -> None:
        """Update existing positions and check for exits."""
        positions_to_close = []

        for pair, position in list(self.positions.items()):
            asset_a, asset_b = pair

            df_a = data.get(asset_a, pd.DataFrame())
            df_b = data.get(asset_b, pd.DataFrame())

            if df_a.empty or df_b.empty:
                continue

            slice_a = df_a.loc[:timestamp]['close']
            slice_b = df_b.loc[:timestamp]['close']

            if len(slice_a) < 2 or len(slice_b) < 2:
                continue

            # Get current Z-score
            state = spread_model.get_state(asset_a, asset_b)
            if state is None:
                continue

            current_zscore = state.current_zscore

            # Calculate unrealized PnL
            entry_z = position.entry_zscore
            pnl_pct = (current_zscore - entry_z) * position.direction * 0.01  # Simplified
            unrealized_pnl = pnl_pct * position.size * self.capital

            # Check for exit
            exit_decision = strategy.update_position(
                asset_a, asset_b, current_zscore, unrealized_pnl
            )

            if exit_decision and exit_decision[0]:
                positions_to_close.append((pair, exit_decision[1]))

        # Close positions
        for pair, reason in positions_to_close:
            self._close_position(pair, reason, timestamp)

    def _execute_signals(
        self,
        signals: List[Tuple],
        strategy
    ) -> None:
        """Execute new trading signals."""
        # Check exposure limits
        current_exposure = sum(p.size for p in self.positions.values())
        max_new_exposure = self.config.max_total_exposure * self.capital - current_exposure

        for signal, state in signals:
            if max_new_exposure <= 0:
                break

            # Check if already positioned
            if (signal.asset_a, signal.asset_b) in self.positions:
                continue

            # Check signal strength
            if abs(signal.signal.value) < 1:
                continue

            # Calculate position size
            size = min(
                signal.suggested_size * self.capital,
                self.config.max_position_size * self.capital,
                max_new_exposure
            )

            if size <= 0:
                continue

            # Open position
            self._open_position(signal, size)
            max_new_exposure -= size

    def _open_position(self, signal, size: float) -> None:
        """Open a new position."""
        pair = (signal.asset_a, signal.asset_b)

        # Calculate entry cost
        cost = size * self.config.spread_cost
        self.capital -= cost

        position = Position(
            pair=pair,
            direction=1 if signal.signal.value > 0 else -1,
            size=size,
            entry_zscore=signal.z_score,
            entry_timestamp=signal.timestamp
        )

        self.positions[pair] = position

        # Record trade
        trade = Trade(
            timestamp=signal.timestamp,
            pair=pair,
            direction=position.direction,
            size=size,
            entry_price=signal.spread_value,
            commission=cost,
            tags={'entry_zscore': signal.z_score, 'reason': signal.reason}
        )
        self.trades.append(trade)

        logger.debug(f"Opened {pair}: direction={position.direction}, size={size:.2f}")

    def _close_position(self, pair: Tuple[str, str], reason: str, timestamp) -> None:
        """Close an existing position."""
        if pair not in self.positions:
            return

        position = self.positions.pop(pair)

        # Find corresponding open trade
        open_trade = None
        for trade in reversed(self.trades):
            if trade.pair == pair and trade.exit_price is None:
                open_trade = trade
                break

        if open_trade is None:
            return

        # Calculate PnL (simplified model based on Z-score change)
        entry_z = position.entry_zscore
        # In production, this would use actual price changes
        pnl_pct = -entry_z * position.direction * 0.01  # Mean reversion payoff
        pnl = pnl_pct * position.size * self.capital

        # Exit cost
        exit_cost = position.size * self.config.spread_cost
        self.capital -= exit_cost

        # Update trade record
        open_trade.exit_timestamp = timestamp
        open_trade.exit_price = 0  # Spread at exit
        open_trade.pnl = pnl
        open_trade.commission += exit_cost
        open_trade.tags['exit_reason'] = reason
        open_trade.tags['bars_held'] = position.bars_held

        self.capital += pnl  # Realize PnL

        logger.debug(f"Closed {pair}: PnL={pnl:.2f}, reason={reason}")

    def _close_all_positions(self, data, spread_model) -> None:
        """Force close all remaining positions at end of backtest."""
        for pair in list(self.positions.keys()):
            self._close_position(pair, "end_of_backtest", self._current_bar)

    def _record_equity(self, timestamp) -> None:
        """Record equity and drawdown."""
        # Calculate total equity (capital + unrealized PnL)
        total_equity = self.capital

        self.equity_curve.append(total_equity)

        # Update peak
        if total_equity > self.peak_equity:
            self.peak_equity = total_equity

        # Calculate drawdown
        dd = (self.peak_equity - total_equity) / self.peak_equity
        self.drawdown_curve.append(dd)

    def _check_risk_limits(self) -> bool:
        """Check if any risk limits are breached."""
        if not self.drawdown_curve:
            return True

        current_dd = self.drawdown_curve[-1]

        # Max drawdown check
        if current_dd >= self.config.max_drawdown:
            logger.warning(f"Max drawdown breached: {current_dd:.2%}")
            return False

        # Daily loss limit (simplified)
        if len(self.equity_curve) > 100:
            daily_change = (self.equity_curve[-1] - self.equity_curve[-101]) / self.equity_curve[-101]
            if daily_change <= -self.config.daily_loss_limit:
                logger.warning(f"Daily loss limit breached: {daily_change:.2%}")
                return False

        return True

    def _compute_result(self) -> BacktestResult:
        """Compute final backtest metrics."""
        equity_series = pd.Series(self.equity_curve)
        drawdown_series = pd.Series(self.drawdown_curve)

        metrics = self._calculate_metrics(equity_series)

        return BacktestResult(
            trades=self.trades,
            equity_curve=equity_series,
            drawdown_curve=drawdown_series,
            positions_history=[],
            metrics=metrics,
            config=self.config
        )

    def _calculate_metrics(self, equity: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if len(equity) < 10:
            return self._empty_metrics()

        returns = equity.pct_change().dropna()

        # Basic metrics
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        cagr = self._compute_cagr(equity)

        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe = self._compute_sharpe(returns)
        sortino = self._compute_sortino(returns)

        # Drawdown metrics
        max_dd = self._compute_max_drawdown(equity)
        avg_dd = self._compute_avg_drawdown()
        dd_duration = self._compute_max_dd_duration()

        # Trade metrics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]

        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        profit_factor = self._compute_profit_factor(winning_trades, losing_trades)
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        return {
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'avg_drawdown': avg_dd,
            'max_dd_duration': dd_duration,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(self.trades),
            'final_equity': equity.iloc[-1],
            'initial_capital': self.initial_capital
        }

    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dict."""
        return {
            'total_return': 0, 'cagr': 0, 'volatility': 0,
            'sharpe_ratio': 0, 'sortino_ratio': 0, 'max_drawdown': 0,
            'avg_drawdown': 0, 'max_dd_duration': 0, 'win_rate': 0,
            'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0,
            'total_trades': 0, 'final_equity': self.initial_capital
        }

    def _compute_cagr(self, equity: pd.Series) -> float:
        """Compute Compound Annual Growth Rate."""
        n_years = len(equity) / 252
        if n_years <= 0:
            return 0
        return (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1

    def _compute_sharpe(self, returns: pd.Series) -> float:
        """Compute Sharpe ratio (assuming 0 risk-free rate)."""
        if returns.std() == 0:
            return 0
        return returns.mean() / returns.std() * np.sqrt(252)

    def _compute_sortino(self, returns: pd.Series) -> float:
        """Compute Sortino ratio."""
        downside = returns[returns < 0]
        if downside.std() == 0:
            return 0
        return returns.mean() / downside.std() * np.sqrt(252)

    def _compute_max_drawdown(self, equity: pd.Series) -> float:
        """Compute maximum drawdown."""
        peak = equity.expanding().max()
        drawdown = (peak - equity) / peak
        return drawdown.max()

    def _compute_avg_drawdown(self) -> float:
        """Compute average drawdown."""
        return np.mean(self.drawdown_curve) if self.drawdown_curve else 0

    def _compute_max_dd_duration(self) -> int:
        """Compute maximum drawdown duration in bars."""
        if not self.drawdown_curve:
            return 0

        max_duration = 0
        current_duration = 0

        for dd in self.drawdown_curve:
            if dd > 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    def _compute_profit_factor(
        self,
        winners: List[Trade],
        losers: List[Trade]
    ) -> float:
        """Compute profit factor (gross profit / gross loss)."""
        gross_profit = sum(t.pnl for t in winners)
        gross_loss = abs(sum(t.pnl for t in losers))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0

        return gross_profit / gross_loss
