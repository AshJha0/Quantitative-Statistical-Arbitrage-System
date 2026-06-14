"""
Mean Reversion Strategy Module
Implements trading logic based on Z-score signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signal types."""
    STRONG_LONG = 2
    LONG = 1
    NEUTRAL = 0
    SHORT = -1
    STRONG_SHORT = -2


class Position(Enum):
    """Position types for spread trading."""
    LONG_SPREAD = 1    # Buy asset A, sell asset B
    SHORT_SPREAD = -1  # Sell asset A, buy asset B
    FLAT = 0


@dataclass
class TradeConfig:
    """Configuration for mean reversion trading."""
    # Entry thresholds
    entry_threshold: float = 2.0       # Z-score to enter trade
    strong_entry_threshold: float = 3.0  # Z-score for strong signal

    # Exit thresholds
    exit_threshold: float = 0.5        # Z-score to exit (take profit)
    stop_loss_threshold: float = 4.0   # Z-score for stop loss

    # Position management
    max_position_size: float = 1.0     # Maximum position (normalized)
    min_holding_period: int = 5        # Minimum bars to hold position
    max_holding_period: int = 100      # Maximum bars before forced exit

    # Risk management
    max_drawdown: float = 0.10         # Maximum portfolio drawdown
    position_limit: float = 0.25       # Max capital per position
    correlation_limit: float = 0.3     # Max correlation between positions

    # Scaling
    enable_scaling: bool = True        # Scale position by Z-score magnitude
    scale_min_z: float = 2.0
    scale_max_z: float = 4.0


@dataclass
class TradeSignal:
    """Represents a trading signal."""
    timestamp: int
    asset_a: str
    asset_b: str
    signal: Signal
    z_score: float
    spread_value: float
    hedge_ratio: float
    suggested_size: float
    confidence: float
    reason: str


@dataclass
class PositionState:
    """Tracks the state of an open position."""
    asset_a: str
    asset_b: str
    entry_time: int
    entry_zscore: float
    entry_spread: float
    position_type: Position
    size: float
    current_zscore: float
    unrealized_pnl: float = 0.0
    bars_held: int = 0


class MeanReversionStrategy:
    """
    Implements mean reversion trading on spread models.
    Enters when Z-score deviates significantly and exits on reversion.
    """

    def __init__(self, config: Optional[TradeConfig] = None):
        self.config = config or TradeConfig()
        self._positions: Dict[Tuple[str, str], PositionState] = {}
        self._trade_history: List[TradeSignal] = []
        self._signals_history: List[Dict] = []

    def compute_signal(
        self,
        z_score: float,
        current_spread: float,
        spread_std: float
    ) -> Signal:
        """
        Determine trading signal based on Z-score.
        """
        if z_score < -self.config.strong_entry_threshold:
            return Signal.STRONG_LONG
        elif z_score < -self.config.entry_threshold:
            return Signal.LONG
        elif z_score > self.config.strong_entry_threshold:
            return Signal.STRONG_SHORT
        elif z_score > self.config.entry_threshold:
            return Signal.SHORT
        else:
            return Signal.NEUTRAL

    def compute_position_size(
        self,
        signal: Signal,
        z_score: float,
        current_volatility: Optional[float] = None
    ) -> float:
        """
        Compute position size based on signal strength and volatility.
        Uses volatility scaling for risk management.
        """
        base_size = self.config.max_position_size

        if not self.config.enable_scaling:
            return base_size * abs(signal.value) / 2

        # Scale by Z-score magnitude
        z_magnitude = abs(z_score)
        if z_magnitude <= self.config.scale_min_z:
            scale_factor = 0.0
        elif z_magnitude >= self.config.scale_max_z:
            scale_factor = 1.0
        else:
            scale_factor = (z_magnitude - self.config.scale_min_z) / \
                          (self.config.scale_max_z - self.config.scale_min_z)

        # Volatility adjustment (reduce size in high vol)
        vol_adjustment = 1.0
        if current_volatility and current_volatility > 0:
            target_vol = 0.02  # 2% daily vol target
            vol_adjustment = min(target_vol / current_volatility, 1.5)

        size = base_size * scale_factor * vol_adjustment
        return min(size, base_size)

    def should_enter(
        self,
        signal: Signal,
        position: Optional[PositionState] = None
    ) -> bool:
        """Determine if we should enter a new position."""
        # Don't enter if already positioned
        if position is not None:
            return False

        # Check signal strength
        if abs(signal.value) < 1:
            return False

        return True

    def should_exit(
        self,
        position: PositionState,
        current_zscore: float,
        current_pnl: float
    ) -> Tuple[bool, str]:
        """
        Determine if we should exit an existing position.
        Returns: (should_exit, reason)
        """
        # Check minimum holding period
        if position.bars_held < self.config.min_holding_period:
            return False, "min_holding_period"

        # Check maximum holding period
        if position.bars_held >= self.config.max_holding_period:
            return True, "max_holding_period"

        # Take profit: Z-score reverted to mean
        if abs(current_zscore) <= self.config.exit_threshold:
            return True, "mean_reversion"

        # Stop loss: Z-score continued against position
        if position.position_type == Position.LONG_SPREAD:
            if current_zscore > self.config.stop_loss_threshold:
                return True, "stop_loss"
        elif position.position_type == Position.SHORT_SPREAD:
            if current_zscore < -self.config.stop_loss_threshold:
                return True, "stop_loss"

        return False, "hold"

    def generate_signal(
        self,
        asset_a: str,
        asset_b: str,
        z_score: float,
        current_spread: float,
        spread_std: float,
        hedge_ratio: float,
        timestamp: int,
        current_volatility: Optional[float] = None
    ) -> TradeSignal:
        """Generate a complete trading signal."""
        signal = self.compute_signal(z_score, current_spread, spread_std)
        size = self.compute_position_size(signal, z_score, current_volatility)

        # Confidence based on Z-score magnitude and distance from thresholds
        confidence = min(abs(z_score) / self.config.strong_entry_threshold, 1.0)

        reason = self._get_signal_reason(signal, z_score)

        return TradeSignal(
            timestamp=timestamp,
            asset_a=asset_a,
            asset_b=asset_b,
            signal=signal,
            z_score=z_score,
            spread_value=current_spread,
            hedge_ratio=hedge_ratio,
            suggested_size=size,
            confidence=confidence,
            reason=reason
        )

    def _get_signal_reason(self, signal: Signal, z_score: float) -> str:
        """Get human-readable reason for signal."""
        if signal == Signal.STRONG_LONG:
            return f"Spread extremely cheap (Z={z_score:.2f} < -{self.config.strong_entry_threshold})"
        elif signal == Signal.LONG:
            return f"Spread cheap (Z={z_score:.2f} < -{self.config.entry_threshold})"
        elif signal == Signal.STRONG_SHORT:
            return f"Spread extremely rich (Z={z_score:.2f} > {self.config.strong_entry_threshold})"
        elif signal == Signal.SHORT:
            return f"Spread rich (Z={z_score:.2f} > {self.config.entry_threshold})"
        else:
            return "No clear signal - wait for better entry"

    def update_position(
        self,
        asset_a: str,
        asset_b: str,
        current_zscore: float,
        current_pnl: float
    ) -> Optional[Tuple[bool, str]]:
        """
        Update existing position and check for exit.
        Returns: (should_exit, reason) or None if no position
        """
        key = (asset_a, asset_b)
        if key not in self._positions:
            return None

        position = self._positions[key]
        position.current_zscore = current_zscore
        position.unrealized_pnl = current_pnl
        position.bars_held += 1

        should_exit, reason = self.should_exit(position, current_zscore, current_pnl)
        return (should_exit, reason)

    def open_position(
        self,
        signal: TradeSignal,
        size: float
    ) -> PositionState:
        """Open a new position based on signal."""
        position_type = Position.LONG_SPREAD if signal.signal.value > 0 else Position.SHORT_SPREAD

        position = PositionState(
            asset_a=signal.asset_a,
            asset_b=signal.asset_b,
            entry_time=signal.timestamp,
            entry_zscore=signal.z_score,
            entry_spread=signal.spread_value,
            position_type=position_type,
            size=size,
            current_zscore=signal.z_score,
            unrealized_pnl=0.0,
            bars_held=0
        )

        self._positions[(signal.asset_a, signal.asset_b)] = position
        return position

    def close_position(self, asset_a: str, asset_b: str, reason: str) -> Optional[PositionState]:
        """Close an existing position."""
        key = (asset_a, asset_b)
        if key not in self._positions:
            return None

        position = self._positions.pop(key)
        logger.info(f"Closed position {asset_a}/{asset_b}: {reason}, PnL: {position.unrealized_pnl:.4f}")
        return position

    def get_position(self, asset_a: str, asset_b: str) -> Optional[PositionState]:
        """Get current position for a pair."""
        return self._positions.get((asset_a, asset_b))

    def get_all_positions(self) -> Dict[Tuple[str, str], PositionState]:
        """Get all open positions."""
        return self._positions.copy()

    def get_portfolio_exposure(self) -> Dict:
        """Calculate total portfolio exposure and risk metrics."""
        if not self._positions:
            return {
                'total_positions': 0,
                'gross_exposure': 0.0,
                'net_exposure': 0.0,
                'avg_zscore': 0.0
            }

        positions = list(self._positions.values())
        total_size = sum(abs(p.size) for p in positions)
        net_size = sum(p.size * p.position_type.value for p in positions)
        avg_zscore = np.mean([p.current_zscore for p in positions])

        return {
            'total_positions': len(positions),
            'gross_exposure': total_size,
            'net_exposure': net_size,
            'avg_zscore': avg_zscore
        }


class AdaptiveMeanReversionStrategy(MeanReversionStrategy):
    """
    Enhanced mean reversion with adaptive thresholds.
    Adjusts entry/exit thresholds based on market regime.
    """

    def __init__(
        self,
        config: Optional[TradeConfig] = None,
        regime_lookback: int = 60
    ):
        super().__init__(config)
        self.regime_lookback = regime_lookback
        self._volatility_regime: str = 'normal'  # 'low', 'normal', 'high'
        self._threshold_multiplier: float = 1.0

    def update_regime(self, volatility_history: pd.Series) -> None:
        """
        Update market regime based on recent volatility.
        Wider thresholds in high vol, tighter in low vol.
        """
        current_vol = volatility_history.iloc[-1]
        avg_vol = volatility_history.rolling(self.regime_lookback).mean().iloc[-1]
        vol_percentile = (volatility_history.iloc[-self.regime_lookback:] <= current_vol).mean()

        if vol_percentile > 0.8:
            self._volatility_regime = 'high'
            self._threshold_multiplier = 1.2  # Wider thresholds
        elif vol_percentile < 0.2:
            self._volatility_regime = 'low'
            self._threshold_multiplier = 0.8  # Tighter thresholds
        else:
            self._volatility_regime = 'normal'
            self._threshold_multiplier = 1.0

    def get_effective_thresholds(self) -> Dict[str, float]:
        """Get thresholds adjusted for current regime."""
        m = self._threshold_multiplier
        return {
            'entry_threshold': self.config.entry_threshold * m,
            'strong_entry_threshold': self.config.strong_entry_threshold * m,
            'exit_threshold': self.config.exit_threshold * m,
            'stop_loss_threshold': self.config.stop_loss_threshold * m
        }

    @property
    def current_regime(self) -> str:
        return self._volatility_regime
