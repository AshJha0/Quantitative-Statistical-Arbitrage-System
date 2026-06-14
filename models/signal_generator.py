"""
Signal Generator Module
Transforms normalized spreads into actionable trading signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """Signal strength levels."""
    VERY_STRONG = 4
    STRONG = 3
    MODERATE = 2
    WEAK = 1
    NEUTRAL = 0


class SignalDirection(Enum):
    """Direction of trade signal."""
    LONG_SPREAD = 1      # Buy asset A, sell asset B
    SHORT_SPREAD = -1    # Sell asset A, buy asset B
    FLAT = 0


@dataclass
class SignalConfig:
    """Configuration for signal generation."""
    # Z-score thresholds
    very_strong_threshold: float = 3.0
    strong_threshold: float = 2.5
    moderate_threshold: float = 2.0
    weak_threshold: float = 1.5

    # Exit thresholds
    take_profit_z: float = 0.5
    stop_loss_z: float = 4.0

    # Confirmation
    require_confirmation: bool = True
    confirmation_bars: int = 3

    # Filter settings
    min_correlation: float = 0.5
    min_cointegration_score: float = 0.7
    max_spread_volatility: Optional[float] = None


@dataclass
class TradingSignal:
    """Complete trading signal with metadata."""
    timestamp: int
    pair: Tuple[str, str]

    # Signal properties
    direction: SignalDirection
    strength: SignalStrength
    z_score: float
    spread_value: float

    # Position sizing
    recommended_size: float
    stop_loss_z: float
    take_profit_z: float

    # Confidence metrics
    confidence_score: float
    correlation: float
    cointegration_pvalue: float
    half_life: float

    # Metadata
    reason: str
    tags: List[str]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'pair': f"{self.pair[0]}/{self.pair[1]}",
            'direction': self.direction.name,
            'strength': self.strength.name,
            'z_score': round(self.z_score, 4),
            'spread_value': round(self.spread_value, 6),
            'recommended_size': round(self.recommended_size, 4),
            'stop_loss_z': round(self.stop_loss_z, 2),
            'take_profit_z': round(self.take_profit_z, 2),
            'confidence_score': round(self.confidence_score, 4),
            'correlation': round(self.correlation, 4),
            'cointegration_pvalue': round(self.cointegration_pvalue, 4) if self.cointegration_pvalue else None,
            'half_life': round(self.half_life, 2),
            'reason': self.reason,
            'tags': self.tags
        }


class SignalGenerator:
    """
    Generates trading signals from normalized spread data.
    Combines Z-score analysis with confirmation filters.
    """

    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()
        self._signal_history: List[TradingSignal] = []
        self._pending_signals: Dict[Tuple[str, str], List[TradingSignal]] = {}

    def determine_signal_strength(self, z_score: float) -> SignalStrength:
        """
        Map Z-score to signal strength.
        """
        abs_z = abs(z_score)

        if abs_z >= self.config.very_strong_threshold:
            return SignalStrength.VERY_STRONG
        elif abs_z >= self.config.strong_threshold:
            return SignalStrength.STRONG
        elif abs_z >= self.config.moderate_threshold:
            return SignalStrength.MODERATE
        elif abs_z >= self.config.weak_threshold:
            return SignalStrength.WEAK
        else:
            return SignalStrength.NEUTRAL

    def determine_direction(self, z_score: float) -> SignalDirection:
        """
        Determine trade direction from Z-score.

        Negative Z-score → spread is cheap → LONG spread
        Positive Z-score → spread is rich → SHORT spread
        """
        if z_score < -self.config.weak_threshold:
            return SignalDirection.LONG_SPREAD
        elif z_score > self.config.weak_threshold:
            return SignalDirection.SHORT_SPREAD
        else:
            return SignalDirection.FLAT

    def compute_position_size(
        self,
        strength: SignalStrength,
        z_score: float,
        spread_volatility: float,
        half_life: float,
        max_size: float = 1.0
    ) -> float:
        """
        Compute recommended position size based on signal quality.

        Factors:
        - Signal strength (stronger = larger)
        - Z-score magnitude (more extreme = larger)
        - Spread volatility (higher vol = smaller)
        - Half-life (shorter = larger, faster turnover)
        """
        # Base size from strength
        base_sizes = {
            SignalStrength.VERY_STRONG: 1.0,
            SignalStrength.STRONG: 0.75,
            SignalStrength.MODERATE: 0.5,
            SignalStrength.WEAK: 0.25,
            SignalStrength.NEUTRAL: 0.0
        }
        base_size = base_sizes.get(strength, 0)

        # Volatility adjustment (inverse)
        if spread_volatility > 0:
            vol_target = 0.02  # 2% daily vol target
            vol_adjustment = min(vol_target / spread_volatility, 1.5)
        else:
            vol_adjustment = 1.0

        # Half-life adjustment (faster mean reversion = can size up)
        if half_life > 0 and half_life < 20:
            hl_adjustment = 1.2  # Fast reversion
        elif half_life > 50:
            hl_adjustment = 0.8  # Slow reversion
        else:
            hl_adjustment = 1.0

        size = base_size * vol_adjustment * hl_adjustment
        return min(size, max_size)

    def compute_confidence_score(
        self,
        z_score: float,
        correlation: float,
        cointegration_pvalue: Optional[float],
        half_life: float,
        spread_kurtosis: float = 0.0
    ) -> float:
        """
        Compute overall confidence score (0 to 1).

        Factors:
        - Z-score magnitude (higher = more confident)
        - Correlation (higher = more confident)
        - Cointegration (significant = more confident)
        - Half-life (reasonable = more confident)
        - Kurtosis (normal = more confident)
        """
        scores = []

        # Z-score component (0.3 weight)
        z_score_component = min(abs(z_score) / 3.0, 1.0)
        scores.append(z_score_component)

        # Correlation component (0.25 weight)
        corr_component = max(0, (abs(correlation) - 0.3) / 0.7)
        scores.append(corr_component)

        # Cointegration component (0.25 weight)
        if cointegration_pvalue is not None:
            coint_component = 1.0 - min(cointegration_pvalue / 0.05, 1.0)
        else:
            coint_component = 0.5  # Neutral if unknown
        scores.append(coint_component)

        # Half-life component (0.1 weight)
        # Optimal half-life is 5-30 bars
        if 5 <= half_life <= 30:
            hl_component = 1.0
        elif half_life < 5:
            hl_component = max(0.3, half_life / 5)
        else:
            hl_component = max(0.3, 1.0 - (half_life - 30) / 50)
        scores.append(hl_component)

        # Kurtosis component (0.1 weight) - penalize fat tails
        kurt_component = 1.0 / (1.0 + abs(spread_kurtosis) / 10)
        scores.append(kurt_component)

        # Weighted average
        weights = [0.3, 0.25, 0.25, 0.1, 0.1]
        confidence = sum(s * w for s, w in zip(scores, weights))

        return min(max(confidence, 0), 1)

    def generate_signal(
        self,
        pair: Tuple[str, str],
        z_score: float,
        spread_value: float,
        spread_std: float,
        correlation: float,
        cointegration_pvalue: Optional[float],
        half_life: float,
        timestamp: int,
        spread_kurtosis: float = 0.0,
        momentum: float = 0.0
    ) -> Optional[TradingSignal]:
        """
        Generate a complete trading signal.

        Args:
            pair: (asset_a, asset_b)
            z_score: Current normalized spread value
            spread_value: Raw spread value
            spread_std: Spread volatility
            correlation: Correlation between assets
            cointegration_pvalue: P-value from cointegration test
            half_life: Expected mean reversion half-life
            timestamp: Current bar index
            spread_kurtosis: Kurtosis of spread distribution
            momentum: Recent spread momentum

        Returns:
            TradingSignal or None if signal is too weak
        """
        # Check filters
        if abs(correlation) < self.config.min_correlation:
            logger.debug(f"Pair {pair} filtered: correlation {correlation:.3f} too low")
            return None

        if cointegration_pvalue and cointegration_pvalue > 0.1:
            logger.debug(f"Pair {pair} filtered: cointegration p-value {cointegration_pvalue:.3f} too high")
            return None

        # Determine signal properties
        strength = self.determine_signal_strength(z_score)
        direction = self.determine_direction(z_score)

        if strength == SignalStrength.NEUTRAL:
            return None

        # Compute position size
        position_size = self.compute_position_size(
            strength=strength,
            z_score=z_score,
            spread_volatility=spread_std,
            half_life=half_life
        )

        # Compute confidence
        confidence = self.compute_confidence_score(
            z_score=z_score,
            correlation=correlation,
            cointegration_pvalue=cointegration_pvalue,
            half_life=half_life,
            spread_kurtosis=spread_kurtosis
        )

        # Generate reason
        reason = self._generate_reason(
            direction=direction,
            strength=strength,
            z_score=z_score,
            correlation=correlation,
            half_life=half_life
        )

        # Generate tags
        tags = self._generate_tags(
            strength=strength,
            half_life=half_life,
            momentum=momentum
        )

        signal = TradingSignal(
            timestamp=timestamp,
            pair=pair,
            direction=direction,
            strength=strength,
            z_score=z_score,
            spread_value=spread_value,
            recommended_size=position_size,
            stop_loss_z=self.config.stop_loss_z,
            take_profit_z=self.config.take_profit_z,
            confidence_score=confidence,
            correlation=correlation,
            cointegration_pvalue=cointegration_pvalue or 1.0,
            half_life=half_life,
            reason=reason,
            tags=tags
        )

        # Store in history
        self._signal_history.append(signal)

        # Handle confirmation requirement
        if self.config.require_confirmation and strength in [SignalStrength.WEAK, SignalStrength.MODERATE]:
            return self._handle_pending_signal(signal)

        return signal

    def _generate_reason(
        self,
        direction: SignalDirection,
        strength: SignalStrength,
        z_score: float,
        correlation: float,
        half_life: float
    ) -> str:
        """Generate human-readable reason for signal."""
        direction_str = "LONG" if direction == SignalDirection.LONG_SPREAD else "SHORT"
        cheap_rich = "cheap" if direction == SignalDirection.LONG_SPREAD else "rich"

        reasons = [
            f"Spread {cheap_rich} at Z={z_score:.2f}",
            f"High correlation ({correlation:.2f})",
            f"Half-life: {half_life:.1f} bars"
        ]

        if strength >= SignalStrength.STRONG:
            reasons.insert(0, f"STRONG {direction_str} SIGNAL")

        return " | ".join(reasons)

    def _generate_tags(
        self,
        strength: SignalStrength,
        half_life: float,
        momentum: float
    ) -> List[str]:
        """Generate signal tags for categorization."""
        tags = []

        # Strength tags
        if strength >= SignalStrength.VERY_STRONG:
            tags.append("high_conviction")
        elif strength >= SignalStrength.STRONG:
            tags.append("strong_signal")

        # Half-life tags
        if half_life < 10:
            tags.append("fast_reversion")
        elif half_life > 50:
            tags.append("slow_reversion")

        # Momentum tags
        if abs(momentum) > 0.5:
            tags.append("momentum_conflict" if momentum > 0 else "momentum_aligned")

        return tags

    def _handle_pending_signal(self, signal: TradingSignal) -> Optional[TradingSignal]:
        """
        Handle signals requiring confirmation.
        Stores signal and returns confirmed signals only.
        """
        pair = signal.pair

        if pair not in self._pending_signals:
            self._pending_signals[pair] = []

        self._pending_signals[pair].append(signal)

        # Check if we have enough confirmation
        pending = self._pending_signals[pair]

        if len(pending) >= self.config.confirmation_bars:
            # Check if signals are consistent
            recent = pending[-self.config.confirmation_bars:]
            same_direction = all(s.direction == recent[0].direction for s in recent)

            # Clear old signals
            self._pending_signals[pair] = pending[-self.config.confirmation_bars:]

            if same_direction:
                # Upgrade strength due to confirmation
                confirmed_signal = recent[-1]
                if confirmed_signal.strength == SignalStrength.WEAK:
                    # Upgrade to MODERATE on confirmation
                    return TradingSignal(
                        timestamp=confirmed_signal.timestamp,
                        pair=confirmed_signal.pair,
                        direction=confirmed_signal.direction,
                        strength=SignalStrength.MODERATE,
                        z_score=confirmed_signal.z_score,
                        spread_value=confirmed_signal.spread_value,
                        recommended_size=confirmed_signal.recommended_size * 1.2,
                        stop_loss_z=confirmed_signal.stop_loss_z,
                        take_profit_z=confirmed_signal.take_profit_z,
                        confidence_score=min(confirmed_signal.confidence_score * 1.1, 1.0),
                        correlation=confirmed_signal.correlation,
                        cointegration_pvalue=confirmed_signal.cointegration_pvalue,
                        half_life=confirmed_signal.half_life,
                        reason=f"Confirmed: {confirmed_signal.reason}",
                        tags=confirmed_signal.tags + ["confirmed"]
                    )
                return confirmed_signal

        return None

    def get_signal_history(
        self,
        pair: Optional[Tuple[str, str]] = None,
        limit: int = 100
    ) -> List[TradingSignal]:
        """Get signal history, optionally filtered by pair."""
        if pair:
            signals = [s for s in self._signal_history if s.pair == pair]
        else:
            signals = self._signal_history
        return signals[-limit:]

    def clear_pending(self, pair: Tuple[str, str]) -> None:
        """Clear pending signals for a pair (e.g., after trade execution)."""
        if pair in self._pending_signals:
            self._pending_signals[pair] = []


class SignalAggregator:
    """
    Aggregates signals from multiple pairs into portfolio-level view.
    """

    def __init__(self, max_total_exposure: float = 1.0):
        self.max_total_exposure = max_total_exposure
        self._active_signals: Dict[Tuple[str, str], TradingSignal] = {}

    def add_signal(self, signal: TradingSignal) -> bool:
        """
        Add signal to active set.
        Returns True if signal was accepted.
        """
        # Check total exposure
        current_exposure = sum(
            s.recommended_size for s in self._active_signals.values()
        )

        if current_exposure + signal.recommended_size > self.max_total_exposure:
            logger.debug(f"Signal rejected: would exceed max exposure")
            return False

        self._active_signals[signal.pair] = signal
        return True

    def remove_signal(self, pair: Tuple[str, str]) -> Optional[TradingSignal]:
        """Remove signal (e.g., after position closed)."""
        return self._active_signals.pop(pair, None)

    def get_portfolio_signals(self) -> List[TradingSignal]:
        """Get all active signals."""
        return list(self._active_signals.values())

    def get_top_signals(self, n: int = 5) -> List[TradingSignal]:
        """Get top N signals by confidence."""
        signals = sorted(
            self._active_signals.values(),
            key=lambda s: s.confidence_score,
            reverse=True
        )
        return signals[:n]

    def get_correlation_adjusted_signals(
        self,
        correlation_matrix: pd.DataFrame
    ) -> List[TradingSignal]:
        """
        Adjust signals for correlation between pairs.
        Reduces size for highly correlated signals.
        """
        adjusted = []

        for pair, signal in self._active_signals.items():
            # Find correlation with other active positions
            max_corr = 0
            for other_pair in self._active_signals.keys():
                if other_pair != pair:
                    # Simplified: use asset correlation
                    for asset1 in pair:
                        for asset2 in other_pair:
                            try:
                                corr = abs(correlation_matrix.loc[asset1, asset2])
                                max_corr = max(max_corr, corr)
                            except (KeyError, IndexError):
                                pass

            # Reduce size for correlated positions
            if max_corr > 0.5:
                adjustment = 1.0 - (max_corr - 0.5) * 0.5
                adjusted_signal = TradingSignal(
                    timestamp=signal.timestamp,
                    pair=signal.pair,
                    direction=signal.direction,
                    strength=signal.strength,
                    z_score=signal.z_score,
                    spread_value=signal.spread_value,
                    recommended_size=signal.recommended_size * adjustment,
                    stop_loss_z=signal.stop_loss_z,
                    take_profit_z=signal.take_profit_z,
                    confidence_score=signal.confidence_score,
                    correlation=signal.correlation,
                    cointegration_pvalue=signal.cointegration_pvalue,
                    half_life=signal.half_life,
                    reason=signal.reason,
                    tags=signal.tags + [f"corr_adjusted_{max_corr:.2f}"]
                )
                adjusted.append(adjusted_signal)
            else:
                adjusted.append(signal)

        return adjusted
