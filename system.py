"""
QuantSystem - Complete Production Integration
Orchestrates all components into a unified trading system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from datetime import datetime

# System imports
from data.data_loader import DataLoader, FeatureEngineer
from models.spread_model import SpreadModel, SpreadConfig, SpreadState
from models.signal_generator import SignalGenerator, SignalConfig, TradingSignal
from models.math_utils import StatisticalTests, HedgeRatioEstimator, generate_math_report
from strategies.mean_reversion import MeanReversionStrategy, TradeConfig, Position
from engine.backtester import BacktestEngine, BacktestConfig, BacktestResult
from rlhf.rl_agent import RLHFAgent, TradingEnvironment, RLConfig, RewardConfig
from rlhf.reward_model import RewardModel, RewardWeights
from rlhf.policy_optimizer import PPOTrainer, HumanFeedbackIntegrator
from utils.risk_manager import RiskManager, RiskLimits
from config.settings import SystemConfig, get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """Complete system state snapshot."""
    timestamp: int
    capital: float
    positions: Dict
    signals: List[TradingSignal]
    spread_states: Dict[Tuple[str, str], SpreadState]
    risk_metrics: Dict
    rl_agent_trained: bool
    backtest_complete: bool


@dataclass
class PairConfig:
    """Configuration for a trading pair."""
    asset_a: str
    asset_b: str
    enabled: bool = True
    hedge_ratio_method: str = 'rolling'  # 'static', 'rolling', 'ewm'
    zscore_window: int = 252
    custom_threshold: Optional[float] = None


class QuantSystem:
    """
    Production-grade statistical arbitrage system.

    Integrates:
    - Data loading and feature engineering
    - Spread modeling and cointegration analysis
    - Signal generation with confidence scoring
    - Mean reversion strategy execution
    - Backtesting engine
    - RLHF optimization
    - Risk management
    """

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        data_dir: str = "data",
        model_dir: str = "models",
        output_dir: str = "output",
        initial_capital: float = 1_000_000
    ):
        self.config = config or get_config()
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.initial_capital = initial_capital

        # Create directories
        self._setup_directories()

        # Initialize components
        self._init_components()

        # State
        self._data: Dict[str, pd.DataFrame] = {}
        self._pairs: List[PairConfig] = []
        self._spread_model: Optional[SpreadModel] = None
        self._signal_generator: Optional[SignalGenerator] = None
        self._strategy: Optional[MeanReversionStrategy] = None
        self._backtester: Optional[BacktestEngine] = None
        self._rl_agent: Optional[RLHFAgent] = None
        self._risk_manager: Optional[RiskManager] = None
        self._reward_model: Optional[RewardModel] = None
        self._feedback_integrator: Optional[HumanFeedbackIntegrator] = None

        # Results cache
        self._backtest_result: Optional[BacktestResult] = None
        self._last_state: Optional[SystemState] = None

    def _setup_directories(self) -> None:
        """Create required directories."""
        for dir_path in [self.data_dir, self.model_dir, self.output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _init_components(self) -> None:
        """Initialize all system components."""
        # Data layer
        self._data_loader = DataLoader(str(self.data_dir))
        self._feature_engineer = FeatureEngineer()

        # Model layer defaults
        self._spread_config = SpreadConfig(
            method='rolling_regression',
            hedge_ratio_window=60,
            lookback_period=252,
            min_correlation=0.5,
            min_cointegration_pvalue=0.05
        )

        # Strategy defaults
        self._trade_config = TradeConfig(
            entry_threshold=2.0,
            strong_entry_threshold=3.0,
            exit_threshold=0.5,
            stop_loss_threshold=4.0,
            max_position_size=0.25,
            min_holding_period=5,
            max_holding_period=100,
            enable_scaling=True
        )

        # Backtest defaults
        self._backtest_config = BacktestConfig(
            initial_capital=self.initial_capital,
            max_position_size=0.25,
            commission_rate=0.0001,
            spread_cost=0.0005,
            max_drawdown=0.15
        )

        # Risk limits
        self._risk_limits = RiskLimits(
            max_gross_exposure=1.0,
            max_net_exposure=0.5,
            max_drawdown=0.15,
            max_var_1d=0.02
        )

        logger.info("System components initialized")

    # ==================== DATA LOADING ====================

    def add_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        source: str = 'direct'
    ) -> 'QuantSystem':
        """
        Add market data for a symbol.

        Args:
            symbol: Asset ticker
            data: DataFrame with OHLCV data
            source: 'direct', 'csv', 'synthetic'
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data.set_index('timestamp')

        # Validate required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        data['symbol'] = symbol
        self._data[symbol] = data

        logger.info(f"Added {source} data for {symbol}: {len(data)} bars")
        return self

    def load_from_csv(
        self,
        symbol: str,
        filepath: str,
        **kwargs
    ) -> 'QuantSystem':
        """Load data from CSV file."""
        df = self._data_loader.load_csv(symbol, filepath)
        return self.add_data(symbol, df, source='csv')

    def generate_synthetic_data(
        self,
        symbols: List[str],
        n_points: int = 10000,
        correlated_pairs: Optional[List[Tuple[str, str, float]]] = None,
        seed: int = 42
    ) -> 'QuantSystem':
        """
        Generate synthetic market data for testing.

        Args:
            symbols: List of symbol names
            n_points: Number of data points per symbol
            correlated_pairs: List of (symbol_a, symbol_b, correlation) tuples
            seed: Random seed
        """
        np.random.seed(seed)

        # Generate base data for all symbols
        for symbol in symbols:
            if symbol not in self._data:
                data = self._data_loader.generate_synthetic_data(
                    symbol=symbol,
                    n_points=n_points,
                    seed=seed + hash(symbol) % 1000
                )
                self._data[symbol] = data

        # Generate correlated pairs
        if correlated_pairs:
            for sym_a, sym_b, corr in correlated_pairs:
                if sym_a in self._data and sym_b in self._data:
                    data_a, data_b = self._data_loader.generate_correlated_pair(
                        symbol_a=sym_a,
                        symbol_b=sym_b,
                        correlation=corr,
                        n_points=n_points,
                        seed=seed + hash(f"{sym_a}{sym_b}") % 1000
                    )
                    self._data[sym_a] = data_a
                    self._data[sym_b] = data_b
                    logger.info(f"Generated correlated pair {sym_a}/{sym_b} (ρ={corr:.2f})")

        return self

    # ==================== PAIR CONFIGURATION ====================

    def add_pair(
        self,
        asset_a: str,
        asset_b: str,
        hedge_ratio_method: str = 'rolling',
        zscore_window: int = 252,
        custom_threshold: Optional[float] = None
    ) -> 'QuantSystem':
        """Add a pair for spread trading."""
        pair = PairConfig(
            asset_a=asset_a,
            asset_b=asset_b,
            hedge_ratio_method=hedge_ratio_method,
            zscore_window=zscore_window,
            custom_threshold=custom_threshold
        )
        self._pairs.append(pair)
        logger.info(f"Added pair: {asset_a}/{asset_b}")
        return self

    def add_basket(
        self,
        target: str,
        basket: List[str],
        weights: Optional[List[float]] = None
    ) -> 'QuantSystem':
        """Add a basket spread (target vs weighted basket)."""
        if weights is None:
            weights = [1.0 / len(basket)] * len(basket)

        for asset, weight in zip(basket, weights):
            self.add_pair(target, asset)

        logger.info(f"Added basket: {target} vs {basket}")
        return self

    def remove_pair(self, asset_a: str, asset_b: str) -> 'QuantSystem':
        """Remove a pair from trading."""
        self._pairs = [
            p for p in self._pairs
            if not (p.asset_a == asset_a and p.asset_b == asset_b)
        ]
        return self

    def enable_pair(self, asset_a: str, asset_b: str, enabled: bool) -> 'QuantSystem':
        """Enable or disable a pair."""
        for pair in self._pairs:
            if pair.asset_a == asset_a and pair.asset_b == asset_b:
                pair.enabled = enabled
                break
        return self

    # ==================== MODEL INITIALIZATION ====================

    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize spread models for all configured pairs.
        Returns diagnostic information.
        """
        logger.info(f"Initializing models for {len(self._pairs)} pairs...")

        # Create spread model with config
        self._spread_model = SpreadModel(self._spread_config)

        # Create signal generator
        signal_config = SignalConfig()
        self._signal_generator = SignalGenerator(signal_config)

        # Create strategy
        self._strategy = MeanReversionStrategy(self._trade_config)

        # Create backtester
        self._backtester = BacktestEngine(self._backtest_config)

        # Create risk manager
        self._risk_manager = RiskManager(self._risk_limits)

        # Analyze each pair
        diagnostics = {}
        pair_configs = {f"{p.asset_a}_{p.asset_b}": p for p in self._pairs}

        for pair in self._pairs:
            if not pair.enabled:
                continue

            if pair.asset_a not in self._data or pair.asset_b not in self._data:
                logger.warning(f"Missing data for pair {pair.asset_a}/{pair.asset_b}")
                continue

            prices_a = self._data[pair.asset_a]['close']
            prices_b = self._data[pair.asset_b]['close']

            # Generate math report
            report = generate_math_report(
                prices_a=prices_a,
                prices_b=prices_b,
                pair_name=f"{pair.asset_a}/{pair.asset_b}"
            )

            diagnostics[f"{pair.asset_a}_{pair.asset_b}"] = report

            logger.info(
                f"{pair.asset_a}/{pair.asset_b}: "
                f"HR={report['hedge_ratio']:.4f}, "
                f"corr={report['correlation']:.3f}, "
                f"coint_p={report['cointegration']['p_value']:.4f}, "
                f"hurst={report['mean_reversion']['hurst_exponent']:.3f}"
            )

        self._model_diagnostics = diagnostics
        return diagnostics

    def get_pair_report(self, asset_a: str, asset_b: str) -> Optional[Dict]:
        """Get detailed mathematical analysis for a pair."""
        return self._model_diagnostics.get(f"{asset_a}_{asset_b}")

    # ==================== SIGNAL GENERATION ====================

    def generate_signals(self, timestamp: Optional[int] = None) -> List[TradingSignal]:
        """
        Generate trading signals for all pairs at current state.

        Returns:
            List of TradingSignal objects
        """
        if self._spread_model is None or self._signal_generator is None:
            raise RuntimeError("Models not initialized. Call initialize_models() first.")

        signals = []
        timestamp = timestamp or len(self._data[list(self._data.keys())[0]])

        for pair in self._pairs:
            if not pair.enabled:
                continue

            if pair.asset_a not in self._data or pair.asset_b not in self._data:
                continue

            prices_a = self._data[pair.asset_a]['close']
            prices_b = self._data[pair.asset_b]['close']

            # Update spread model
            state = self._spread_model.update_state(
                prices_a=prices_a,
                prices_b=prices_b,
                asset_a=pair.asset_a,
                asset_b=pair.asset_b,
                current_idx=timestamp
            )

            if not state.is_valid:
                continue

            # Get additional statistics
            returns_a = np.log(prices_a).diff()
            returns_b = np.log(prices_b).diff()
            spread = np.log(prices_a) - state.hedge_ratio * np.log(prices_b)

            # Generate signal
            signal = self._signal_generator.generate_signal(
                pair=(pair.asset_a, pair.asset_b),
                z_score=state.current_zscore,
                spread_value=state.current_spread,
                spread_std=state.spread_std,
                correlation=state.correlation,
                cointegration_pvalue=state.cointegration_pvalue,
                half_life=20,  # Would compute from OU params
                timestamp=timestamp,
                spread_kurtosis=spread.kurtosis(),
                momentum=state.current_zscore - (state.current_zscore * 0.9)  # Simplified
            )

            if signal:
                signals.append(signal)

        self._current_signals = signals
        return signals

    def get_signal_summary(self) -> pd.DataFrame:
        """Get summary of current signals as DataFrame."""
        if not hasattr(self, '_current_signals') or not self._current_signals:
            return pd.DataFrame()

        data = [s.to_dict() for s in self._current_signals]
        df = pd.DataFrame(data)

        if not df.empty:
            df = df.sort_values('confidence_score', ascending=False)

        return df

    # ==================== BACKTESTING ====================

    def run_backtest(
        self,
        progress_callback: Optional[callable] = None
    ) -> BacktestResult:
        """
        Run complete backtest simulation.

        Returns:
            BacktestResult with metrics, equity curve, and trades
        """
        if self._backtester is None or self._strategy is None:
            raise RuntimeError("Backtester not initialized")

        logger.info("Starting backtest...")

        # Prepare data for backtester
        pairs = [(p.asset_a, p.asset_b) for p in self._pairs if p.enabled]

        result = self._backtester.run(
            data=self._data,
            spread_model=self._spread_model,
            strategy=self._strategy,
            pairs=pairs,
            progress_callback=progress_callback
        )

        self._backtest_result = result

        # Log results
        self._log_backtest_results(result)

        return result

    def _log_backtest_results(self, result: BacktestResult) -> None:
        """Log backtest results."""
        logger.info("=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 60)

        metrics = result.metrics

        # Performance metrics
        logger.info(f"Total Return:    {metrics.get('total_return', 0):.2%}")
        logger.info(f"CAGR:            {metrics.get('cagr', 0):.2%}")
        logger.info(f"Volatility:      {metrics.get('volatility', 0):.2%}")
        logger.info(f"Sharpe Ratio:    {metrics.get('sharpe_ratio', 0):.3f}")
        logger.info(f"Sortino Ratio:   {metrics.get('sortino_ratio', 0):.3f}")

        # Risk metrics
        logger.info(f"Max Drawdown:    {metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"Avg Drawdown:    {metrics.get('avg_drawdown', 0):.2%}")

        # Trade metrics
        logger.info(f"Total Trades:    {metrics.get('total_trades', 0)}")
        logger.info(f"Win Rate:        {metrics.get('win_rate', 0):.2%}")
        logger.info(f"Profit Factor:   {metrics.get('profit_factor', 0):.2f}")
        logger.info(f"Avg Win:         ${metrics.get('avg_win', 0):,.2f}")
        logger.info(f"Avg Loss:        ${metrics.get('avg_loss', 0):,.2f}")

        logger.info("=" * 60)

    def export_backtest_results(self, filepath: Optional[str] = None) -> Path:
        """Export backtest results to files."""
        if self._backtest_result is None:
            raise RuntimeError("No backtest results to export")

        filepath = Path(filepath or self.output_dir / "backtest_results")
        filepath.mkdir(parents=True, exist_ok=True)

        # Export equity curve
        self._backtest_result.equity_curve.to_csv(
            filepath / 'equity_curve.csv',
            index=True
        )

        # Export drawdown
        self._backtest_result.drawdown_curve.to_csv(
            filepath / 'drawdown_curve.csv',
            index=True
        )

        # Export trades
        trades_data = [
            {
                'timestamp': t.timestamp,
                'pair': str(t.pair),
                'direction': t.direction,
                'size': t.size,
                'pnl': t.pnl,
                'commission': t.commission,
                'exit_reason': t.tags.get('exit_reason', '')
            }
            for t in self._backtest_result.trades
        ]
        trades_df = pd.DataFrame(trades_data)
        trades_df.to_csv(filepath / 'trades.csv', index=False)

        # Export metrics
        metrics_df = pd.DataFrame(
            list(self._backtest_result.metrics.items()),
            columns=['metric', 'value']
        )
        metrics_df.to_csv(filepath / 'metrics.csv', index=False)

        # Export summary JSON
        summary = {
            'timestamp': datetime.now().isoformat(),
            'metrics': self._backtest_result.metrics,
            'n_trades': len(self._backtest_result.trades),
            'n_pairs': len(self._pairs)
        }
        with open(filepath / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Results exported to {filepath}")
        return filepath

    # ==================== RLHF TRAINING ====================

    def initialize_rl_agent(self) -> 'QuantSystem':
        """Initialize RL agent for training."""
        if self._spread_model is None:
            raise RuntimeError("Models not initialized")

        # Prepare spread states for environment
        spread_states = {}
        for pair in self._pairs:
            if not pair.enabled:
                continue
            state = self._spread_model.get_state(pair.asset_a, pair.asset_b)
            if state:
                spread_states[(pair.asset_a, pair.asset_b)] = {
                    'z_score': state.current_zscore,
                    'spread_value': state.current_spread,
                    'hedge_ratio': state.hedge_ratio,
                    'correlation': state.correlation,
                    'prev_z_score': state.current_zscore
                }

        # Create environment
        env = TradingEnvironment(
            data=self._data,
            spread_states=spread_states,
            config=RLConfig(max_steps=10000),
            reward_config=RewardConfig()
        )

        # Initialize reward model
        self._reward_model = RewardModel(RewardWeights())

        # Initialize feedback integrator
        self._feedback_integrator = HumanFeedbackIntegrator(feedback_decay=0.95)

        # Create RL agent
        self._rl_agent = RLHFAgent(env, config=RLConfig())

        logger.info("RL agent initialized")
        return self

    def train_rl_agent(
        self,
        total_timesteps: int = 50000,
        save_path: Optional[str] = None,
        human_feedback: Optional[List[Dict]] = None
    ) -> Dict[str, float]:
        """
        Train RL agent with optional human feedback.

        Args:
            total_timesteps: Number of training steps
            save_path: Path to save trained model
            human_feedback: Optional list of feedback entries

        Returns:
            Training metrics
        """
        if self._rl_agent is None:
            self.initialize_rl_agent()

        logger.info(f"Training RL agent for {total_timesteps} timesteps...")

        # Add human feedback if provided
        if human_feedback:
            for fb in human_feedback:
                self._rl_agent.add_human_feedback(
                    trade_id=fb.get('trade_id', ''),
                    score=fb.get('score', 0),
                    comment=fb.get('comment')
                )

        # Train
        self._rl_agent.train(
            total_timesteps=total_timesteps,
            save_path=save_path
        )

        # Evaluate
        eval_metrics = self._rl_agent.evaluate(n_episodes=3)

        logger.info("RL Agent Evaluation:")
        for key, value in eval_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

        return eval_metrics

    def add_human_feedback(
        self,
        trade_id: str,
        score: float,
        comment: Optional[str] = None,
        features: Optional[Dict] = None
    ) -> 'QuantSystem':
        """
        Add human feedback for a trade.

        Args:
            trade_id: Unique trade identifier
            score: Feedback score (-1 to 1)
            comment: Optional comment
            features: Trade features for similarity matching
        """
        if self._feedback_integrator is None:
            self._feedback_integrator = HumanFeedbackIntegrator()

        self._feedback_integrator.add_feedback(
            feedback_type='trade_approval',
            target=trade_id,
            score=score,
            features=features or {},
            comment=comment
        )

        # Also add to RL agent if initialized
        if self._rl_agent:
            self._rl_agent.add_human_feedback(trade_id, score, comment)

        logger.info(f"Added feedback for {trade_id}: score={score}")
        return self

    def get_feedback_summary(self) -> Dict:
        """Get summary of human feedback."""
        if self._feedback_integrator is None:
            return {'count': 0}
        return self._feedback_integrator.get_feedback_statistics()

    # ==================== RISK MANAGEMENT ====================

    def check_risk_limits(self) -> Tuple[bool, List[str]]:
        """
        Check current portfolio against risk limits.

        Returns:
            (all_ok, list_of_breaches)
        """
        if self._risk_manager is None:
            return True, []

        # Get current positions from strategy
        positions = self._strategy.get_all_positions() if self._strategy else {}

        # Compute returns for VaR
        if hasattr(self, '_backtest_result') and self._backtest_result:
            returns = self._backtest_result.equity_curve.pct_change().dropna()
        else:
            returns = pd.Series([0.0])

        # Get current equity
        equity = self.initial_capital
        if hasattr(self, '_backtest_result') and self._backtest_result:
            equity = self._backtest_result.equity_curve.iloc[-1]

        # Compute risk metrics
        metrics = self._risk_manager.update_portfolio(
            positions=positions,
            returns=returns,
            equity=equity
        )

        # Check limits
        all_ok, breaches = self._risk_manager.check_limits(metrics)

        if not all_ok:
            for breach in breaches:
                logger.warning(f"Risk breach: {breach}")

        return all_ok, breaches

    def get_risk_report(self) -> str:
        """Generate human-readable risk report."""
        if self._risk_manager is None:
            return "Risk manager not initialized"

        positions = self._strategy.get_all_positions() if self._strategy else {}

        if hasattr(self, '_backtest_result') and self._backtest_result:
            returns = self._backtest_result.equity_curve.pct_change().dropna()
            equity = self._backtest_result.equity_curve.iloc[-1]
        else:
            returns = pd.Series([0.0])
            equity = self.initial_capital

        metrics = self._risk_manager.update_portfolio(
            positions=positions,
            returns=returns,
            equity=equity
        )

        return self._risk_manager.generate_risk_report(metrics)

    # ==================== STATE MANAGEMENT ====================

    def get_system_state(self) -> SystemState:
        """Get complete system state snapshot."""
        positions = self._strategy.get_all_positions() if self._strategy else {}
        signals = self._current_signals if hasattr(self, '_current_signals') else []
        spread_states = self._spread_model.get_all_states() if self._spread_model else {}

        # Risk metrics
        risk_metrics = {}
        if self._risk_manager:
            returns = self._backtest_result.equity_curve.pct_change().dropna() if hasattr(self, '_backtest_result') and self._backtest_result else pd.Series([0.0])
            equity = self._backtest_result.equity_curve.iloc[-1] if hasattr(self, '_backtest_result') and self._backtest_result else self.initial_capital
            metrics = self._risk_manager.update_portfolio(positions, returns, equity)
            risk_metrics = {
                'gross_exposure': metrics.gross_exposure,
                'net_exposure': metrics.net_exposure,
                'current_drawdown': metrics.current_drawdown,
                'var_1d': metrics.var_1d
            }

        state = SystemState(
            timestamp=len(self._data[list(self._data.keys())[0]]) if self._data else 0,
            capital=self.initial_capital,
            positions={str(k): v for k, v in positions.items()},
            signals=signals,
            spread_states={str(k): v for k, v in spread_states.items()},
            risk_metrics=risk_metrics,
            rl_agent_trained=self._rl_agent is not None and self._rl_agent.model is not None,
            backtest_complete=self._backtest_result is not None
        )

        self._last_state = state
        return state

    def export_state(self, filepath: Optional[str] = None) -> Path:
        """Export system state to JSON."""
        state = self.get_system_state()

        filepath = Path(filepath or self.output_dir / f"state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        state_dict = {
            'timestamp': state.timestamp,
            'capital': state.capital,
            'positions': state.positions,
            'signals': [s.to_dict() for s in state.signals],
            'spread_states': state.spread_states,
            'risk_metrics': state.risk_metrics,
            'rl_agent_trained': state.rl_agent_trained,
            'backtest_complete': state.backtest_complete
        }

        with open(filepath, 'w') as f:
            json.dump(state_dict, f, indent=2, default=str)

        logger.info(f"State exported to {filepath}")
        return filepath

    # ==================== CONVENIENCE METHODS ====================

    def run_complete_pipeline(
        self,
        symbols: List[str],
        pairs: List[Tuple[str, str]],
        n_points: int = 10000,
        train_rl: bool = True,
        rl_timesteps: int = 50000,
        export_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete system pipeline.

        Args:
            symbols: List of asset symbols
            pairs: List of (asset_a, asset_b) pairs to trade
            n_points: Data points for synthetic data
            train_rl: Whether to train RL agent
            rl_timesteps: RL training steps
            export_results: Whether to export results

        Returns:
            Dictionary with all results
        """
        logger.info("=" * 60)
        logger.info("RUNNING COMPLETE PIPELINE")
        logger.info("=" * 60)

        # Step 1: Load data
        logger.info("Step 1: Loading data...")
        self.generate_synthetic_data(
            symbols=symbols,
            n_points=n_points,
            seed=42
        )

        # Step 2: Configure pairs
        logger.info("Step 2: Configuring pairs...")
        for asset_a, asset_b in pairs:
            self.add_pair(asset_a, asset_b)

        # Step 3: Initialize models
        logger.info("Step 3: Initializing models...")
        diagnostics = self.initialize_models()

        # Step 4: Generate signals
        logger.info("Step 4: Generating signals...")
        signals = self.generate_signals()
        logger.info(f"Generated {len(signals)} signals")

        # Step 5: Run backtest
        logger.info("Step 5: Running backtest...")
        backtest_result = self.run_backtest()

        # Step 6: Train RL agent
        if train_rl:
            logger.info("Step 6: Training RL agent...")
            rl_metrics = self.train_rl_agent(total_timesteps=rl_timesteps)
        else:
            rl_metrics = {}

        # Step 7: Export results
        if export_results:
            logger.info("Step 7: Exporting results...")
            self.export_backtest_results()
            self.export_state()

        # Summary
        results = {
            'diagnostics': diagnostics,
            'backtest_metrics': backtest_result.metrics,
            'rl_metrics': rl_metrics,
            'n_signals': len(signals),
            'n_trades': len(backtest_result.trades)
        }

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)

        return results


def create_system(
    initial_capital: float = 1_000_000,
    data_dir: str = "data",
    model_dir: str = "models"
) -> QuantSystem:
    """Factory function to create QuantSystem instance."""
    return QuantSystem(
        initial_capital=initial_capital,
        data_dir=data_dir,
        model_dir=model_dir
    )
