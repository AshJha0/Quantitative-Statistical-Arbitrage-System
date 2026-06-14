"""
RLHF Agent Module
Reinforcement Learning with Human Feedback for strategy optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import logging

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Available actions for the RL agent."""
    HOLD = 0
    ENTER_LONG = 1
    ENTER_SHORT = 2
    EXIT_LONG = 3
    EXIT_SHORT = 4
    INCREASE_SIZE = 5
    DECREASE_SIZE = 6


@dataclass
class HumanFeedback:
    """Represents human feedback on a trade or strategy."""
    trade_id: str
    feedback_type: str  # 'trade_approval', 'strategy_preference', 'risk_adjustment'
    score: float  # -1 to 1 scale
    comment: Optional[str]
    timestamp: int
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""
    # PnL components
    pnl_weight: float = 1.0
    sharpe_weight: float = 0.5
    drawdown_penalty: float = 2.0
    turnover_penalty: float = 0.1

    # Risk adjustments
    max_drawdown_penalty: float = 5.0
    concentration_penalty: float = 0.5

    # Human feedback
    human_feedback_weight: float = 0.3

    # Time preferences
    holding_period_reward: float = 0.1  # Reward patience
    early_exit_penalty: float = 0.5


@dataclass
class RLConfig:
    """Configuration for RL training."""
    # Model architecture
    policy_type: str = 'MlpPolicy'
    hidden_layers: Tuple[int, int] = (128, 64)
    activation: str = 'tanh'

    # Training parameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # PPO specific
    clip_range: float = 0.2
    ent_coef: float = 0.01  # Entropy coefficient
    vf_coef: float = 0.5

    # Environment
    max_steps: int = 10000
    lookback_window: int = 60

    # Human feedback
    feedback_decay: float = 0.9  # How quickly feedback decays


class TradingEnvironment(gym.Env):
    """
    Custom Gym environment for spread trading.
    Simulates trading decisions with realistic constraints.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        spread_states: Dict[Tuple[str, str], Dict],
        config: Optional[RLConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        initial_capital: float = 1_000_000
    ):
        super().__init__()

        self.config = config or RLConfig()
        self.reward_config = reward_config or RewardConfig()
        self.initial_capital = initial_capital

        # Store data
        self.data = data
        self.spread_states = spread_states
        self.pairs = list(spread_states.keys())

        # State tracking
        self.current_step = 0
        self.capital = initial_capital
        self.positions: Dict[Tuple[str, str], Dict] = {}
        self.trade_history: List[Dict] = []
        self.equity_curve: List[float] = [initial_capital]

        # Feature dimensions
        n_features = self._get_feature_dim()
        n_pairs = len(self.pairs)

        # Action space: [action_type (7), size_modifier (3)]
        self.action_space = spaces.MultiDiscrete([7, 3])

        # Observation space: features for each pair + global state
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features * n_pairs + 5,),
            dtype=np.float32
        )

        # Human feedback buffer
        self._feedback_buffer: List[HumanFeedback] = []
        self._feedback_weights: Dict[str, float] = {}

    def _get_feature_dim(self) -> int:
        """Get number of features per pair."""
        return 12  # z_score, spread_value, volatility, momentum, etc.

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector from current state."""
        obs = []

        for pair in self.pairs:
            state = self.spread_states.get(pair, {})
            features = [
                state.get('z_score', 0),
                state.get('spread_value', 0),
                state.get('hedge_ratio', 0),
                state.get('correlation', 0),
                self._get_position_feature(pair),
                self._get_pnl_feature(pair),
                self._get_volatility_feature(pair),
                self._get_momentum_feature(pair),
                self._get_time_feature(),
                self._get_regime_feature(),
                self._get_human_feedback_feature(pair),
                self._get_risk_feature(pair)
            ]
            obs.extend(features)

        # Global features
        global_features = [
            self.capital / self.initial_capital,  # Normalized capital
            len(self.positions) / max(len(self.pairs), 1),  # Position utilization
            self._get_portfolio_sharpe(),
            self._get_portfolio_drawdown(),
            self._get_portfolio_turnover()
        ]
        obs.extend(global_features)

        return np.array(obs, dtype=np.float32)

    def _get_position_feature(self, pair: Tuple[str, str]) -> float:
        """Normalized position feature."""
        if pair in self.positions:
            pos = self.positions[pair]
            return pos.get('size', 0) * pos.get('direction', 0)
        return 0.0

    def _get_pnl_feature(self, pair: Tuple[str, str]) -> float:
        """Unrealized PnL for pair."""
        if pair in self.positions:
            return self.positions[pair].get('unrealized_pnl', 0) / 10000
        return 0.0

    def _get_volatility_feature(self, pair: Tuple[str, str]) -> float:
        """Recent volatility."""
        # Simplified - would use actual volatility in production
        return 0.0

    def _get_momentum_feature(self, pair: Tuple[str, str]) -> float:
        """Spread momentum."""
        state = self.spread_states.get(pair, {})
        return state.get('z_score', 0) - state.get('prev_z_score', 0)

    def _get_time_feature(self) -> float:
        """Time of day / session feature."""
        # Normalized time within episode
        return self.current_step / self.config.max_steps

    def _get_regime_feature(self) -> float:
        """Market regime indicator."""
        # Would use volatility clustering in production
        return 0.5

    def _get_human_feedback_feature(self, pair: Tuple[str, str]) -> float:
        """Aggregated human feedback for pair."""
        pair_key = f"{pair[0]}_{pair[1]}"
        return self._feedback_weights.get(pair_key, 0.0)

    def _get_risk_feature(self, pair: Tuple[str, str]) -> float:
        """Risk metric for pair."""
        state = self.spread_states.get(pair, {})
        return 1.0 / (1.0 + abs(state.get('z_score', 0)))

    def _get_portfolio_sharpe(self) -> float:
        """Calculate portfolio Sharpe ratio."""
        if len(self.equity_curve) < 10:
            return 0.0
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        if np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)

    def _get_portfolio_drawdown(self) -> float:
        """Calculate current drawdown."""
        peak = max(self.equity_curve)
        current = self.equity_curve[-1]
        return (peak - current) / peak

    def _get_portfolio_turnover(self) -> float:
        """Calculate turnover rate."""
        if len(self.trade_history) == 0:
            return 0.0
        return min(len(self.trade_history) / self.current_step, 1.0) if self.current_step > 0 else 0.0

    def step(self, action: np.ndarray):
        """
        Execute one step in the environment.
        action: [action_type, size_modifier]
        """
        self.current_step += 1

        # Parse action
        action_type = ActionType(action[0])
        size_mod = action[1]  # 0=decrease, 1=neutral, 2=increase

        # Execute action for each pair
        total_reward = 0.0
        info = {'actions': {}, 'rewards': {}}

        for pair in self.pairs:
            state = self.spread_states.get(pair, {})
            pair_reward, pair_info = self._execute_pair_action(
                pair, state, action_type, size_mod
            )
            total_reward += pair_reward
            info['actions'][pair] = pair_info

        # Add risk-adjusted component
        sharpe = self._get_portfolio_sharpe()
        drawdown = self._get_portfolio_drawdown()

        risk_bonus = self.reward_config.sharpe_weight * sharpe
        risk_penalty = self.reward_config.drawdown_penalty * drawdown

        total_reward += risk_bonus + risk_penalty

        # Add human feedback component
        feedback_bonus = self._compute_feedback_reward()
        total_reward += self.reward_config.human_feedback_weight * feedback_bonus

        # Update equity
        self.equity_curve.append(self.capital)

        # Check termination
        terminated = self.capital <= self.initial_capital * 0.5  # 50% drawdown limit
        truncated = self.current_step >= self.config.max_steps

        return self._get_observation(), total_reward, terminated, truncated, info

    def _execute_pair_action(
        self,
        pair: Tuple[str, str],
        state: Dict,
        action_type: ActionType,
        size_mod: int
    ) -> Tuple[float, Dict]:
        """Execute action for a single pair."""
        reward = 0.0
        info = {'action': action_type.name, 'executed': False}

        z_score = state.get('z_score', 0)
        current_position = self.positions.get(pair)

        # Entry logic
        if action_type == ActionType.ENTER_LONG and z_score < -2:
            if pair not in self.positions:
                self._open_position(pair, direction=1, size=0.5)
                reward += 0.1  # Small reward for valid entry
                info['executed'] = True

        elif action_type == ActionType.ENTER_SHORT and z_score > 2:
            if pair not in self.positions:
                self._open_position(pair, direction=-1, size=0.5)
                reward += 0.1
                info['executed'] = True

        # Exit logic
        elif action_type == ActionType.EXIT_LONG and current_position:
            if current_position.get('direction', 0) > 0:
                pnl = self._close_position(pair)
                reward += pnl * 0.01  # Scale PnL to reward
                info['executed'] = True
                info['pnl'] = pnl

        elif action_type == ActionType.EXIT_SHORT and current_position:
            if current_position.get('direction', 0) < 0:
                pnl = self._close_position(pair)
                reward += pnl * 0.01
                info['executed'] = True
                info['pnl'] = pnl

        # Size adjustments
        elif action_type == ActionType.INCREASE_SIZE and current_position:
            current_size = current_position.get('size', 0)
            if current_size < 1.0:
                current_position['size'] = min(current_size + 0.25, 1.0)
                info['executed'] = True

        elif action_type == ActionType.DECREASE_SIZE and current_position:
            current_size = current_position.get('size', 0)
            if current_size > 0.25:
                current_position['size'] = max(current_size - 0.25, 0.25)
                info['executed'] = True

        # Update unrealized PnL for open positions
        if pair in self.positions:
            pos = self.positions[pair]
            entry_z = pos.get('entry_zscore', z_score)
            direction = pos.get('direction', 0)
            size = pos.get('size', 0)

            # PnL based on Z-score movement
            pnl = (z_score - entry_z) * direction * size * 1000
            pos['unrealized_pnl'] = pnl
            self.capital += pnl * 0.001  # Mark-to-market

        return reward, info

    def _open_position(self, pair: Tuple[str, str], direction: int, size: float) -> None:
        """Open a new position."""
        state = self.spread_states.get(pair, {})
        self.positions[pair] = {
            'direction': direction,
            'size': size,
            'entry_zscore': state.get('z_score', 0),
            'entry_step': self.current_step,
            'unrealized_pnl': 0.0
        }

    def _close_position(self, pair: Tuple[str, str]) -> float:
        """Close a position and return realized PnL."""
        if pair not in self.positions:
            return 0.0

        pos = self.positions.pop(pair)
        pnl = pos.get('unrealized_pnl', 0)

        # Realize PnL
        self.capital += pnl
        self.trade_history.append({
            'pair': pair,
            'pnl': pnl,
            'step': self.current_step,
            'direction': pos['direction']
        })

        return pnl

    def _compute_feedback_reward(self) -> float:
        """Compute reward from human feedback."""
        if not self._feedback_buffer:
            return 0.0

        # Weight recent feedback more heavily
        total_weight = 0
        weighted_feedback = 0

        for feedback in self._feedback_buffer[-10:]:  # Last 10 feedback
            weight = self.config.feedback_decay ** (self.current_step - feedback.timestamp)
            weighted_feedback += feedback.score * weight
            total_weight += weight

        return weighted_feedback / total_weight if total_weight > 0 else 0.0

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)

        self.current_step = 0
        self.capital = self.initial_capital
        self.positions = {}
        self.trade_history = []
        self.equity_curve = [self.initial_capital]

        return self._get_observation(), {}

    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Capital: ${self.capital:,.2f}")
            print(f"Positions: {len(self.positions)}")
            print(f"Equity Curve: {self.equity_curve[-10:]}")

    def add_human_feedback(self, feedback: HumanFeedback) -> None:
        """Add human feedback to the buffer."""
        self._feedback_buffer.append(feedback)

        # Update feedback weights
        pair_key = f"{feedback.trade_id}"
        self._feedback_weights[pair_key] = feedback.score

    def get_trade_log(self) -> List[Dict]:
        """Get complete trade history."""
        return self.trade_history.copy()

    def get_equity_curve(self) -> List[float]:
        """Get equity curve."""
        return self.equity_curve.copy()


class RLHFAgent:
    """
    Main RL agent with human feedback integration.
    Trains PPO policy and incorporates human preferences.
    """

    def __init__(
        self,
        env: TradingEnvironment,
        config: Optional[RLConfig] = None,
        model_path: Optional[str] = None
    ):
        self.env = env
        self.config = config or RLConfig()
        self.model = None
        self.model_path = Path(model_path) if model_path else None

        # Human feedback storage
        self._feedback_history: List[HumanFeedback] = []

    def train(
        self,
        total_timesteps: int = 100000,
        callback: Optional[BaseCallback] = None,
        eval_freq: int = 10000,
        save_path: Optional[str] = None
    ) -> PPO:
        """
        Train the RL agent using PPO.
        """
        # Wrap environment
        vec_env = DummyVecEnv([lambda: self.env])

        # Initialize model
        self.model = PPO(
            policy=self.config.policy_type,
            env=vec_env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            verbose=1
        )

        # Train
        logger.info(f"Training for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

        # Save model
        if save_path:
            self.model.save(save_path)
            logger.info(f"Model saved to {save_path}")

        return self.model

    def evaluate(self, n_episodes: int = 3) -> Dict[str, float]:
        """Evaluate the trained agent."""
        if self.model is None:
            raise ValueError("No trained model available. Call train() first.")

        vec_env = DummyVecEnv([lambda: self.env])

        mean_reward, std_reward = evaluate_policy(
            self.model,
            vec_env,
            n_eval_episodes=n_episodes
        )

        # Get additional metrics
        equity_curve = self.env.get_equity_curve()
        trade_log = self.env.get_trade_log()

        # Calculate metrics
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            max_dd = self._calculate_max_drawdown(equity_curve)
            win_rate = len([t for t in trade_log if t['pnl'] > 0]) / len(trade_log) if trade_log else 0
        else:
            sharpe = max_dd = win_rate = 0

        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'total_trades': len(trade_log)
        }

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown."""
        peak = equity_curve[0]
        max_dd = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def add_human_feedback(
        self,
        trade_id: str,
        score: float,
        comment: Optional[str] = None,
        features: Optional[Dict] = None
    ) -> None:
        """
        Add human feedback for a specific trade.
        Score: -1 (very bad) to 1 (very good)
        """
        feedback = HumanFeedback(
            trade_id=trade_id,
            feedback_type='trade_approval',
            score=score,
            comment=comment,
            timestamp=self.env.current_step,
            features=features or {}
        )

        self._feedback_history.append(feedback)
        self.env.add_human_feedback(feedback)

        logger.info(f"Added feedback for {trade_id}: score={score}")

    def get_feedback_summary(self) -> Dict:
        """Get summary of human feedback."""
        if not self._feedback_history:
            return {'count': 0}

        scores = [f.score for f in self._feedback_history]
        return {
            'count': len(scores),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': min(scores),
            'max_score': max(scores)
        }

    def save(self, path: str) -> None:
        """Save model and feedback history."""
        if self.model:
            self.model.save(path)

        feedback_path = Path(path).parent / 'feedback_history.json'
        with open(feedback_path, 'w') as f:
            json.dump([
                {
                    'trade_id': fb.trade_id,
                    'score': fb.score,
                    'comment': fb.comment,
                    'timestamp': fb.timestamp
                }
                for fb in self._feedback_history
            ], f)

    def load(self, path: str) -> None:
        """Load model and feedback history."""
        if Path(path).exists():
            vec_env = DummyVecEnv([lambda: self.env])
            self.model = PPO.load(path, env=vec_env)

        feedback_path = Path(path).parent / 'feedback_history.json'
        if feedback_path.exists():
            with open(feedback_path, 'r') as f:
                data = json.load(f)
                self._feedback_history = [
                    HumanFeedback(**d) for d in data
                ]

    def get_action_recommendation(
        self,
        observation: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get action recommendation from trained model."""
        if self.model is None:
            raise ValueError("No trained model available.")

        action, _ = self.model.predict(observation, deterministic=True)
        return action, _
