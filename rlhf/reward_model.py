"""
Reward Model Module
Defines reward functions for RL agent training with human feedback integration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RewardWeights:
    """Weights for different reward components."""
    # PnL-related
    pnl_weight: float = 1.0
    sharpe_weight: float = 0.5
    sortino_weight: float = 0.3

    # Risk penalties
    drawdown_penalty: float = 2.0
    volatility_penalty: float = 0.5
    concentration_penalty: float = 0.3

    # Trading behavior
    turnover_penalty: float = 0.1
    holding_period_reward: float = 0.1
    timing_reward: float = 0.2

    # Human feedback
    human_feedback_weight: float = 0.4
    alignment_weight: float = 0.3


class RewardModel:
    """
    Computes reward signals for RL agent training.
    Combines financial metrics with human feedback alignment.
    """

    def __init__(self, weights: Optional[RewardWeights] = None):
        self.weights = weights or RewardWeights()
        self._feedback_history: List[Dict] = []

    def compute_reward(
        self,
        pnl: float,
        current_sharpe: float,
        current_drawdown: float,
        volatility: float,
        turnover: float,
        holding_period: int,
        human_feedback_score: Optional[float] = None,
        action_quality: Optional[float] = None
    ) -> float:
        """
        Compute composite reward from multiple signals.

        Args:
            pnl: Period PnL (scaled)
            current_sharpe: Rolling Sharpe ratio
            current_drawdown: Current drawdown (positive value)
            volatility: Realized volatility
            turnover: Portfolio turnover rate
            holding_period: Average holding period in bars
            human_feedback_score: Human feedback (-1 to 1), optional
            action_quality: Quality of timing/execution (0 to 1), optional

        Returns:
            Scalar reward value
        """
        rewards = []
        penalties = []

        # PnL reward (scaled)
        pnl_reward = pnl * self.weights.pnl_weight
        rewards.append(pnl_reward)

        # Risk-adjusted return bonus
        sharpe_bonus = max(0, current_sharpe) * self.weights.sharpe_weight
        rewards.append(sharpe_bonus)

        # Drawdown penalty (convex penalty for large drawdowns)
        dd_penalty = (current_drawdown ** 1.5) * self.weights.drawdown_penalty
        penalties.append(dd_penalty)

        # Volatility penalty
        vol_penalty = max(0, volatility - 0.02) * self.weights.volatility_penalty
        penalties.append(vol_penalty)

        # Turnover penalty (discourage excessive trading)
        turn_penalty = turnover * self.weights.turnover_penalty
        penalties.append(turn_penalty)

        # Holding period reward (encourage patience, but not too long)
        if 5 <= holding_period <= 50:
            hp_reward = self.weights.holding_period_reward
        elif holding_period < 5:
            hp_reward = self.weights.holding_period_reward * 0.5  # Penalize too fast
        else:
            hp_reward = self.weights.holding_period_reward * 0.8  # Slight penalty for too slow
        rewards.append(hp_reward)

        # Action quality reward (timing)
        if action_quality is not None:
            timing_reward = action_quality * self.weights.timing_reward
            rewards.append(timing_reward)

        # Human feedback component
        if human_feedback_score is not None:
            hf_reward = human_feedback_score * self.weights.human_feedback_weight
            rewards.append(hf_reward)

        # Compute final reward
        total_reward = sum(rewards) - sum(penalties)

        return total_reward

    def compute_sparse_reward(
        self,
        trade_pnl: float,
        trade_direction: int,
        entry_zscore: float,
        exit_zscore: float,
        bars_held: int,
        max_adverse_zscore: float
    ) -> float:
        """
        Compute sparse reward at trade completion.

        Rewards good entry timing and successful mean reversion capture.
        """
        rewards = []

        # Base PnL reward
        pnl_reward = trade_pnl * 0.01  # Scale down
        rewards.append(pnl_reward)

        # Entry quality: entered at extreme Z-score
        entry_quality = min(abs(entry_zscore) / 3.0, 1.0)
        rewards.append(entry_quality * 0.1)

        # Exit quality: exited near mean
        exit_quality = 1.0 - min(abs(exit_zscore) / 1.0, 1.0)
        rewards.append(exit_quality * 0.1)

        # Mean reversion capture
        if trade_direction == 1:  # Long spread
            z_improvement = entry_zscore - exit_zscore
        else:  # Short spread
            z_improvement = exit_zscore - entry_zscore

        capture_reward = min(max(z_improvement, 0) / 4.0, 1.0) * 0.2
        rewards.append(capture_reward)

        # Patience bonus (but not too long)
        if 10 <= bars_held <= 50:
            rewards.append(0.1)
        elif bars_held < 5:
            rewards.append(-0.05)  # Penalize impatience

        # Adverse selection penalty (how bad did it get?)
        if max_adverse_zscore > abs(entry_zscore):
            adverse_penalty = 0.1 * min((max_adverse_zscore - abs(entry_zscore)) / 2.0, 1.0)
            rewards.append(-adverse_penalty)

        return sum(rewards)

    def add_human_feedback(
        self,
        trade_id: str,
        score: float,
        features: Dict,
        timestamp: int
    ) -> None:
        """Record human feedback for a trade."""
        self._feedback_history.append({
            'trade_id': trade_id,
            'score': score,
            'features': features,
            'timestamp': timestamp
        })

    def get_feedback_weight(self, trade_features: Dict) -> float:
        """
        Get human feedback weight for similar trades.
        Uses nearest neighbor matching on trade features.
        """
        if not self._feedback_history:
            return 0.0

        # Find similar historical trades
        similar_feedbacks = []

        for fb in self._feedback_history[-50:]:  # Last 50 feedback
            similarity = self._compute_similarity(trade_features, fb['features'])
            if similarity > 0.7:  # Similar trades
                similar_feedbacks.append((fb['score'], similarity))

        if not similar_feedbacks:
            return 0.0

        # Weighted average of similar feedback
        total_weight = sum(sim for _, sim in similar_feedbacks)
        weighted_score = sum(score * sim for score, sim in similar_feedbacks)

        return weighted_score / total_weight if total_weight > 0 else 0.0

    def _compute_similarity(self, features1: Dict, features2: Dict) -> float:
        """Compute similarity between two trade feature vectors."""
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0

        similarities = []
        for key in common_keys:
            val1, val2 = features1[key], features2[key]
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Normalize and compute similarity
                max_val = max(abs(val1), abs(val2), 1.0)
                sim = 1.0 - abs(val1 - val2) / max_val
                similarities.append(sim)

        return np.mean(similarities) if similarities else 0.0

    def shape_reward(
        self,
        base_reward: float,
        state_features: Dict,
        action: int,
        next_state_features: Dict
    ) -> float:
        """
        Apply reward shaping to guide learning.

        Adds potential-based reward shaping:
        F(s, a, s') = F(s, a, s') + γ*Φ(s') - Φ(s)

        Where Φ is a potential function based on state quality.
        """
        # Potential function: based on Z-score extremeness and position
        current_potential = self._state_potential(state_features)
        next_potential = self._state_potential(next_state_features)

        # Discounted potential difference
        gamma = 0.99
        shaped_reward = base_reward + gamma * next_potential - current_potential

        return shaped_reward

    def _state_potential(self, state_features: Dict) -> float:
        """
        Compute potential function for reward shaping.

        Higher potential for states with:
        - Extreme Z-scores (opportunity)
        - Appropriate position (aligned with mean reversion)
        - Low risk
        """
        z_score = state_features.get('z_score', 0)
        position = state_features.get('position', 0)
        correlation = state_features.get('correlation', 0.5)

        # Opportunity potential: extreme Z-scores are good opportunities
        opportunity = abs(z_score) / 3.0

        # Alignment potential: position aligned with mean reversion
        if position > 0 and z_score < 0:
            alignment = 1.0  # Long cheap spread
        elif position < 0 and z_score > 0:
            alignment = 1.0  # Short rich spread
        elif position == 0:
            alignment = 0.5  # Neutral is okay
        else:
            alignment = 0.0  # Wrong direction

        # Quality potential: high correlation pairs are better
        quality = correlation

        potential = 0.4 * opportunity + 0.4 * alignment + 0.2 * quality
        return potential


class PreferenceModel:
    """
    Learns human preferences over trajectories.
    Used for RLHF alignment.
    """

    def __init__(self, feature_dim: int = 10):
        self.feature_dim = feature_dim
        self._preferences: List[Tuple[Dict, Dict, int]] = []  # (traj_a, traj_b, preferred)
        self._preference_weights: Optional[np.ndarray] = None

    def record_preference(
        self,
        trajectory_a: Dict,
        trajectory_b: Dict,
        preferred: int  # 1 for A, 2 for B
    ) -> None:
        """Record a human preference between two trajectories."""
        self._preferences.append((trajectory_a, trajectory_b, preferred))

    def learn_preferences(self, n_iterations: int = 100, learning_rate: float = 0.01) -> None:
        """
        Learn preference weights via gradient descent.

        Uses Bradley-Terry model for pairwise preferences.
        """
        if not self._preferences:
            return

        # Initialize weights
        self._preference_weights = np.zeros(self.feature_dim)

        for iteration in range(n_iterations):
            total_gradient = np.zeros(self.feature_dim)

            for traj_a, traj_b, preferred in self._preferences:
                # Extract features
                features_a = self._extract_features(traj_a)
                features_b = self._extract_features(traj_b)

                # Compute preference probability (Bradley-Terry)
                score_a = np.dot(self._preference_weights, features_a)
                score_b = np.dot(self._preference_weights, features_b)

                prob_a = 1.0 / (1.0 + np.exp(score_b - score_a))

                # Gradient
                if preferred == 1:
                    gradient = features_a * (1 - prob_a) - features_b * prob_a
                else:
                    gradient = features_b * (1 - prob_b := 1.0 - prob_a) - features_a * prob_b
                    gradient = features_b * prob_a - features_a * (1 - prob_a)

                total_gradient += gradient

            # Update weights
            self._preference_weights += learning_rate * total_gradient / len(self._preferences)

    def _extract_features(self, trajectory: Dict) -> np.ndarray:
        """Extract feature vector from trajectory."""
        features = [
            trajectory.get('total_pnl', 0),
            trajectory.get('sharpe', 0),
            trajectory.get('max_drawdown', 0),
            trajectory.get('win_rate', 0),
            trajectory.get('avg_holding_period', 0),
            trajectory.get('turnover', 0),
            trajectory.get('avg_entry_zscore', 0),
            trajectory.get('avg_exit_zscore', 0),
            trajectory.get('risk_adjusted_return', 0),
            trajectory.get('consistency', 0)
        ]
        return np.array(features[:self.feature_dim])

    def get_preferred_action(self, features_a: np.ndarray, features_b: np.ndarray) -> Tuple[int, float]:
        """
        Predict which action humans would prefer.

        Returns: (preferred_index, confidence)
        """
        if self._preference_weights is None:
            return (0, 0.5)  # No preference learned

        score_a = np.dot(self._preference_weights, features_a)
        score_b = np.dot(self._preference_weights, features_b)

        if score_a > score_b:
            confidence = 1.0 / (1.0 + np.exp(score_b - score_a))
            return (0, confidence)
        else:
            confidence = 1.0 / (1.0 + np.exp(score_a - score_b))
            return (1, confidence)

    def get_alignment_score(self, trajectory: Dict) -> float:
        """
        Compute how aligned a trajectory is with learned preferences.
        """
        if self._preference_weights is None:
            return 0.5

        features = self._extract_features(trajectory)
        score = np.dot(self._preference_weights, features)

        # Normalize to 0-1
        return 1.0 / (1.0 + np.exp(-score))


class InverseReinforcementLearning:
    """
    Infers reward function from expert demonstrations.
    """

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._expert_trajectories: List[List[Dict]] = []
        self._reward_weights: Optional[np.ndarray] = None

    def add_expert_trajectory(self, trajectory: List[Dict]) -> None:
        """Add an expert demonstration trajectory."""
        self._expert_trajectories.append(trajectory)

    def learn_reward(
        self,
        agent_trajectories: List[List[Dict]],
        n_iterations: int = 50,
        learning_rate: float = 0.01
    ) -> np.ndarray:
        """
        Learn reward weights via feature matching.

        The learned reward should make agent behavior match expert behavior.
        """
        # Define feature expectations
        expert_features = self._compute_expected_features(self._expert_trajectories)

        # Initialize reward weights
        self._reward_weights = np.zeros(10)  # 10 reward features

        for iteration in range(n_iterations):
            # Compute agent's expected features
            agent_features = self._compute_expected_features(agent_trajectories)

            # Gradient: difference between expert and agent
            gradient = expert_features - agent_features

            # Update weights
            self._reward_weights += learning_rate * gradient

            # Generate new agent trajectories with updated reward (simulated)
            # In practice, this would require re-running the agent

        return self._reward_weights

    def _compute_expected_features(self, trajectories: List[List[Dict]]) -> np.ndarray:
        """Compute expected feature counts from trajectories."""
        if not trajectories:
            return np.zeros(10)

        all_features = []
        for trajectory in trajectories:
            for step in trajectory:
                features = np.array([
                    step.get('pnl', 0),
                    step.get('sharpe_contribution', 0),
                    step.get('risk_taken', 0),
                    step.get('timing_quality', 0),
                    step.get('entry_quality', 0),
                    step.get('exit_quality', 0),
                    step.get('holding_period_score', 0),
                    step.get('turnover_cost', 0),
                    step.get('consistency', 0),
                    step.get('capital_efficiency', 0)
                ])
                all_features.append(features)

        return np.mean(all_features, axis=0) if all_features else np.zeros(10)

    def get_reward(self, state: np.ndarray, action: int, next_state: np.ndarray) -> float:
        """Compute learned reward for a transition."""
        if self._reward_weights is None:
            return 0.0

        # Feature vector from transition
        features = self._transition_features(state, action, next_state)
        return np.dot(self._reward_weights, features)

    def _transition_features(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray
    ) -> np.ndarray:
        """Extract features from a state transition."""
        # Simplified feature extraction
        return np.array([
            next_state[0] - state[0],  # PnL change
            0,  # Sharpe contribution (computed over window)
            np.abs(action),  # Risk taken
            0,  # Timing quality
            abs(state[1]) if action > 0 else 0,  # Entry quality
            0,  # Exit quality
            0,  # Holding period
            1 if action != 0 else 0,  # Turnover cost
            0,  # Consistency
            0   # Capital efficiency
        ])
