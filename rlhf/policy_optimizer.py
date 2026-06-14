"""
Policy Optimization Module
PPO-based policy optimizer with human feedback integration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PolicyConfig:
    """Configuration for policy network."""
    # Architecture
    hidden_sizes: Tuple[int, int] = (128, 64)
    activation: str = 'tanh'

    # PPO hyperparameters
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Training
    learning_rate: float = 3e-4
    n_epochs: int = 10
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Human feedback
    feedback_bonus_scale: float = 0.5
    preference_alignment_weight: float = 0.3


class PolicyNetwork:
    """
    Simple policy network implementation.
    In production, this would wrap stable-baselines3 or custom PyTorch.
    """

    def __init__(self, state_dim: int, action_dim: int, config: Optional[PolicyConfig] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or PolicyConfig()

        # Placeholder for actual network weights
        self._weights: Optional[np.ndarray] = None
        self._is_trained = False

    def predict(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, np.ndarray]:
        """
        Predict action given state.

        Returns: (action, action_probs)
        """
        if not self._is_trained:
            # Return random action before training
            action_probs = np.ones(self.action_dim) / self.action_dim
            if deterministic:
                return 0, action_probs
            return np.random.choice(self.action_dim, p=action_probs), action_probs

        # In production: forward pass through neural network
        # Here: placeholder that returns reasonable default
        action_probs = self._compute_action_probs(state)

        if deterministic:
            return np.argmax(action_probs), action_probs
        return np.random.choice(self.action_dim, p=action_probs), action_probs

    def _compute_action_probs(self, state: np.ndarray) -> np.ndarray:
        """Compute action probabilities based on state."""
        # Simplified heuristic for demonstration
        z_score = state[0] if len(state) > 0 else 0

        # Base probabilities
        probs = np.ones(self.action_dim) / self.action_dim

        # Adjust based on Z-score (mean reversion logic)
        if z_score < -2:
            probs[1] = 0.4  # Enter long
            probs[0] = 0.3  # Hold
        elif z_score > 2:
            probs[2] = 0.4  # Enter short
            probs[0] = 0.3  # Hold
        elif -0.5 < z_score < 0.5:
            probs[3] = 0.4  # Exit long
            probs[4] = 0.4  # Exit short
            probs[0] = 0.2  # Hold

        return probs / probs.sum()

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        old_log_probs: np.ndarray
    ) -> Dict[str, float]:
        """
        Update policy using PPO objective.

        Returns:
            Dictionary with loss components
        """
        # This is a placeholder - in production would use PyTorch/TensorFlow
        self._is_trained = True

        return {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.5,
            'clip_fraction': 0.0
        }

    def get_action_mask(self, state: np.ndarray) -> np.ndarray:
        """
        Get mask of valid actions for current state.
        Used for action masking in constrained environments.
        """
        mask = np.ones(self.action_dim, dtype=bool)

        # Example constraints
        z_score = abs(state[0]) if len(state) > 0 else 0

        if z_score < 1.5:
            # Don't enter positions when no clear signal
            mask[1] = False  # Can't enter long
            mask[2] = False  # Can't enter short

        return mask


class PPOTrainer:
    """
    Proximal Policy Optimization trainer with human feedback.
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        config: Optional[PolicyConfig] = None,
        human_feedback_callback: Optional[Callable] = None
    ):
        self.policy = policy
        self.config = config or PolicyConfig()
        self.human_feedback_callback = human_feedback_callback

        self._training_history: List[Dict] = []
        self._feedback_buffer: List[Dict] = []

    def train_epoch(
        self,
        rollout_buffer: Dict,
        human_feedback: Optional[List[Dict]] = None
    ) -> Dict[str, float]:
        """
        Train one epoch using PPO.

        Args:
            rollout_buffer: Dictionary with states, actions, rewards, etc.
            human_feedback: Optional list of human feedback entries

        Returns:
            Training metrics
        """
        states = rollout_buffer['states']
        actions = rollout_buffer['actions']
        rewards = rollout_buffer['rewards']
        dones = rollout_buffer['dones']

        # Compute advantages using GAE
        advantages, returns = self._compute_gae(
            rewards=rewards,
            values=rollout_buffer.get('values', np.zeros_like(rewards)),
            dones=dones
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Apply human feedback bonus if available
        if human_feedback:
            rewards = self._apply_feedback_bonus(rewards, human_feedback)

        # PPO update
        metrics = self.policy.update(
            states=states,
            actions=actions,
            advantages=advantages,
            returns=returns,
            old_log_probs=rollout_buffer.get('log_probs', np.zeros_like(actions))
        )

        # Record training history
        self._training_history.append({
            'epoch': len(self._training_history),
            **metrics
        })

        return metrics

    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        gamma: float = None,
        lam: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.

        GAE provides lower variance advantage estimates than simple Monte Carlo.
        """
        gamma = gamma or self.config.gamma
        lam = lam or self.config.gae_lambda

        n_steps = len(rewards)

        # Compute temporal difference errors
        td_errors = np.zeros(n_steps)
        for t in range(n_steps - 1):
            td_errors[t] = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]

        # Bootstrap last step
        td_errors[-1] = rewards[-1] - values[-1]

        # Compute advantages using cumulative product
        advantages = np.zeros(n_steps)
        advantages[-1] = td_errors[-1]

        for t in range(n_steps - 2, -1, -1):
            advantages[t] = td_errors[t] + gamma * lam * (1 - dones[t]) * advantages[t + 1]

        # Returns = advantages + values
        returns = advantages + values

        return advantages, returns

    def _apply_feedback_bonus(
        self,
        rewards: np.ndarray,
        feedback: List[Dict]
    ) -> np.ndarray:
        """
        Apply human feedback bonus to rewards.

        Aligns agent optimization with human preferences.
        """
        if not feedback:
            return rewards

        # Create feedback lookup by timestep
        feedback_map = {fb.get('timestep', 0): fb.get('score', 0) for fb in feedback}

        # Apply bonus
        bonus = np.zeros_like(rewards)
        for t in range(len(rewards)):
            if t in feedback_map:
                bonus[t] = feedback_map[t] * self.config.feedback_bonus_scale

        return rewards + bonus

    def add_human_feedback(self, feedback: Dict) -> None:
        """Add human feedback to buffer."""
        self._feedback_buffer.append(feedback)

    def clear_feedback_buffer(self) -> None:
        """Clear feedback buffer after applying."""
        self._feedback_buffer = []

    def get_training_metrics(self) -> Dict:
        """Get aggregated training metrics."""
        if not self._training_history:
            return {}

        metrics = {
            'epochs': len(self._training_history),
            'avg_policy_loss': np.mean([h['policy_loss'] for h in self._training_history]),
            'avg_value_loss': np.mean([h['value_loss'] for h in self._training_history]),
            'avg_entropy': np.mean([h['entropy'] for h in self._training_history])
        }

        return metrics


class HumanFeedbackIntegrator:
    """
    Integrates human feedback into RL training loop.
    Supports multiple feedback modalities.
    """

    def __init__(self, feedback_decay: float = 0.95):
        self.feedback_decay = feedback_decay
        self._feedback_history: List[Dict] = []
        self._preference_model: Optional[PreferenceModel] = None

    def add_feedback(
        self,
        feedback_type: str,
        target: str,
        score: float,
        features: Optional[Dict] = None,
        comment: Optional[str] = None
    ) -> None:
        """
        Add human feedback.

        Args:
            feedback_type: 'trade_approval', 'trajectory_preference', 'feature_importance'
            target: ID of trade or trajectory
            score: Feedback score (-1 to 1)
            features: Relevant features for this feedback
            comment: Optional human comment
        """
        feedback_entry = {
            'type': feedback_type,
            'target': target,
            'score': score,
            'features': features or {},
            'comment': comment,
            'timestamp': len(self._feedback_history)
        }

        self._feedback_history.append(feedback_entry)

        # Decay old feedback
        self._decay_feedback()

    def _decay_feedback(self) -> None:
        """Apply decay to old feedback weights."""
        for i, fb in enumerate(self._feedback_history):
            fb['weight'] = self.feedback_decay ** (len(self._feedback_history) - i)

    def get_feedback_for_trajectory(
        self,
        trajectory: List[Dict]
    ) -> List[Dict]:
        """Get relevant feedback for a trajectory."""
        if not self._feedback_history:
            return []

        # Find feedback for trades in this trajectory
        trade_ids = {step.get('trade_id') for step in trajectory if step.get('trade_id')}

        relevant_feedback = [
            fb for fb in self._feedback_history
            if fb['target'] in trade_ids
        ]

        return relevant_feedback

    def compute_alignment_reward(
        self,
        trajectory: List[Dict],
        preference_model: Optional[PreferenceModel] = None
    ) -> float:
        """
        Compute how aligned a trajectory is with human preferences.
        """
        if not self._feedback_history:
            return 0.0

        # Extract trajectory features
        traj_features = self._extract_trajectory_features(trajectory)

        # Find similar historical feedback
        total_score = 0.0
        total_weight = 0.0

        for fb in self._feedback_history[-20:]:  # Recent feedback
            similarity = self._compute_feature_similarity(traj_features, fb.get('features', {}))
            if similarity > 0.5:
                total_score += fb['score'] * fb.get('weight', 1.0) * similarity
                total_weight += similarity

        return total_score / total_weight if total_weight > 0 else 0.0

    def _extract_trajectory_features(self, trajectory: List[Dict]) -> Dict:
        """Extract summary features from trajectory."""
        if not trajectory:
            return {}

        pnls = [step.get('pnl', 0) for step in trajectory]
        zscores = [step.get('z_score', 0) for step in trajectory]

        return {
            'total_pnl': sum(pnls),
            'avg_pnl': np.mean(pnls) if pnls else 0,
            'pnl_std': np.std(pnls) if pnls else 0,
            'avg_zscore': np.mean(zscores) if zscores else 0,
            'max_zscore': max(abs(z) for z in zscores) if zscores else 0,
            'n_trades': len([s for s in trajectory if s.get('action_taken')])
        }

    def _compute_feature_similarity(self, features1: Dict, features2: Dict) -> float:
        """Compute similarity between feature vectors."""
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0

        similarities = []
        for key in common_keys:
            val1 = features1.get(key, 0)
            val2 = features2.get(key, 0)

            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                max_val = max(abs(val1), abs(val2), 1.0)
                similarities.append(1.0 - abs(val1 - val2) / max_val)

        return np.mean(similarities) if similarities else 0.0

    def get_feedback_statistics(self) -> Dict:
        """Get summary statistics of feedback."""
        if not self._feedback_history:
            return {'count': 0}

        scores = [fb['score'] for fb in self._feedback_history]

        return {
            'count': len(scores),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'by_type': self._count_by_type()
        }

    def _count_by_type(self) -> Dict[str, int]:
        """Count feedback by type."""
        counts = {}
        for fb in self._feedback_history:
            fb_type = fb.get('type', 'unknown')
            counts[fb_type] = counts.get(fb_type, 0) + 1
        return counts

    def export_feedback(self, filepath: str) -> None:
        """Export feedback history to JSON."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self._feedback_history, f, indent=2)

    def import_feedback(self, filepath: str) -> None:
        """Import feedback history from JSON."""
        import json
        with open(filepath, 'r') as f:
            self._feedback_history = json.load(f)
