# STEP 3: RLHF Design (Reinforcement Learning with Human Feedback)

This document details the RLHF architecture for optimizing the statistical arbitrage strategy.

---

## 1. Overview

The RLHF system enhances the mean reversion strategy by learning optimal trading policies through:
1. **Reinforcement Learning**: PPO agent learns from trading rewards
2. **Human Feedback**: Trader preferences guide policy alignment
3. **Reward Modeling**: Learned rewards capture complex trading objectives

```
┌─────────────────────────────────────────────────────────────────┐
│                      RLHF ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   AGENT      │    │  ENVIRONMENT │    │   REWARD     │      │
│  │  (PPO)       │───▶│  (Trading)   │───▶│   MODEL      │      │
│  │              │◀───│              │◀───│              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         ▲                                       │               │
│         │                                       │               │
│         │         ┌──────────────┐              │               │
│         └─────────│   HUMAN      │◀─────────────┘               │
│                   │   FEEDBACK   │                              │
│                   └──────────────┘                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Environment Design

### 2.1 Observation Space

The trading environment provides the following features to the agent:

| Feature | Dimension | Description |
|---------|-----------|-------------|
| Z-score | 1 | Current normalized spread |
| Spread value | 1 | Raw spread value |
| Hedge ratio | 1 | Current β estimate |
| Correlation | 1 | Asset correlation |
| Position | 1 | Current position (-1, 0, 1) |
| Unrealized PnL | 1 | Current position PnL |
| Volatility | 1 | Recent spread volatility |
| Momentum | 1 | Spread momentum |
| Time | 1 | Normalized time in episode |
| Regime | 1 | Market regime indicator |
| Human feedback | 1 | Aggregated feedback score |
| Risk metric | 1 | Position risk level |
| **Global features** | | |
| Normalized capital | 1 | Capital / initial |
| Position utilization | 1 | Active positions / max |
| Portfolio Sharpe | 1 | Rolling Sharpe ratio |
| Portfolio drawdown | 1 | Current drawdown |
| Portfolio turnover | 1 | Trading frequency |

**Total dimension**: 12 × n_pairs + 5

### 2.2 Action Space

Multi-discrete action space:

```python
action_space = MultiDiscrete([7, 3])

# Action type (0-6):
0: HOLD
1: ENTER_LONG
2: ENTER_SHORT
3: EXIT_LONG
4: EXIT_SHORT
5: INCREASE_SIZE
6: DECREASE_SIZE

# Size modifier (0-2):
0: Decrease
1: Neutral
2: Increase
```

### 2.3 Action Masking

Not all actions are valid in all states:

| State Condition | Invalid Actions |
|-----------------|-----------------|
| \|Z\| < 1.5 | ENTER_LONG, ENTER_SHORT |
| No position | EXIT_LONG, EXIT_SHORT |
| Max exposure | ENTER_LONG, ENTER_SHORT |
| Min size | DECREASE_SIZE |

---

## 3. Reward Design

### 3.1 Dense Reward Components

**PnL Reward:**
$$r_{pnl} = w_{pnl} \cdot \Delta \text{PnL}$$

**Sharpe Bonus:**
$$r_{sharpe} = w_{sharpe} \cdot \max(0, \text{Sharpe})$$

**Drawdown Penalty:**
$$r_{dd} = -w_{dd} \cdot (\text{Drawdown})^{1.5}$$

Convex penalty discourages large drawdowns.

**Volatility Penalty:**
$$r_{vol} = -w_{vol} \cdot \max(0, \sigma - \sigma_{target})$$

**Turnover Penalty:**
$$r_{turn} = -w_{turn} \cdot \text{Turnover}$$

Discourages excessive trading.

**Holding Period Reward:**
$$
r_{hp} = \begin{cases}
0.5 \cdot w_{hp} & \text{if } hp < 5 \text{ (too fast)} \\
w_{hp} & \text{if } 5 \leq hp \leq 50 \\
0.8 \cdot w_{hp} & \text{if } hp > 50 \text{ (too slow)}
\end{cases}
$$

### 3.2 Sparse Trade-Level Rewards

Computed at trade completion:

**Entry Quality:**
$$r_{entry} = 0.1 \cdot \min\left(\frac{|Z_{entry}|}{3}, 1\right)$$

**Exit Quality:**
$$r_{exit} = 0.1 \cdot \left(1 - \min\left(\frac{|Z_{exit}|}{1}, 1\right)\right)$$

**Mean Reversion Capture:**
$$r_{capture} = 0.2 \cdot \min\left(\frac{Z_{improvement}}{4}, 1\right)$$

Where $Z_{improvement} = Z_{entry} - Z_{exit}$ for long positions.

### 3.3 Human Feedback Reward

$$r_{hf} = w_{hf} \cdot \text{feedback\_score}$$

Feedback scores are interpolated from similar historical trades.

### 3.4 Reward Shaping

Potential-based reward shaping for faster learning:

$$F(s, a, s') = r(s, a, s') + \gamma \Phi(s') - \Phi(s)$$

Where the potential function is:

$$\Phi(s) = 0.4 \cdot \text{Opportunity}(s) + 0.4 \cdot \text{Alignment}(s) + 0.2 \cdot \text{Quality}(s)$$

---

## 4. Human Feedback Integration

### 4.1 Feedback Types

| Type | Target | Scale | Usage |
|------|--------|-------|-------|
| trade_approval | Individual trade | [-1, 1] | Direct reward bonus |
| trajectory_preference | Pair of trajectories | A or B preferred | Preference learning |
| feature_importance | Strategy feature | [0, 1] | Reward weight adjustment |

### 4.2 Feedback Processing

**Step 1: Collection**
```python
feedback = {
    'type': 'trade_approval',
    'trade_id': 'XAUUSD_XAGUSD_12345',
    'score': 0.8,  # Positive feedback
    'features': {
        'entry_zscore': -2.5,
        'exit_zscore': 0.3,
        'pnl': 1500,
        'holding_period': 15
    },
    'comment': 'Good entry timing'
}
```

**Step 2: Storage with Decay**
$$w_t = \text{decay}^{T - t}$$

Recent feedback weighted more heavily.

**Step 3: Interpolation**
For new trade with features $f$, find similar historical feedback:
$$\text{feedback\_weight} = \frac{\sum_i \text{score}_i \cdot \text{similarity}(f, f_i)}{\sum_i \text{similarity}(f, f_i)}$$

### 4.3 Preference Learning (Bradley-Terry Model)

For pairwise trajectory preferences:

$$P(A \succ B) = \frac{e^{w^T \phi(A)}}{e^{w^T \phi(A)} + e^{w^T \phi(B)}}$$

Where $\phi(\cdot)$ extracts trajectory features.

**Gradient for weight update:**
$$\nabla_w \log P(A \succ B) = \phi(A) - \phi(B) \cdot P(B \succ A)$$

---

## 5. PPO Training Algorithm

### 5.1 Objective Function

$$L^{PPO}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]$$

Where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$
- $\hat{A}_t$ is the advantage estimate

### 5.2 Generalized Advantage Estimation (GAE)

$$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

Where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

### 5.3 Training Loop

```
Algorithm: PPO with Human Feedback
────────────────────────────────────
1: Initialize policy π_θ and value function V_φ
2: for iteration = 1, 2, ... do
3:     Run policy in environment → collect trajectories
4:     Compute rewards r_t
5:     
6:     # Human feedback integration
7:     if human_feedback available then
8:         r_t ← r_t + w_hf · feedback_score(s_t, a_t)
9:     end if
10:    
11:    Compute advantages using GAE
12:    Normalize advantages
13:    
14:    # PPO update
15:    for epoch = 1 to n_epochs do
16:        Shuffle minibatches
17:        for each minibatch do
18:            Compute PPO loss L^PPO
19:            Compute value loss L^VF
20:            Compute entropy bonus
21:            Update θ, φ via SGD
22:        end for
23:    end for
24:    
25:    # Update preference model if new feedback
26:    if new_preferences then
27:        Update preference weights w_pref
28:    end if
29: end for
```

---

## 6. Architecture Summary

### 6.1 Component Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `RewardModel` | reward_model.py | Computes composite rewards |
| `PreferenceModel` | reward_model.py | Learns trajectory preferences |
| `InverseReinforcementLearning` | reward_model.py | Infers rewards from demonstrations |
| `PolicyNetwork` | policy_optimizer.py | Neural policy approximator |
| `PPOTrainer` | policy_optimizer.py | PPO optimization loop |
| `HumanFeedbackIntegrator` | policy_optimizer.py | Feedback collection and processing |
| `TradingEnvironment` | rl_agent.py | Gym environment for trading |
| `RLHFAgent` | rl_agent.py | Main agent orchestrator |

### 6.2 Data Flow

```
1. Environment provides state s_t
       ↓
2. Policy predicts action a_t ~ π(a|s)
       ↓
3. Environment executes action → new state s_{t+1}, reward r_t
       ↓
4. Human feedback collected → feedback_score
       ↓
5. Reward model computes: r'_t = r_t + w_hf · feedback_score
       ↓
6. Trajectory stored in buffer
       ↓
7. PPO update on buffer → policy improvement
```

---

## 7. Configuration

### 7.1 RL Hyperparameters

```python
RLConfig(
    policy_type='MlpPolicy',
    hidden_layers=(128, 64),
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5
)
```

### 7.2 Reward Weights

```python
RewardWeights(
    pnl_weight=1.0,
    sharpe_weight=0.5,
    drawdown_penalty=2.0,
    volatility_penalty=0.5,
    turnover_penalty=0.1,
    holding_period_reward=0.1,
    human_feedback_weight=0.4,
    alignment_weight=0.3
)
```

### 7.3 Feedback Settings

```python
feedback_decay=0.95      # Feedback half-life ~13 steps
feedback_bonus_scale=0.5 # Max bonus from feedback
preference_alignment_weight=0.3
```

---

## 8. Usage Example

```python
from rlhf import RLHFAgent, TradingEnvironment, RLConfig

# Create environment
env = TradingEnvironment(
    data=market_data,
    spread_states=spread_states,
    config=RLConfig()
)

# Initialize agent
agent = RLHFAgent(env)

# Train
agent.train(total_timesteps=100000)

# Add human feedback
agent.add_human_feedback(
    trade_id='trade_001',
    score=0.8,
    comment='Good entry, early exit'
)

# Continue training with feedback
agent.train(total_timesteps=50000)

# Evaluate
metrics = agent.evaluate(n_episodes=10)
```

---

## 9. Expected Behavior

After training with RLHF:

1. **Entry Timing**: Agent learns to enter at more extreme Z-scores
2. **Exit Timing**: Agent holds positions longer for full mean reversion
3. **Position Sizing**: Agent sizes up on high-confidence signals
4. **Risk Management**: Agent avoids trades during high volatility
5. **Human Alignment**: Agent behavior matches trader preferences

---

This RLHF design provides a principled approach to combining reinforcement learning with human expertise for optimal trading policy learning.
