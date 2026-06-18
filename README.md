# Quantitative Statistical Arbitrage System

A production-grade AI/ML system for relative value trading using spread models between correlated assets, enhanced with Reinforcement Learning from Human Feedback (RLHF).

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     QUANT TRADING TERMINAL                      │
├─────────────────────────────────────────────────────────────────┤
│  DATA LAYER  │  MODEL LAYER  │  STRATEGY LAYER  │  RLHF LAYER  │
│              │               │                  │              │
│  • Market    │  • Spread     │  • Signal        │  • PPO Agent │
│    Data      │    Models     │    Generation    │  • Reward    │
│  • Features  │  • Cointegration│  • Position    │    Modeling  │
│  • Storage   │  • Z-Score    │    Sizing        │  • Feedback  │
└─────────────────────────────────────────────────────────────────┘
         │                │               │               │
         └────────────────┴───────────────┴───────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
  ┌────────────┐  ┌────────────┐  ┌────────────┐
  │ BACKTEST   │  │ METRICS    │  │ DASHBOARD  │
  │ ENGINE     │  │ ENGINE     │  │ (Streamlit)│
  └────────────┘  └────────────┘  └────────────┘
```

## Directory Structure

```
quant_arb_system/
├── data/               # Data loading and feature engineering
│   ├── data_loader.py
│   └── __init__.py
├── models/             # Spread modeling and cointegration
│   ├── spread_model.py
│   └── __init__.py
├── strategies/         # Trading strategies
│   ├── mean_reversion.py
│   └── __init__.py
├── engine/             # Backtesting engine
│   ├── backtester.py
│   └── __init__.py
├── rlhf/               # RL agent with human feedback
│   ├── rl_agent.py
│   └── __init__.py
├── dashboard/          # Obsidian-style UI
│   ├── app.py
│   └── __init__.py
├── utils/              # Utilities
├── config/             # Configuration files
├── main.py             # Main entry point
└── requirements.txt    # Dependencies
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Backtest

```bash
python main.py
```

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```

## Core Components

### 1. Data Layer (`data/`)
- `DataLoader`: Market data ingestion, synthetic data generation
- `FeatureEngineer`: Technical indicators, returns, volatility

### 2. Model Layer (`models/`)
- `SpreadModel`: Hedge ratio estimation, cointegration testing
- `BasketSpreadModel`: Multi-asset basket spreads

### 3. Strategy Layer (`strategies/`)
- `MeanReversionStrategy`: Z-score based trading
- `AdaptiveMeanReversionStrategy`: Regime-adaptive thresholds

### 4. RLHF Layer (`rlhf/`)
- `TradingEnvironment`: Gym environment for spread trading
- `RLHFAgent`: PPO-based agent with human feedback

### 5. Backtest Engine (`engine/`)
- Event-driven simulation
- Realistic costs and slippage
- Risk limit enforcement

### 6. Dashboard (`dashboard/`)
- Live spread charts
- Z-score visualization
- Performance metrics
- Position monitoring

## Mathematical Framework

### Spread Construction
```
Spread = log(P_A) - β × log(P_B)
```

### Z-Score Normalization
```
Z = (Spread - μ) / σ
```

### Trading Signals
- **Enter Long**: Z < -2.0 (spread cheap)
- **Enter Short**: Z > +2.0 (spread rich)
- **Exit**: Z → 0 (mean reversion)

## Configuration

Edit `TradeConfig` in `strategies/mean_reversion.py`:

```python
TradeConfig(
    entry_threshold=2.0,       # Z-score to enter
    exit_threshold=0.5,        # Z-score to exit
    stop_loss_threshold=4.0,   # Stop loss level
    max_position_size=0.25     # 25% capital per position
)
```


