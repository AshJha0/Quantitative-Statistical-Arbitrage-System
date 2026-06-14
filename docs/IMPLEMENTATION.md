# STEP 4: Full Implementation

This document describes the complete production-ready implementation of the statistical arbitrage system.

---

## 1. System Integration Overview

The full system integrates all components through the `QuantSystem` orchestrator class:

```
┌─────────────────────────────────────────────────────────────────┐
│                     QuantSystem Orchestrator                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │    Data     │  │   Models    │  │  Strategy   │             │
│  │   Loader    │──│   Spread    │──│   Mean Rev  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│         │                │                │                     │
│         ▼                ▼                ▼                     │
│  ┌─────────────────────────────────────────────────┐           │
│  │              Backtest Engine                     │           │
│  └─────────────────────────────────────────────────┘           │
│         │                │                │                     │
│         ▼                ▼                ▼                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │    Risk     │  │    RLHF     │  │  Dashboard  │             │
│  │   Manager   │  │   Agent     │  │   (Streamlit)│            │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Implementation Files

### 2.1 System Orchestrator (`system.py`)

The `QuantSystem` class provides the unified API:

```python
# Create system
system = QuantSystem(
    data_dir="data",
    model_dir="models",
    initial_capital=1_000_000
)

# Add data
system.add_data("XAUUSD", price_data)
system.add_data("XAGUSD", price_data)

# Configure pairs
system.add_pair("XAUUSD", "XAGUSD")

# Initialize models
diagnostics = system.initialize_models()

# Generate signals
signals = system.generate_signals()

# Run backtest
results = system.run_backtest()

# Train RL agent
rl_metrics = system.train_rl_agent(total_timesteps=50000)

# Add human feedback
system.add_human_feedback(
    trade_id="trade_001",
    score=0.8,
    comment="Good entry timing"
)

# Export results
system.export_backtest_results()
```

### 2.2 Main Entry Point (`main.py`)

Command-line interface for running the system:

```bash
# Run complete demo
python main.py

# Skip RL training
python main.py --no-rl

# Custom RL timesteps
python main.py --timesteps 100000

# Verbose logging
python main.py --verbose
```

### 2.3 Dashboard (`dashboard/app.py`)

Streamlit-based UI for monitoring:

```bash
streamlit run dashboard/app.py
```

---

## 3. Module Dependencies

```
system.py
├── data/data_loader.py
│   ├── DataLoader
│   └── FeatureEngineer
├── models/
│   ├── spread_model.py
│   ├── signal_generator.py
│   └── math_utils.py
├── strategies/mean_reversion.py
├── engine/backtester.py
├── rlhf/
│   ├── rl_agent.py
│   ├── reward_model.py
│   └── policy_optimizer.py
├── utils/risk_manager.py
└── config/settings.py
```

---

## 4. Data Flow

### 4.1 Initialization Phase

```
1. QuantSystem.__init__()
   ├── Create directories (data/, models/, output/)
   ├── Initialize component configs
   └── Create empty state

2. system.add_data() or system.generate_synthetic_data()
   ├── Load/generate OHLCV data
   ├── Validate columns
   └── Store in _data dictionary
```

### 4.2 Model Setup Phase

```
3. system.initialize_models()
   ├── Create SpreadModel with config
   ├── Create SignalGenerator
   ├── Create MeanReversionStrategy
   ├── Create BacktestEngine
   ├── Create RiskManager
   └── Analyze each pair (cointegration, correlation, Hurst)
```

### 4.3 Signal Generation Phase

```
4. system.generate_signals()
   ├── For each pair:
   │   ├── Update spread model state
   │   ├── Compute Z-score
   │   └── Generate TradingSignal
   └── Return list of signals
```

### 4.4 Backtest Phase

```
5. system.run_backtest()
   ├── Reset backtester state
   ├── For each bar in data:
   │   ├── Update spread models
   │   ├── Generate signals
   │   ├── Execute trades
   │   └── Update positions
   ├── Compute metrics
   └── Return BacktestResult
```

### 4.5 RLHF Training Phase

```
6. system.train_rl_agent()
   ├── Create TradingEnvironment
   ├── Initialize RLHFAgent
   ├── Train with PPO
   ├── Apply human feedback bonuses
   └── Evaluate policy
```

---

## 5. Configuration System

### 5.1 System Config (`config/settings.py`)

```python
@dataclass
class SystemConfig:
    data_dir: str = "data"
    model_dir: str = "models"
    output_dir: str = "output"
    initial_capital: float = 1_000_000
    max_drawdown: float = 0.15
    max_position_size: float = 0.25
    log_level: str = "INFO"
```

### 5.2 Spread Config

```python
@dataclass
class SpreadConfig:
    method: str = 'rolling_regression'
    hedge_ratio_window: int = 60
    lookback_period: int = 252
    min_correlation: float = 0.5
    min_cointegration_pvalue: float = 0.05
```

### 5.3 Trade Config

```python
@dataclass
class TradeConfig:
    entry_threshold: float = 2.0
    strong_entry_threshold: float = 3.0
    exit_threshold: float = 0.5
    stop_loss_threshold: float = 4.0
    max_position_size: float = 0.25
    enable_scaling: bool = True
```

---

## 6. API Reference

### QuantSystem Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `add_data(symbol, data)` | Add market data | self |
| `load_from_csv(symbol, filepath)` | Load from CSV | self |
| `generate_synthetic_data(symbols, n_points)` | Generate test data | self |
| `add_pair(asset_a, asset_b)` | Configure trading pair | self |
| `initialize_models()` | Setup all models | diagnostics dict |
| `generate_signals()` | Generate trading signals | List[TradingSignal] |
| `run_backtest()` | Execute backtest | BacktestResult |
| `train_rl_agent(timesteps)` | Train RL policy | eval metrics |
| `add_human_feedback(trade_id, score)` | Add trader feedback | self |
| `check_risk_limits()` | Validate risk | (ok, breaches) |
| `export_backtest_results()` | Save results | Path |
| `export_state()` | Save system state | Path |

---

## 7. Output Files

### 7.1 Backtest Results (`output/`)

| File | Content |
|------|---------|
| `equity_curve.csv` | Timestamp, equity value |
| `drawdown_curve.csv` | Timestamp, drawdown % |
| `trades.csv` | All executed trades with PnL |
| `metrics.csv` | Performance metrics |
| `summary.json` | Complete summary |

### 7.2 State Exports

```
output/state_YYYYMMDD_HHMMSS.json
{
    "timestamp": 10000,
    "capital": 1050000,
    "positions": {...},
    "signals": [...],
    "spread_states": {...},
    "risk_metrics": {...},
    "rl_agent_trained": true,
    "backtest_complete": true
}
```

---

## 8. Error Handling

The system implements comprehensive error handling:

```python
# Data validation
if missing columns:
    raise ValueError(f"Missing columns: {missing}")

# Model initialization checks
if self._spread_model is None:
    raise RuntimeError("Models not initialized")

# Risk limit enforcement
if not self._check_risk_limits():
    logger.warning("Risk limit breached - stopping backtest")
    break
```

---

## 9. Logging System

Hierarchical logging with configurable levels:

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Usage in modules
logger = logging.getLogger(__name__)
logger.info("Starting backtest...")
logger.debug(f"Processing bar {i}")
logger.warning(f"Risk breach: {breach}")
logger.error(f"Failed to load data: {e}")
```

---

## 10. Performance Optimizations

### 10.1 Vectorization

All numerical operations use NumPy vectorization:

```python
# Instead of loops
spread = np.log(prices_a) - beta * np.log(prices_b)
z_score = (spread - rolling_mean) / rolling_std
```

### 10.2 Caching

Frequently accessed data is cached:

```python
self._cache: Dict[str, pd.DataFrame] = {}
self._states: Dict[Tuple[str, str], SpreadState] = {}
```

### 10.3 Lazy Initialization

Components created on demand:

```python
def get_rl_agent(self):
    if self._rl_agent is None:
        self.initialize_rl_agent()
    return self._rl_agent
```

---

## 11. Testing Guidelines

### 11.1 Unit Tests

```python
def test_spread_model():
    model = SpreadModel()
    hedge_ratio = model.compute_hedge_ratio_static(prices_a, prices_b)
    assert 0 < hedge_ratio < 10

def test_signal_generator():
    gen = SignalGenerator()
    signal = gen.generate_signal(z_score=-3.0, ...)
    assert signal.direction == SignalDirection.LONG_SPREAD
    assert signal.strength == SignalStrength.VERY_STRONG
```

### 11.2 Integration Tests

```python
def test_complete_pipeline():
    system = QuantSystem()
    results = system.run_complete_pipeline(
        symbols=['A', 'B'],
        pairs=[('A', 'B')],
        train_rl=False
    )
    assert results['n_trades'] > 0
    assert 'sharpe_ratio' in results['backtest_metrics']
```

---

## 12. Production Deployment

### 12.1 Environment Variables

```bash
export QUANT_DATA_DIR=/data/market
export QUANT_MODEL_DIR=/models/saved
export QUANT_CAPITAL=10000000
export QUANT_LOG_LEVEL=WARNING
```

### 12.2 Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

### 12.3 Scheduled Execution

```bash
# Cron job for daily backtest
0 2 * * * cd /quant_arb_system && python main.py --no-rl >> logs/daily.log
```

---

## 13. Monitoring and Alerts

### 13.1 Health Checks

```python
def health_check(system: QuantSystem) -> Dict:
    return {
        'status': 'healthy',
        'data_loaded': len(system._data) > 0,
        'models_initialized': system._spread_model is not None,
        'rl_trained': system._rl_agent is not None,
        'risk_ok': system.check_risk_limits()[0]
    }
```

### 13.2 Alert Conditions

```python
# Drawdown alert
if metrics['max_drawdown'] > 0.10:
    send_alert("Drawdown exceeded 10%")

# RL performance alert
if rl_metrics['mean_reward'] < -0.5:
    send_alert("RL agent underperforming")
```

---

This implementation provides a complete, production-ready statistical arbitrage system with RLHF optimization.
