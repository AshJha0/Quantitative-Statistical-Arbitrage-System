# STEP 5: Backtesting Framework

Complete documentation for the backtesting engine.

---

## 1. Overview

The backtesting framework is an **event-driven simulation** that models realistic trading conditions for spread-based strategies.

### Key Features

- **Warmup Period**: Initializes models without trading
- **Realistic Costs**: Commission, spread, borrow costs
- **Risk Enforcement**: Max drawdown, daily loss limits, VaR
- **Position Management**: Full lifecycle tracking
- **Metrics**: Comprehensive performance analysis

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BACKTEST ENGINE                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐ │
│  │   WARMUP     │     │   TRADING    │     │   CLOSEOUT   │ │
│  │   PHASE      │────▶│   PHASE      │────▶│   PHASE      │ │
│  │              │     │              │     │              │ │
│  │ • Update     │     │ • Generate   │     │ • Close all  │ │
│  │   models     │     │   signals    │     │   remaining  │ │
│  │ • No trades  │     │ • Execute    │     │   positions  │ │
│  │              │     │ • Manage     │     │ • Final PnL  │ │
│  │              │     │   positions  │     │              │ │
│  └──────────────┘     └──────────────┘     └──────────────┘ │
│                                                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │                    RISK MONITOR                          ││
│  │  • Max drawdown check  • Daily loss limit  • VaR limit  ││
│  └──────────────────────────────────────────────────────────┘│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Configuration

### BacktestConfig Parameters

```python
@dataclass
class BacktestConfig:
    # Capital and sizing
    initial_capital: float = 1_000_000
    max_position_size: float = 0.25      # 25% per position
    max_total_exposure: float = 1.0       # Max gross exposure
    
    # Transaction costs
    commission_rate: float = 0.0001       # 1 bps
    spread_cost: float = 0.0005           # 5 bps slippage
    borrow_cost: float = 0.03             # 3% annual
    
    # Risk limits
    max_drawdown: float = 0.15            # 15% max DD
    daily_loss_limit: float = 0.05        # 5% daily loss
    var_limit: float = 0.02               # 2% daily VaR
    
    # Data handling
    warmup_period: int = 252              # Initial model warmup
    rebalance_frequency: int = 1          # Rebalance every N bars
```

### Recommended Settings by Asset Class

| Asset Class | Commission | Spread | Max Position |
|-------------|------------|--------|--------------|
| FX Majors | 0.5 bps | 1 bps | 50% |
| Commodities | 1 bps | 5 bps | 25% |
| Equities | 2 bps | 3 bps | 20% |
| Crypto | 5 bps | 10 bps | 10% |

---

## 4. Main Loop

### 4.1 Warmup Phase

```python
for i, timestamp in enumerate(common_index):
    if i < warmup_period:
        # Update models but don't trade
        _warmup_step(data, spread_model, pairs, timestamp, i)
        continue
    
    # Trading phase
    _trading_step(data, spread_model, strategy, pairs, timestamp, i)
    _record_equity(timestamp)
```

### 4.2 Trading Step

```python
def _trading_step(self, data, spread_model, strategy, pairs, timestamp, bar_idx):
    signals_to_execute = []
    
    # 1. Update models and generate signals
    for asset_a, asset_b in pairs:
        state = spread_model.update_state(
            prices_a=df_a.loc[:timestamp]['close'],
            prices_b=df_b.loc[:timestamp]['close'],
            asset_a=asset_a,
            asset_b=asset_b,
            current_idx=bar_idx
        )
        
        if state.is_valid:
            signal = strategy.generate_signal(...)
            signals_to_execute.append((signal, state))
    
    # 2. Update existing positions (check for exits)
    _update_positions(strategy, spread_model, data, timestamp)
    
    # 3. Execute new signals
    _execute_signals(signals_to_execute, strategy)
```

### 4.3 Position Management

```python
def _update_positions(self, strategy, spread_model, data, timestamp):
    positions_to_close = []
    
    for pair, position in list(self.positions.items()):
        # Get current Z-score
        state = spread_model.get_state(*pair)
        current_zscore = state.current_zscore
        
        # Calculate unrealized PnL
        pnl_pct = (current_zscore - entry_z) * direction * 0.01
        unrealized_pnl = pnl_pct * size * capital
        
        # Check exit conditions
        exit_decision = strategy.update_position(
            asset_a, asset_b, current_zscore, unrealized_pnl
        )
        
        if exit_decision and exit_decision[0]:
            positions_to_close.append((pair, exit_decision[1]))
    
    # Close positions
    for pair, reason in positions_to_close:
        _close_position(pair, reason, timestamp)
```

---

## 5. Exit Logic

### Exit Conditions

| Condition | Trigger | Reason |
|-----------|---------|--------|
| Mean Reversion | |Z| < 0.5 | Take profit |
| Stop Loss | |Z| > 4.0 | Limit losses |
| Max Holding | bars > 100 | Time exit |
| Risk Breach | DD > limit | Risk management |

### Exit Reason Codes

```python
# In Trade.tags dictionary
'exit_reason': 'mean_reversion'  # Normal profit
'exit_reason': 'stop_loss'       # Hit stop
'exit_reason': 'max_holding_period'  # Time exit
'exit_reason': 'end_of_backtest'  # Final closeout
```

---

## 6. Metrics Calculation

### 6.1 Return Metrics

```python
# Total Return
total_return = (equity_final - equity_initial) / equity_initial

# CAGR (Compound Annual Growth Rate)
n_years = len(equity) / 252
cagr = (equity_final / equity_initial) ** (1 / n_years) - 1
```

### 6.2 Risk-Adjusted Metrics

```python
# Sharpe Ratio (assuming 0 risk-free rate)
returns = equity.pct_change().dropna()
sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

# Sortino Ratio (downside deviation only)
downside = returns[returns < 0]
sortino = (returns.mean() / downside.std()) * np.sqrt(252)
```

### 6.3 Drawdown Metrics

```python
# Maximum Drawdown
peak = equity.expanding().max()
drawdown = (peak - equity) / peak
max_dd = drawdown.max()

# Average Drawdown
avg_dd = drawdown.mean()

# Maximum DD Duration
max_duration = max consecutive bars where drawdown > 0
```

### 6.4 Trade Metrics

```python
# Win Rate
winning_trades = [t for t in trades if t.pnl > 0]
win_rate = len(winning_trades) / len(trades)

# Profit Factor
gross_profit = sum(t.pnl for t in winning_trades)
gross_loss = abs(sum(t.pnl for t in losing_trades))
profit_factor = gross_profit / gross_loss if gross_loss > 0 else inf

# Average Win/Loss
avg_win = np.mean([t.pnl for t in winning_trades])
avg_loss = np.mean([t.pnl for t in losing_trades])
```

---

## 7. Risk Limit Enforcement

### 7.1 Max Drawdown Check

```python
def _check_risk_limits(self) -> bool:
    current_dd = self.drawdown_curve[-1]
    
    if current_dd >= self.config.max_drawdown:
        logger.warning(f"Max drawdown breached: {current_dd:.2%}")
        return False
    
    # Daily loss limit
    if len(self.equity_curve) > 100:
        daily_change = (self.equity_curve[-1] - self.equity_curve[-101]) / self.equity_curve[-101]
        if daily_change <= -self.config.daily_loss_limit:
            logger.warning(f"Daily loss limit breached: {daily_change:.2%}")
            return False
    
    return True
```

### 7.2 Action on Breach

When risk limits are breached:
1. Stop opening new positions
2. Close existing positions at next opportunity
3. Terminate backtest early
4. Record breach in metrics

---

## 8. Output Format

### 8.1 BacktestResult Object

```python
@dataclass
class BacktestResult:
    trades: List[Trade]              # All executed trades
    equity_curve: pd.Series          # Equity over time
    drawdown_curve: pd.Series        # Drawdown over time
    positions_history: List[Dict]    # Position snapshots
    metrics: Dict[str, float]        # Performance metrics
    config: BacktestConfig           # Config used
```

### 8.2 Exported Files

```
output/
├── equity_curve.csv       # timestamp, equity
├── drawdown_curve.csv     # timestamp, drawdown_pct
├── trades.csv             # All trades with PnL
├── metrics.csv            # Metrics table
└── summary.json           # Complete summary
```

### 8.3 Example Output

```json
{
  "timestamp": "2026-04-04T12:00:00",
  "metrics": {
    "total_return": 0.1523,
    "cagr": 0.1489,
    "volatility": 0.0842,
    "sharpe_ratio": 1.768,
    "sortino_ratio": 2.341,
    "max_drawdown": 0.0834,
    "avg_drawdown": 0.0234,
    "win_rate": 0.623,
    "profit_factor": 2.15,
    "avg_win": 1250.00,
    "avg_loss": -580.00,
    "total_trades": 47
  },
  "n_trades": 47,
  "n_pairs": 2
}
```

---

## 9. Usage Examples

### 9.1 Basic Backtest

```python
from engine.backtester import BacktestEngine, BacktestConfig
from models.spread_model import SpreadModel, SpreadConfig
from strategies.mean_reversion import MeanReversionStrategy, TradeConfig

# Initialize components
engine = BacktestEngine(BacktestConfig(initial_capital=100000))
spread_model = SpreadModel(SpreadConfig())
strategy = MeanReversionStrategy(TradeConfig())

# Run backtest
result = engine.run(
    data={'XAUUSD': df_gold, 'XAGUSD': df_silver},
    spread_model=spread_model,
    strategy=strategy,
    pairs=[('XAUUSD', 'XAGUSD')]
)

# Access metrics
print(f"Sharpe: {result.metrics['sharpe_ratio']:.3f}")
print(f"Max DD: {result.metrics['max_drawdown']:.2%}")
print(f"Trades: {result.metrics['total_trades']}")
```

### 9.2 With Progress Callback

```python
def progress_callback(current, total):
    pct = current / total * 100
    print(f"Progress: {pct:.1f}%")

result = engine.run(
    data=data,
    spread_model=model,
    strategy=strategy,
    pairs=pairs,
    progress_callback=progress_callback
)
```

### 9.3 Multiple Pairs

```python
pairs = [
    ('XAUUSD', 'XAGUSD'),  # Gold/Silver
    ('EURUSD', 'GBPUSD'),  # FX pairs
    ('BTCUSD', 'ETHUSD')   # Crypto
]

result = engine.run(
    data=all_data,
    spread_model=model,
    strategy=strategy,
    pairs=pairs
)
```

---

## 10. Common Issues and Solutions

### Issue: Lookahead Bias

**Problem**: Using future data in signal generation

**Solution**: Always use `.loc[:timestamp]` to slice data

```python
# WRONG - uses entire series
spread = model.compute_spread(df_a['close'], df_b['close'])

# CORRECT - only data up to current time
spread = model.compute_spread(
    df_a.loc[:timestamp]['close'],
    df_b.loc[:timestamp]['close']
)
```

### Issue: Insufficient Warmup

**Problem**: Models not initialized properly at start

**Solution**: Use adequate warmup period (252+ bars)

```python
config = BacktestConfig(warmup_period=252)  # ~1 year of daily data
```

### Issue: Overfitting

**Problem**: Parameters optimized for historical data

**Solution**: 
- Use walk-forward optimization
- Test on out-of-sample data
- Apply parameter stability checks

---

## 11. Performance Optimization

### 11.1 Vectorization

Where possible, use vectorized operations:

```python
# Slow (loop)
for i in range(len(df)):
    z_score[i] = (spread[i] - mean[i]) / std[i]

# Fast (vectorized)
z_score = (spread - rolling_mean) / rolling_std
```

### 11.2 Caching

Cache frequently accessed calculations:

```python
self._states: Dict[Tuple[str, str], SpreadState] = {}

def get_state(self, asset_a, asset_b):
    return self._states.get((asset_a, asset_b))
```

---

## 12. Testing the Backtester

### Unit Tests

```python
def test_backtest_basic():
    engine = BacktestEngine(BacktestConfig(initial_capital=100000))
    
    # Verify initial state
    assert engine.capital == 100000
    assert len(engine.positions) == 0
    
def test_risk_limits():
    config = BacktestConfig(max_drawdown=0.10)
    engine = BacktestEngine(config)
    
    # Simulate drawdown breach
    engine.drawdown_curve = [0.0, 0.05, 0.10, 0.12]
    
    assert engine._check_risk_limits() == False
```

### Integration Tests

```python
def test_full_backtest():
    # Setup with synthetic data
    loader = DataLoader()
    df_a, df_b = loader.generate_correlated_pair(
        "A", "B", correlation=0.8, n_points=2000
    )
    
    engine = BacktestEngine(BacktestConfig())
    model = SpreadModel(SpreadConfig())
    strategy = MeanReversionStrategy(TradeConfig())
    
    result = engine.run(
        data={'A': df_a, 'B': df_b},
        spread_model=model,
        strategy=strategy,
        pairs=[('A', 'B')]
    )
    
    # Verify results
    assert result.metrics['total_trades'] > 0
    assert 'sharpe_ratio' in result.metrics
    assert result.metrics['max_drawdown'] <= 0.15  # Within limits
```

---

This backtesting framework provides a robust foundation for evaluating spread-based trading strategies with realistic constraints and comprehensive performance analysis.
