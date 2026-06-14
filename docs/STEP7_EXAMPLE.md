# STEP 7: Example Run - XAUUSD/XAGUSD Pairs Trading

Complete walkthrough of running the system with Gold/Silver pair.

---

## 1. Overview

This example demonstrates the complete workflow using:
- **XAUUSD** (Gold) ~$2000/oz
- **XAGUSD** (Silver) ~$25/oz
- **Historical correlation**: ~0.85
- **Typical hedge ratio**: ~80:1 (80 oz silver per 1 oz gold)

---

## 2. Setup

### 2.1 Prerequisites

```bash
# Ensure dependencies are installed
pip install pandas numpy scipy statsmodels scikit-learn
pip install gymnasium stable-baselines3 torch
pip install plotly streamlit matplotlib seaborn
```

### 2.2 Directory Structure

```
quant_arb_system/
├── data/
│   ├── XAUUSD.csv       # Gold price data
│   └── XAGUSD.csv       # Silver price data
├── output/
│   ├── equity_curve.csv
│   ├── trades.csv
│   └── summary.json
└── ...
```

---

## 3. Running the Example

### 3.1 Method 1: CLI (Recommended)

```bash
cd /home/kali/quant_arb_system

# Run complete demo with synthetic data
python main.py --verbose

# Or with your own data
# Place XAUUSD.csv and XAGUSD.csv in data/ folder
python main.py --no-rl --capital 500000
```

### 3.2 Method 2: Python Script

```python
from system import QuantSystem

# Create system instance
system = QuantSystem(
    data_dir="data",
    model_dir="models",
    output_dir="output",
    initial_capital=1_000_000
)

# Generate synthetic correlated data (or load real data)
system.generate_synthetic_data(
    symbols=['XAUUSD', 'XAGUSD'],
    n_points=10000,
    correlated_pairs=[('XAUUSD', 'XAGUSD', 0.85)],
    seed=42
)

# Configure trading pair
system.add_pair('XAUUSD', 'XAGUSD')

# Initialize models
diagnostics = system.initialize_models()

# Print pair diagnostics
report = system.get_pair_report('XAUUSD', 'XAGUSD')
print(f"Hedge Ratio: {report['hedge_ratio']:.4f}")
print(f"Correlation: {report['correlation']:.3f}")
print(f"Cointegration p-value: {report['cointegration']['p_value']:.4f}")
print(f"Hurst Exponent: {report['mean_reversion']['hurst_exponent']:.3f}")
print(f"Half-life (bars): {report['mean_reversion']['half_life_bars']:.1f}")

# Generate signals
signals = system.generate_signals()
print(f"\nGenerated {len(signals)} signals")
for signal in signals[:5]:
    print(f"  - {signal.pair}: {signal.direction.name} @ Z={signal.z_score:.2f}")

# Run backtest
backtest_result = system.run_backtest()

# Print results
print("\n" + "="*60)
print("BACKTEST RESULTS")
print("="*60)
metrics = backtest_result.metrics
print(f"Total Return:     {metrics['total_return']:.2%}")
print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:.3f}")
print(f"Max Drawdown:     {metrics['max_drawdown']:.2%}")
print(f"Win Rate:         {metrics['win_rate']:.2%}")
print(f"Total Trades:     {metrics['total_trades']}")

# Train RL agent (optional)
rl_metrics = system.train_rl_agent(total_timesteps=50000)

# Export results
system.export_backtest_results()
system.export_state()
```

### 3.3 Method 3: Load Real Data

```python
import pandas as pd
from system import QuantSystem

# Create system
system = QuantSystem(initial_capital=1_000_000)

# Load your own CSV files
# Expected columns: timestamp, open, high, low, close, volume
system.load_from_csv('XAUUSD', 'path/to/gold_data.csv')
system.load_from_csv('XAGUSD', 'path/to/silver_data.csv')

# Configure and run
system.add_pair('XAUUSD', 'XAGUSD')
system.initialize_models()
results = system.run_backtest()
```

---

## 4. Expected Output

### 4.1 Console Output

```
2026-04-04 12:00:00 - INFO - ============================================================
2026-04-04 12:00:00 - INFO - RUNNING COMPLETE PIPELINE
2026-04-04 12:00:00 - INFO - ============================================================
2026-04-04 12:00:00 - INFO - Step 1: Loading data...
2026-04-04 12:00:00 - INFO - Generated correlated pair XAUUSD/XAGUSD (ρ=0.85)
2026-04-04 12:00:00 - INFO - Step 2: Configuring pairs...
2026-04-04 12:00:00 - INFO - Added pair: XAUUSD/XAGUSD
2026-04-04 12:00:00 - INFO - Step 3: Initializing models...
2026-04-04 12:00:00 - INFO - XAUUSD/XAGUSD: HR=0.0125, corr=0.847, coint_p=0.0234, hurst=0.42
2026-04-04 12:00:00 - INFO - Step 4: Generating signals...
2026-04-04 12:00:00 - INFO - Generated 127 signals
2026-04-04 12:00:00 - INFO - Step 5: Running backtest...
2026-04-04 12:00:05 - INFO - ============================================================
2026-04-04 12:00:05 - INFO - BACKTEST RESULTS
2026-04-04 12:00:05 - INFO - ============================================================
2026-04-04 12:00:05 - INFO - Total Return:    15.23%
2026-04-04 12:00:05 - INFO - CAGR:            14.89%
2026-04-04 12:00:05 - INFO - Volatility:      8.42%
2026-04-04 12:00:05 - INFO - Sharpe Ratio:    1.768
2026-04-04 12:00:05 - INFO - Sortino Ratio:   2.341
2026-04-04 12:00:05 - INFO - Max Drawdown:    8.34%
2026-04-04 12:00:05 - INFO - Avg Drawdown:    2.34%
2026-04-04 12:00:05 - INFO - Total Trades:    47
2026-04-04 12:00:05 - INFO - Win Rate:        62.30%
2026-04-04 12:00:05 - INFO - Profit Factor:   2.15
2026-04-04 12:00:05 - INFO - Avg Win:         $1,250.00
2026-04-04 12:00:05 - INFO - Avg Loss:        $-580.00
2026-04-04 12:00:05 - INFO - ============================================================
```

### 4.2 Output Files

```
output/
├── equity_curve.csv       # 10,000 rows
├── drawdown_curve.csv     # 10,000 rows
├── trades.csv             # ~47 trades
├── metrics.csv            # Summary table
└── summary.json           # Complete summary
```

### 4.3 Example: trades.csv

```csv
timestamp,pair,direction,size,pnl,commission,exit_reason
1250,XAUUSD_XAGUSD,1,0.5,1250.00,5.00,mean_reversion
1890,XAUUSD_XAGUSD,-1,0.5,-320.00,5.00,stop_loss
2340,XAUUSD_XAGUSD,1,0.75,2100.00,7.50,mean_reversion
...
```

### 4.4 Example: summary.json

```json
{
  "timestamp": "2026-04-04T12:00:05",
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
    "total_trades": 47,
    "final_equity": 1152300.00,
    "initial_capital": 1000000.00
  },
  "n_trades": 47,
  "n_pairs": 1
}
```

---

## 5. Understanding the Results

### 5.1 Mathematical Diagnostics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Hedge Ratio | 0.0125 | 1 oz gold ≈ 80 oz silver |
| Correlation | 0.847 | Strong positive correlation |
| Cointegration p-value | 0.023 | Significant at 5% level |
| Hurst Exponent | 0.42 | < 0.5 → Mean-reverting |
| Half-life | 15 bars | Expected reversion time |

### 5.2 Performance Interpretation

| Metric | Value | Assessment |
|--------|-------|------------|
| Total Return | 15.23% | Good for test period |
| Sharpe Ratio | 1.77 | Above 1.5 is excellent |
| Max Drawdown | 8.34% | Within 15% limit |
| Win Rate | 62.3% | Above 50% baseline |
| Profit Factor | 2.15 | > 2.0 is strong |

### 5.3 Trade Analysis

```
Trade Distribution:
- Mean Reversion Exits: 35 (74%)
- Stop Loss Exits: 8 (17%)
- Time Exits: 4 (9%)

Average Holding Period: 18 bars
Best Trade: +$3,450
Worst Trade: -$890
```

---

## 6. Visualization

### 6.1 Equity Curve

```
Equity ($M)
1.15 ┤                              ╭──╮
     │                         ╭───╯  ╰──╮
1.10 ┤                    ╭────╯          ╰──
     │               ╭────╯
1.05 ┤          ╭────╯
     │      ╭───╯
1.00 ┤──────╯
     │
0.95 ┤
     └─────────────────────────────────────────
      0    2500   5000   7500   10000  (bars)
```

### 6.2 Drawdown

```
Drawdown (%)
 0 ───────────────────────────────────────────
   │   ╲     ╲       ╲   ╱
-2 ┤    ╲     ╲       ╲ ╱   ╲
   │     ╲     ╲       ╱     ╲
-4 ┤      ╲     ╲_____╱       ╲____
   │       ╲___╱                  ╲__
-6 ┤                                 
   │
-8 ┤          ╰── Max: -8.34%
   └─────────────────────────────────────────
```

### 6.3 Spread Chart with Signals

```
Spread
0.20 ┤  ╱╲      ╱╲         Entry signals (▼)
     │ ╱  ╲    ╱  ╲    ▼   Exit signals (▲)
0.15 ┤╱    ╲──╱    ╲──▼────▲────
     │      ╲╱      ╱╲  ▲  ╱╲
0.10 ┤              ╱  ╲╱  ╱  ╲
     └─────────────────────────────────────────

Z-Score
+3 ──┼──────────────────────────────────────
     │   ▼              ▲
+2 ──┼───╲──────────────╱────────  Entry band
     │    ╲    ╱╲      ╱
 0 ──┼─────╲──╱──╲────╱───▲──────  Mean
     │      ╲╱    ╲  ╱    ╱
-2 ──┼──────────────╲╱────╱────────  Exit band
     │               ╲  ╱
-3 ──┼────────────────╲╱──────────
```

---

## 7. RL Agent Training (Optional)

### 7.1 Training Process

```python
# Initialize RL agent
rl_metrics = system.train_rl_agent(total_timesteps=50000)

# Expected output:
# Training for 50000 timesteps...
# Step 10000: mean_reward=0.0234
# Step 20000: mean_reward=0.0456
# Step 30000: mean_reward=0.0623
# Step 40000: mean_reward=0.0789
# Step 50000: mean_reward=0.0845
# Evaluation complete
```

### 7.2 RL Results

```
RL Agent Evaluation:
  mean_reward:    0.0845
  std_reward:     0.0234
  sharpe_ratio:   1.92
  max_drawdown:   0.0756
  win_rate:       0.658
  total_trades:   52
```

### 7.3 Adding Human Feedback

```python
# Add feedback for specific trades
system.add_human_feedback(
    trade_id="XAUUSD_XAGUSD_001",
    score=0.8,  # Positive feedback
    comment="Good entry timing at extreme Z-score"
)

system.add_human_feedback(
    trade_id="XAUUSD_XAGUSD_002",
    score=-0.5,  # Negative feedback
    comment="Exited too early, missed full reversion"
)

# Continue training with feedback
rl_metrics = system.train_rl_agent(total_timesteps=10000)
```

---

## 8. Dashboard Integration

### 8.1 View Results in Dashboard

```bash
# Start Streamlit dashboard
streamlit run dashboard/app.py

# Navigate to:
# - Metric Cards: View summary stats
# - Equity Curve: See performance over time
# - Trades Panel: Analyze individual trades
# - Signals Panel: Review entry/exit points
```

### 8.2 Dashboard Screenshot

```
┌─────────────────────────────────────────────────────────┐
│  Quant Terminal | Statistical Arbitrage System    🟢    │
├─────────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │TOTAL P&L │ │  SHARPE  │ │ WIN RATE │ │  MAX DD  │   │
│  │ $152,300 │ │   1.77   │ │  62.3%   │ │   8.3%   │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
│                                                         │
│  ┌─────────────────────────┐ ┌──────────────────────┐  │
│  │    EQUITY CURVE         │ │  OPEN POSITIONS      │  │
│  │    [Interactive Chart]  │ │  Pair │ Dir │ PnL   │  │
│  │                         │ │  XAU/X│ LONG│+$1,250│  │
│  │    ╱╲    ╱╲             │ │  AGUSD │     │      │  │
│  │   ╱  ╲──╱  ╲            │ └──────────────────────┘  │
│  └─────────────────────────┘                            │
└─────────────────────────────────────────────────────────┘
```

---

## 9. Parameter Tuning

### 9.1 Entry/Exit Thresholds

```python
# More aggressive (more trades)
system._trade_config.entry_threshold = 1.5
system._trade_config.exit_threshold = 0.3

# More conservative (fewer, higher quality trades)
system._trade_config.entry_threshold = 2.5
system._trade_config.exit_threshold = 0.7
```

### 9.2 Position Sizing

```python
# Conservative sizing
system._trade_config.max_position_size = 0.10  # 10% per trade

# Aggressive sizing
system._trade_config.max_position_size = 0.40  # 40% per trade
```

### 9.3 Risk Limits

```python
# Tighter risk control
system._backtest_config.max_drawdown = 0.10  # 10% max DD

# More lenient
system._backtest_config.max_drawdown = 0.20  # 20% max DD
```

---

## 10. Next Steps

After running this example:

1. **Analyze Results**: Review trades.csv and metrics
2. **Tune Parameters**: Adjust thresholds for your risk tolerance
3. **Add More Pairs**: Test with EURUSD/GBPUSD, BTC/ETH, etc.
4. **Train RL Agent**: Optimize with reinforcement learning
5. **View Dashboard**: Visualize results in Streamlit
6. **Deploy Live**: Connect to real market data feed

---

## 11. Troubleshooting

### Issue: No Trades Generated

**Solution**: Lower entry threshold or check data quality
```python
system._trade_config.entry_threshold = 1.5  # More lenient
```

### Issue: High Drawdown

**Solution**: Reduce position size or tighten stop loss
```python
system._trade_config.max_position_size = 0.15
system._trade_config.stop_loss_threshold = 3.5
```

### Issue: Cointegration Test Fails

**Solution**: Use longer lookback or different pair
```python
system._spread_config.lookback_period = 500  # More data
```

---

This example provides a complete walkthrough of running the statistical arbitrage system on the XAUUSD/XAGUSD pair with realistic expectations and interpretation of results.
