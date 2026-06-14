# STEP 8: Performance Evaluation

Comprehensive guide to evaluating statistical arbitrage strategies.

---

## 1. Overview

Performance evaluation assesses whether a strategy is:
- **Profitable**: Generates positive returns
- **Robust**: Works across different market conditions
- **Scalable**: Can handle larger capital
- **Sustainable**: Risk-adjusted returns justify deployment

---

## 2. Key Metrics

### 2.1 Return Metrics

| Metric | Formula | Good Value | Interpretation |
|--------|---------|------------|----------------|
| **Total Return** | (P_final - P_initial) / P_initial | > 10% | Overall profitability |
| **CAGR** | (P_final/P_initial)^(1/n) - 1 | > 8% | Annualized growth rate |
| **Monthly Return** | Mean of monthly returns | > 1% | Consistency |

### 2.2 Risk-Adjusted Metrics

| Metric | Formula | Good Value | Interpretation |
|--------|---------|------------|----------------|
| **Sharpe Ratio** | (R_p - R_f) / σ_p | > 1.5 | Return per unit risk |
| **Sortino Ratio** | (R_p - R_f) / σ_downside | > 2.0 | Return per downside risk |
| **Calmar Ratio** | CAGR / Max_DD | > 1.0 | Return vs worst drawdown |
| **Information Ratio** | (R_p - R_b) / TE | > 0.5 | Excess return vs tracking error |

### 2.3 Risk Metrics

| Metric | Formula | Acceptable | Interpretation |
|--------|---------|------------|----------------|
| **Max Drawdown** | Max peak-to-trough decline | < 15% | Worst loss from peak |
| **Avg Drawdown** | Mean of drawdown periods | < 5% | Typical pain |
| **DD Duration** | Max time in drawdown | < 50 bars | Recovery time |
| **VaR (95%)** | 5th percentile of returns | < 2%/day | Daily loss at risk |
| **Expected Shortfall** | Mean of tail losses | < 3%/day | Average worst-case |

### 2.4 Trading Metrics

| Metric | Formula | Good Value | Interpretation |
|--------|---------|------------|----------------|
| **Win Rate** | Wins / Total Trades | > 50% | Hit ratio |
| **Profit Factor** | Gross Profit / Gross Loss | > 1.5 | Payoff ratio |
| **Avg Win/Loss** | Mean winner / Mean loser | > 1.5 | Reward/risk per trade |
| **Expectancy** | (W% × AvgWin) - (L% × AvgLoss) | > 0 | Expected profit per trade |
| **Turnover** | Trades / Period | Context | Trading frequency |

---

## 3. Evaluation Framework

### 3.1 In-Sample vs Out-of-Sample

```
Data Timeline:
├─────────────┬─────────────┬─────────────┤
│   IS-1      │   IS-2      │    OOS      │
│  Training   │  Validation │   Testing   │
│  (60%)      │  (20%)      │   (20%)     │
└─────────────┴─────────────┴─────────────┘
```

**Best Practices:**
- Train on IS-1, tune on IS-2, test on OOS
- OOS performance should be within 20% of IS
- If OOS << IS → Overfitting detected

### 3.2 Walk-Forward Analysis

```
Window 1: [Train─────][Test]
Window 2:    [Train─────][Test]
Window 3:       [Train─────][Test]
Window 4:          [Train─────][Test]
```

**Purpose**: Validate strategy robustness across time periods

### 3.3 Monte Carlo Simulation

```python
def monte_carlo_backtest(strategy, data, n_simulations=1000):
    """Randomize trade sequence to test robustness."""
    results = []
    
    for _ in range(n_simulations):
        # Randomly sample trades with replacement
        sampled_trades = np.random.choice(
            original_trades, 
            size=len(original_trades),
            replace=True
        )
        
        # Compute equity curve
        equity = np.cumsum(sampled_trades)
        results.append({
            'final_equity': equity[-1],
            'max_dd': max_drawdown(equity)
        })
    
    return results

# Analyze distribution
mc_results = monte_carlo_backtest(strategy, data)
print(f"Probability of profit: {np.mean([r['final_equity'] > 0 for r in mc_results]):.2%}")
print(f"Probability of >20% DD: {np.mean([r['max_dd'] > 0.2 for r in mc_results]):.2%}")
```

---

## 4. Statistical Tests

### 4.1 T-Test for Mean Returns

```python
from scipy import stats

# Test if mean return is significantly different from zero
returns = equity_curve.pct_change().dropna()
t_stat, p_value = stats.ttest_1samp(returns, 0)

if p_value < 0.05:
    print(f"Returns are significant (p={p_value:.4f})")
else:
    print("Returns not statistically significant")
```

### 4.2 Bootstrap Confidence Intervals

```python
def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap confidence interval."""
    bootstrapped = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped.append(np.mean(sample))
    
    lower = np.percentile(bootstrapped, (1-ci)/2 * 100)
    upper = np.percentile(bootstrapped, (1+ci)/2 * 100)
    return (lower, upper)

# Example: Sharpe ratio CI
sharpe_bootstrap = bootstrap_ci(returns / returns.std() * np.sqrt(252))
print(f"Sharpe 95% CI: [{sharpe_bootstrap[0]:.3f}, {sharpe_bootstrap[1]:.3f}]")
```

### 4.3 Deflated Sharpe Ratio

Adjusts for multiple testing:

```python
def deflated_sharpe(sharpe, n_trials, T):
    """
    Adjust Sharpe for multiple testing.
    
    Args:
        sharpe: Observed Sharpe ratio
        n_trials: Number of strategies tested
        T: Number of observations
    
    Returns:
        Deflated Sharpe ratio
    """
    from scipy.stats import norm
    
    # Probability of observing this Sharpe by chance
    prob = 1 - norm.cdf(sharpe * np.sqrt(T))
    
    # Adjust for multiple testing
    adjusted_prob = 1 - (1 - prob) ** n_trials
    
    # Convert back to Sharpe
    deflated_sharpe = norm.ppf(1 - adjusted_prob) / np.sqrt(T)
    
    return deflated_sharpe

# Example: If you tested 50 strategies
observed_sharpe = 1.8
n_strategies = 50
n_observations = 10000

ds = deflated_sharpe(observed_sharpe, n_strategies, n_observations)
print(f"Deflated Sharpe: {ds:.3f}")
```

---

## 5. Benchmarking

### 5.1 Relevant Benchmarks

| Benchmark | Purpose |
|-----------|---------|
| **Buy & Hold** | Beat passive exposure |
| **Market Index** | Outperform broad market |
| **Sector Index** | Beat relevant sector |
| **Risk-Free Rate** | Exceed treasury yields |
| **60/40 Portfolio** | Traditional alternative |

### 5.2 Relative Metrics

```python
def compute_relative_metrics(strategy_returns, benchmark_returns):
    """Compute alpha, beta, and information ratio."""
    
    # Beta: sensitivity to benchmark
    covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
    variance = np.var(benchmark_returns)
    beta = covariance / variance
    
    # Alpha: excess return not explained by beta
    strategy_mean = strategy_returns.mean()
    benchmark_mean = benchmark_returns.mean()
    alpha = strategy_mean - beta * benchmark_mean
    
    # Annualized
    alpha_annual = alpha * 252
    beta_annual = beta
    
    # Information Ratio
    active_return = strategy_returns - benchmark_returns
    ir = active_return.mean() / active_return.std() * np.sqrt(252)
    
    return {
        'alpha': alpha_annual,
        'beta': beta_annual,
        'information_ratio': ir
    }
```

---

## 6. Performance Attribution

### 6.1 By Pair

```python
def attribute_by_pair(trades):
    """Break down P&L by trading pair."""
    pair_pnl = {}
    
    for trade in trades:
        pair = trade['pair']
        pnl = trade['pnl']
        pair_pnl[pair] = pair_pnl.get(pair, 0) + pnl
    
    total_pnl = sum(pair_pnl.values())
    
    attribution = {}
    for pair, pnl in pair_pnl.items():
        attribution[pair] = {
            'pnl': pnl,
            'contribution': pnl / total_pnl if total_pnl != 0 else 0,
            'trades': len([t for t in trades if t['pair'] == pair])
        }
    
    return attribution
```

### 6.2 By Signal Type

```python
def attribute_by_signal_type(trades):
    """Break down P&L by signal strength."""
    type_pnl = {
        'very_strong': [],
        'strong': [],
        'moderate': [],
        'weak': []
    }
    
    for trade in trades:
        strength = trade.get('signal_strength', 'moderate')
        type_pnl[strength].append(trade['pnl'])
    
    attribution = {}
    for strength, pnls in type_pnl.items():
        if pnls:
            attribution[strength] = {
                'total_pnl': sum(pnls),
                'avg_pnl': np.mean(pnls),
                'win_rate': np.mean([p > 0 for p in pnls]),
                'n_trades': len(pnls)
            }
    
    return attribution
```

### 6.3 By Time Period

```python
def attribute_by_time(trades, equity_curve):
    """Analyze performance by time period."""
    # Monthly breakdown
    monthly_returns = equity_curve.resample('M').last().pct_change()
    
    # Hourly breakdown (for intraday)
    hourly_pnl = equity_curve.resample('H').last().diff()
    
    return {
        'monthly': monthly_returns,
        'best_month': monthly_returns.max(),
        'worst_month': monthly_returns.min(),
        'positive_months': (monthly_returns > 0).sum() / len(monthly_returns)
    }
```

---

## 7. Risk Analysis

### 7.1 Drawdown Analysis

```python
def analyze_drawdowns(equity_curve):
    """Detailed drawdown analysis."""
    peak = equity_curve.expanding().max()
    drawdown = (peak - equity_curve) / peak
    
    # Find drawdown periods
    in_drawdown = drawdown > 0.01  # 1% threshold
    
    drawdown_periods = []
    start_idx = None
    
    for i, in_dd in enumerate(in_drawdown):
        if in_dd and start_idx is None:
            start_idx = i
        elif not in_dd and start_idx is not None:
            dd_start = start_idx
            dd_end = i
            dd_depth = drawdown[dd_start:dd_end].max()
            dd_duration = dd_end - dd_start
            drawdown_periods.append({
                'start': dd_start,
                'end': dd_end,
                'depth': dd_depth,
                'duration': dd_duration
            })
            start_idx = None
    
    return {
        'max_drawdown': drawdown.max(),
        'avg_drawdown': drawdown[in_drawdown].mean(),
        'n_drawdowns': len(drawdown_periods),
        'avg_duration': np.mean([d['duration'] for d in drawdown_periods]) if drawdown_periods else 0,
        'periods': drawdown_periods
    }
```

### 7.2 Tail Risk Analysis

```python
def analyze_tail_risk(returns, confidence=0.95):
    """Analyze extreme losses."""
    
    # Value at Risk
    var_95 = np.percentile(returns, (1-confidence) * 100)
    var_99 = np.percentile(returns, 0.01 * 100)
    
    # Expected Shortfall (CVaR)
    es_95 = returns[returns <= var_95].mean()
    es_99 = returns[returns <= var_99].mean()
    
    # Worst days
    worst_5 = returns.nsmallest(5)
    
    return {
        'var_95': var_95,
        'var_99': var_99,
        'es_95': es_95,
        'es_99': es_99,
        'worst_5_days': worst_5.tolist()
    }
```

---

## 8. Evaluation Checklist

### 8.1 Pre-Deployment Checklist

```
□ Backtest period > 5 years (or equivalent data)
□ Out-of-sample testing complete
□ Walk-forward analysis passed
□ Monte Carlo simulation shows robustness
□ Sharpe ratio > 1.5 in OOS
□ Max drawdown < 15%
□ Win rate > 45%
□ Profit factor > 1.5
□ No single trade > 5% of total P&L
□ Strategy works across multiple pairs
□ Transaction costs included
□ Slippage modeled realistically
```

### 8.2 Red Flags

| Issue | Severity | Action |
|-------|----------|--------|
| OOS Sharpe < 0.5 | Critical | Do not deploy |
| Max DD > 25% | Critical | Reduce leverage |
| Win rate < 40% | High | Review entry logic |
| Profit factor < 1.2 | High | Optimize exits |
| Single trade > 10% P&L | Medium | Reduce size |
| Overfitting detected | Critical | Simplify strategy |

---

## 9. Reporting Template

### 9.1 Executive Summary

```
STRATEGY PERFORMANCE REPORT
═══════════════════════════════════════════════════════════

Strategy: Mean Reversion Pairs Trading
Period: 2024-01-01 to 2026-04-04
Initial Capital: $1,000,000
Final Capital: $1,152,300

KEY METRICS
───────────────────────────────────────────────────────────
Total Return:        15.23%
CAGR:                7.89%
Sharpe Ratio:        1.77
Max Drawdown:        8.34%
Win Rate:            62.3%
Profit Factor:       2.15

RISK ASSESSMENT
───────────────────────────────────────────────────────────
VaR (95%):           1.2%/day
Expected Shortfall:  1.8%/day
Beta to Market:      0.12
Alpha:               6.5%

RECOMMENDATION: APPROVED FOR DEPLOYMENT
```

### 9.2 Detailed Metrics Table

| Metric | Value | Benchmark | Status |
|--------|-------|-----------|--------|
| Total Return | 15.23% | 8.0% | ✅ Pass |
| Sharpe Ratio | 1.77 | 1.5 | ✅ Pass |
| Max Drawdown | 8.34% | 15% | ✅ Pass |
| Win Rate | 62.3% | 50% | ✅ Pass |
| Profit Factor | 2.15 | 1.5 | ✅ Pass |
| Calmar Ratio | 1.83 | 1.0 | ✅ Pass |
| OOS Degradation | 12% | 20% | ✅ Pass |

---

## 10. Continuous Monitoring

### 10.1 Live Performance Tracking

```python
def monitor_live_performance(live_trades, expected_metrics, tolerance=0.2):
    """Compare live performance to backtest expectations."""
    
    live_metrics = compute_metrics(live_trades)
    
    deviations = {}
    for metric in ['sharpe', 'win_rate', 'profit_factor']:
        expected = expected_metrics.get(metric, 0)
        actual = live_metrics.get(metric, 0)
        deviation = abs(actual - expected) / expected if expected > 0 else 0
        deviations[metric] = deviation
    
    # Alert if deviation exceeds tolerance
    alerts = []
    for metric, dev in deviations.items():
        if dev > tolerance:
            alerts.append(f"{metric} deviated by {dev:.2%}")
    
    return {
        'deviations': deviations,
        'alerts': alerts,
        'status': 'OK' if not alerts else 'ALERT'
    }
```

### 10.2 Decay Detection

```python
def detect_performance_decay(metrics_history, window=20):
    """Detect if strategy performance is decaying."""
    
    if len(metrics_history) < window * 2:
        return {'decay_detected': False}
    
    recent = metrics_history[-window:]
    historical = metrics_history[:-window]
    
    # Compare Sharpe ratios
    recent_sharpe = np.mean(recent)
    historical_sharpe = np.mean(historical)
    
    decay = (historical_sharpe - recent_sharpe) / historical_sharpe
    
    return {
        'decay_detected': decay > 0.3,  # 30% degradation threshold
        'decay_rate': decay,
        'recent_sharpe': recent_sharpe,
        'historical_sharpe': historical_sharpe
    }
```

---

## 11. Common Pitfalls

### 11.1 Look-Ahead Bias

**Problem**: Using future data in signal generation

**Detection**: 
- Check if any `.shift()` operations are missing
- Verify all indicators use only past data

**Fix**:
```python
# WRONG
z_score = (spread - spread.rolling(252).mean()) / spread.rolling(252).std()

# CORRECT
z_score = (spread - spread.rolling(252).mean().shift(1)) / spread.rolling(252).std().shift(1)
```

### 11.2 Survivorship Bias

**Problem**: Testing only on assets that survived

**Fix**: Include delisted assets in backtest universe

### 11.3 Overfitting

**Symptoms**:
- IS Sharpe >> OOS Sharpe
- Many parameters tuned
- Complex rules

**Fix**:
- Simplify strategy
- Reduce parameters
- Use cross-validation

---

This evaluation framework ensures rigorous assessment of statistical arbitrage strategies before and after deployment.
