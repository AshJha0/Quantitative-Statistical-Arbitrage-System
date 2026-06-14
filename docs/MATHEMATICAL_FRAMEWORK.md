# STEP 2: Mathematical Modeling Framework

This document details the mathematical foundations of the statistical arbitrage system.

---

## 1. Spread Construction

### 1.1 Log-Price Spread

For two assets A and B with prices $P_A$ and $P_B$:

$$\text{Spread}_t = \log(P_{A,t}) - \beta \cdot \log(P_{B,t})$$

Where:
- $\beta$ is the hedge ratio (see Section 2)
- Log transformation ensures returns-like behavior and better statistical properties

### 1.2 Rationale for Log Prices

1. **Stationarity**: Log prices typically have better stationarity properties than raw prices
2. **Symmetry**: Log returns are symmetric around zero
3. **Additivity**: Multi-period log returns are additive
4. **Normalization**: Spreads are naturally scaled

---

## 2. Hedge Ratio Estimation

### 2.1 OLS Estimator

The standard approach estimates $\beta$ via Ordinary Least Squares:

$$\log(P_A) = \alpha + \beta \cdot \log(P_B) + \varepsilon$$

$$\hat{\beta}_{OLS} = \frac{\text{Cov}(\log P_A, \log P_B)}{\text{Var}(\log P_B)}$$

**Standard Error:**

$$SE(\hat{\beta}) = \sqrt{\frac{\hat{\sigma}^2_\varepsilon}{\sum (x_i - \bar{x})^2}}$$

**Implementation:** `HedgeRatioEstimator.ols()`

### 2.2 Rolling OLS

For time-varying hedge ratios:

$$\hat{\beta}_t = \text{OLS}(\log P_A[t-w:t], \log P_B[t-w:t])$$

Where $w$ is the rolling window (default: 60 bars).

**Trade-off:**
- Shorter window: More responsive, higher variance
- Longer window: More stable, slower adaptation

### 2.3 Exponentially Weighted Moving (EWM)

$$\hat{\beta}_t = \frac{\text{EWM}[\text{Cov}(\Delta \log P_A, \Delta \log P_B)]}{\text{EWM}[\text{Var}(\Delta \log P_B)]}$$

With span parameter controlling decay rate.

**Advantage:** Smooth adaptation without discrete window effects.

### 2.4 Total Least Squares (TLS)

When both assets have measurement error:

$$\hat{\beta}_{TLS} = \arg\min_{\beta} \sum_i \left[(y_i - \beta x_i)^2 + \gamma \cdot \varepsilon_{x,i}^2\right]$$

Solved via SVD of the data matrix.

---

## 3. Cointegration Testing

### 3.1 Engle-Granger Two-Step Method

**Step 1:** Estimate long-run relationship
$$\log(P_A) = \alpha + \beta \log(P_B) + \varepsilon_t$$

**Step 2:** Test residuals for stationarity using ADF:
$$\Delta \varepsilon_t = \alpha + \beta \varepsilon_{t-1} + \sum_{i=1}^p \gamma_i \Delta \varepsilon_{t-i} + u_t$$

**Null hypothesis:** $\beta = 0$ (unit root, no cointegration)

**Critical values** (for 2 variables):
| Significance | Critical Value |
|--------------|----------------|
| 1%           | -3.90          |
| 5%           | -3.34          |
| 10%          | -3.04          |

**Implementation:** `StatisticalTests.engle_granger()`

### 3.2 Johansen Test (Multi-Asset)

For basket spreads with $n > 2$ assets, use Johansen's maximum likelihood approach.

Tests the rank $r$ of the cointegration matrix $\Pi$ in the VECM:

$$\Delta X_t = \Pi X_{t-1} + \sum_{i=1}^{k-1} \Gamma_i \Delta X_{t-i} + \varepsilon_t$$

**Trace statistic:**
$$\lambda_{trace} = -T \sum_{i=r+1}^n \log(1 - \hat{\lambda}_i)$$

Where $\hat{\lambda}_i$ are the eigenvalues of $\Pi$.

---

## 4. Spread Normalization

### 4.1 Rolling Z-Score

$$Z_t = \frac{\text{Spread}_t - \mu_{t-1}}{\sigma_{t-1}}$$

Where:
- $\mu_{t-1} = \frac{1}{w} \sum_{i=t-w}^{t-1} \text{Spread}_i$
- $\sigma_{t-1} = \sqrt{\frac{1}{w-1} \sum_{i=t-w}^{t-1} (\text{Spread}_i - \mu_{t-1})^2}$

**Critical:** Uses only past data to avoid lookahead bias.

### 4.2 Expanding Z-Score

$$Z_t = \frac{\text{Spread}_t - \mu_{[0:t-1]}}{\sigma_{[0:t-1]}}$$

More stable but slower to adapt to regime changes.

### 4.3 EWM Z-Score

$$Z_t = \frac{\text{Spread}_t - \text{EWM}[\text{Spread}]_{t-1}}{\text{EWM}[\sigma]_{t-1}}$$

Faster adaptation, controlled by span parameter.

### 4.4 Percentile Rank (Robust Alternative)

$$\text{Percentile}_t = \frac{\text{Rank}(\text{Spread}_t \text{ in window})}{\text{Window Size}}$$

More robust to outliers and non-normal distributions.

---

## 5. Mean Reversion Analysis

### 5.1 Hurst Exponent

Measures the tendency of a series to mean-revert or trend:

- $H < 0.5$: Mean-reverting
- $H = 0.5$: Random walk (GBM)
- $H > 0.5$: Trending

**Calculation via R/S analysis:**

$$R/S_n = \frac{\max(R_1, ..., R_n) - \min(R_1, ..., R_n)}{\sigma_n} \propto n^H$$

Where $R_i$ are cumulative deviations from mean.

**Implementation:** `StatisticalTests.hurst_exponent()`

### 5.2 Ornstein-Uhlenbeck Process

Models the spread as a mean-reverting process:

$$dX_t = \theta (\mu - X_t) dt + \sigma dW_t$$

Where:
- $\theta$: Mean reversion speed (higher = faster)
- $\mu$: Long-term mean
- $\sigma$: Volatility of innovations
- $W_t$: Wiener process

**Discrete approximation:**
$$X_{t+1} - X_t = \theta \mu - \theta X_t + \sigma \varepsilon_t$$

**Parameter estimation via OLS:**
$$\Delta X_t = \alpha + \beta X_t + \varepsilon_t$$

Mapping:
- $\theta = -\beta$
- $\mu = \alpha / \theta$
- $\sigma = \text{std}(\varepsilon_t)$

### 5.3 Half-Life of Mean Reversion

$$\text{Half-Life} = \frac{\ln(2)}{\theta}$$

Interpretation: Expected time for a deviation to reduce by 50%.

**Trading implications:**
- Short half-life (< 10 bars): High-frequency strategies viable
- Medium half-life (10-50 bars): Standard holding periods
- Long half-life (> 50 bars): May require excessive capital

---

## 6. Signal Generation

### 6.1 Signal Strength Mapping

| Z-Score Range | Signal Strength | Action |
|---------------|-----------------|--------|
| $|Z| < 1.5$ | NEUTRAL | No action |
| $1.5 \leq |Z| < 2.0$ | WEAK | Consider entry |
| $2.0 \leq |Z| < 2.5$ | MODERATE | Standard entry |
| $2.5 \leq |Z| < 3.0$ | STRONG | Conviction entry |
| $|Z| \geq 3.0$ | VERY_STRONG | Maximum conviction |

### 6.2 Direction Determination

$$\text{Direction} = \begin{cases}
\text{LONG_SPREAD} & \text{if } Z < -threshold \\
\text{FLAT} & \text{if } |Z| \leq threshold \\
\text{SHORT_SPREAD} & \text{if } Z > threshold
\end{cases}$$

**Rationale:**
- Negative Z → Spread below mean → Expect reversion up → Long
- Positive Z → Spread above mean → Expect reversion down → Short

### 6.3 Position Sizing Formula

$$\text{Size} = \text{BaseSize}(\text{Strength}) \times \text{VolAdj} \times \text{HLAdj}$$

Where:
- $\text{BaseSize}$: 0.25 (WEAK) to 1.0 (VERY_STRONG)
- $\text{VolAdj} = \min\left(\frac{\sigma_{target}}{\sigma_{spread}}, 1.5\right)$
- $\text{HLAdj}$: 1.2 (fast) to 0.8 (slow)

### 6.4 Confidence Score

$$\text{Confidence} = \sum_{i} w_i \cdot c_i$$

| Component | Weight | Formula |
|-----------|--------|---------|
| Z-Score | 30% | $\min(|Z|/3, 1)$ |
| Correlation | 25% | $\max(0, (|\rho| - 0.3)/0.7)$ |
| Cointegration | 25% | $1 - \min(p/0.05, 1)$ |
| Half-Life | 10% | 1.0 if 5-30, else scaled |
| Kurtosis | 10% | $1/(1 + |\kappa|/10)$ |

---

## 7. Exit Conditions

### 7.1 Take Profit (Mean Reversion)

$$\text{Exit when: } |Z_t| \leq Z_{exit}$$

Default: $Z_{exit} = 0.5$

### 7.2 Stop Loss (Continued Divergence)

$$\text{Exit when: } |Z_t| \geq Z_{stop}$$

Default: $Z_{stop} = 4.0$

### 7.3 Maximum Holding Period

$$\text{Exit when: } t - t_{entry} \geq T_{max}$$

Default: $T_{max} = 100$ bars

Prevents capital being tied up in non-reverting spreads.

---

## 8. Correlation Analysis

### 8.1 Rolling Correlation

$$\rho_{t} = \frac{\text{Cov}(r_A, r_B)[t-w:t]}{\sigma_A[t-w:t] \cdot \sigma_B[t-w:t]}$$

Where $r_A, r_B$ are log returns.

### 8.2 Eigen Decomposition

For correlation matrix $C$:

$$C = Q \Lambda Q^T$$

Where:
- $\Lambda$: Diagonal matrix of eigenvalues
- $Q$: Orthogonal matrix of eigenvectors

**Application:** Identify principal components driving asset correlations.

### 8.3 Minimum Spanning Tree

Constructs a tree structure from correlation matrix:

$$d_{ij} = \sqrt{2(1 - \rho_{ij})}$$

Useful for visualizing asset relationships and identifying clusters.

---

## 9. Summary of Key Equations

| Concept | Equation |
|---------|----------|
| Spread | $S_t = \log(P_A) - \beta \log(P_B)$ |
| Z-Score | $Z_t = (S_t - \mu_{t-1}) / \sigma_{t-1}$ |
| Hedge Ratio | $\hat{\beta} = \text{Cov}(A,B) / \text{Var}(B)$ |
| OU Process | $dX_t = \theta(\mu - X_t)dt + \sigma dW_t$ |
| Half-Life | $\text{HL} = \ln(2) / \theta$ |
| Position Size | $\text{Size} = f(\text{Strength}, \text{Vol}, \text{HL})$ |

---

## 10. Implementation Reference

All mathematical functions are implemented in:

- `models/math_utils.py` - Core statistical tests and estimators
- `models/spread_model.py` - Spread construction and normalization
- `models/signal_generator.py` - Signal generation and aggregation

### Key Classes:

```python
# Statistical tests
StatisticalTests.engle_granger(series_y, series_x)
StatisticalTests.augmented_dickey_fuller(series)
StatisticalTests.hurst_exponent(series)
StatisticalTests.ornstein_uhlenbeck_params(series)

# Hedge ratio estimation
HedgeRatioEstimator.ols(y, x)
HedgeRatioEstimator.rolling_ols(y, x, window=60)
HedgeRatioEstimator.ewm(y, x, span=60)

# Spread normalization
SpreadNormalizer.rolling_zscore(spread, window=252)
SpreadNormalizer.ewm_zscore(spread, span=252)

# Signal generation
SignalGenerator(config).generate_signal(...)
```

---

This mathematical framework provides the foundation for the trading strategy and RLHF optimization in subsequent steps.
