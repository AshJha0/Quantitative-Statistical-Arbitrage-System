# STEP 6: Dashboard Implementation

Complete documentation for the Obsidian-style Quant Terminal dashboard.

---

## 1. Overview

The dashboard is a **Streamlit-based** trading terminal with a dark theme, modular panels, and graph-like navigation inspired by Obsidian.

### Features

- **Dark Theme**: Professional quant terminal aesthetic
- **Modular Panels**: Independent, composable components
- **Real-time Updates**: Live mode with configurable refresh
- **Interactive Charts**: Plotly-based visualizations
- **Multi-pair Support**: Monitor multiple spreads simultaneously

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     QUANT TERMINAL DASHBOARD                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  HEADER: "Quant Terminal" | System Status                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────┐ │
│  │  TOTAL P&L  │  │   SHARPE    │  │  WIN RATE   │  │ MAX DD │ │
│  │  $152,340   │  │    1.76     │  │   62.3%     │  │  8.3%  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────┘ │
│                                                                  │
│  ┌─────────────────────────────┐  ┌──────────────────────────┐  │
│  │     EQUITY CURVE            │  │   OPEN POSITIONS         │  │
│  │     (Plotly Chart)          │  │   ┌──────────────────┐   │  │
│  │                             │  │   │ Pair  │ Dir │ PnL│   │  │
│  │  ─────────────────────      │  │   ├───────┼─────┼────│   │  │
│  │  ╱╲    ╱╲                   │  │   │ XAU/X │ LONG│+$1K│   │  │
│  │ ╱  ╲──╱  ╲                  │  │   │ AGUSD │     │    │   │  │
│  └─────────────────────────────┘  └──────────────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────┐  ┌──────────────────────────┐  │
│  │   LATEST SIGNALS            │  │  P&L ATTRIBUTION         │  │
│  │   📈 XAUUSD/XAGUSD          │  │    (Bar Chart)           │  │
│  │   Z-Score: -2.50            │  │                          │  │
│  │   LONG_SPREAD               │  │  ██ XAU/XAG  +$50K       │  │
│  │   Confidence: 0.85          │  │  ██ EUR/GBP  +$30K       │  │
│  └─────────────────────────────┘  └──────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Dashboard Components

### 3.1 DashboardConfig

```python
@dataclass
class DashboardConfig:
    theme: str = 'dark'
    primary_color: str = '#00D4AA'      # Teal
    secondary_color: str = '#7B61FF'    # Purple
    up_color: str = '#00D4AA'           # Green for profit/long
    down_color: str = '#FF4B4B'         # Red for loss/short
    background_color: str = '#0E1117'   # Dark background
    grid_color: str = '#262730'         # Grid lines
    font_color: str = '#FAFAFA'         # Light text
```

### 3.2 QuantDashboard Class

Main dashboard orchestrator:

```python
from dashboard.app import QuantDashboard, create_dashboard

# Create dashboard
dashboard = create_dashboard()

# Or use Streamlit directly
streamlit run dashboard/app.py
```

---

## 4. Panel Implementations

### 4.1 Header Panel

```python
def render_header(self, title: str, subtitle: str = ""):
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(title)
        if subtitle:
            st.markdown(f"*{subtitle}*")
    with col2:
        st.markdown("<div style='text-align: right; color: #888;'>🟢 System Online</div>", unsafe_allow_html=True)
```

**Usage:**
```python
dashboard.render_header("Quant Terminal", "Statistical Arbitrage System")
```

### 4.2 Metric Cards

```python
def render_metric_cards(self, metrics: Dict[str, float]):
    cols = st.columns(4)
    
    metric_configs = [
        ('Total P&L', '$', 0, True),
        ('Sharpe Ratio', '', 2, False),
        ('Win Rate', '%', 2, False),
        ('Max Drawdown', '%', 2, True)
    ]
    
    for i, (key, unit, decimals, invert) in enumerate(metric_configs):
        value = metrics[key]
        display_value = f"{unit}{value:,.{decimals}f}" if unit == '$' else f"{value:.{decimals}f}{unit}"
        
        with cols[i]:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{display_value}</div>
                <div class='metric-label'>{key}</div>
            </div>
            """, unsafe_allow_html=True)
```

**CSS Styling:**
```css
.metric-card {
    background: linear-gradient(135deg, #1E2330 0%, #141821 100%);
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #262730;
    margin: 10px 0;
}
.metric-value {
    font-size: 32px;
    font-weight: bold;
    color: #00D4AA;
}
.metric-label {
    font-size: 14px;
    color: #888888;
    margin-top: 5px;
}
```

### 4.3 Equity Curve Chart

```python
def render_equity_chart(self, equity_curve: pd.Series, drawdown_curve: pd.Series):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=('Equity Curve', 'Drawdown')
    )
    
    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            name='Equity',
            line=dict(color=self.config.primary_color, width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 170, 0.1)'
        ),
        row=1, col=1
    )
    
    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=drawdown_curve.index,
            y=-drawdown_curve.values,
            name='Drawdown',
            line=dict(color=self.config.down_color, width=1.5),
            fill='tozeroy',
            fillcolor='rgba(255, 75, 75, 0.2)'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        plot_bgcolor=self.config.background_color,
        paper_bgcolor=self.config.background_color
    )
    
    st.plotly_chart(fig, use_container_width=True)
```

### 4.4 Spread Chart with Z-Score

```python
def render_spread_chart(
    self,
    spread: pd.Series,
    z_score: pd.Series,
    entry_signals: List[int],
    exit_signals: List[int],
    pair_name: str
):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.4],
        subplot_titles=(f'Spread: {pair_name}', 'Z-Score')
    )
    
    # Spread
    fig.add_trace(
        go.Scatter(
            x=spread.index,
            y=spread.values,
            name='Spread',
            line=dict(color=self.config.secondary_color, width=2)
        ),
        row=1, col=1
    )
    
    # Z-score with bands
    fig.add_trace(
        go.Scatter(
            x=z_score.index,
            y=z_score.values,
            name='Z-Score',
            line=dict(color=self.config.primary_color, width=2)
        ),
        row=2, col=1
    )
    
    # Entry/Exit bands
    fig.add_hline(y=2.0, line_dash="dash", line_color=self.config.down_color, row=2, col=1)
    fig.add_hline(y=-2.0, line_dash="dash", line_color=self.config.up_color, row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#888888", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
```

### 4.5 Positions Table

```python
def render_positions_table(self, positions: List[Dict]):
    if not positions:
        st.info("No open positions")
        return
    
    df = pd.DataFrame(positions)
    df = df[['Pair', 'Direction', 'Size', 'Entry Z', 'Current Z', 'Unrealized P&L', 'Bars Held']]
    
    # Color coding
    def color_direction(val):
        if val == 'LONG':
            return f'color: {self.config.up_color}'
        elif val == 'SHORT':
            return f'color: {self.config.down_color}'
        return ''
    
    def color_pnl(val):
        if val > 0:
            return f'color: {self.config.up_color}'
        elif val < 0:
            return f'color: {self.config.down_color}'
        return ''
    
    st.dataframe(
        df.style
        .applymap(color_direction, subset=['Direction'])
        .applymap(color_pnl, subset=['Unrealized P&L']),
        use_container_width=True
    )
```

### 4.6 Signals Panel

```python
def render_signals_panel(self, signals: List[Dict]):
    st.subheader("📡 Latest Signals")
    
    if not signals:
        st.markdown("No recent signals")
        return
    
    for signal in signals[-10:]:
        signal_type = signal.get('signal', 'NEUTRAL')
        pair = signal.get('pair', '')
        z_score = signal.get('z_score', 0)
        
        if signal_type > 0:
            signal_class = 'signal-long'
            signal_icon = '📈'
        elif signal_type < 0:
            signal_class = 'signal-short'
            signal_icon = '📉'
        else:
            signal_class = 'signal-neutral'
            signal_icon = '⏸'
        
        st.markdown(f"""
        <div style='padding: 10px; margin: 5px 0; background: #1E2330; border-radius: 8px;'>
            {signal_icon} <span class='{signal_class}'>{pair}</span> | Z-Score: {z_score:.2f}
        </div>
        """, unsafe_allow_html=True)
```

### 4.7 Performance Attribution

```python
def render_performance_attribution(self, trades: List[Dict]):
    if not trades:
        return
    
    # Group by pair
    pair_pnl = {}
    for trade in trades:
        pair = trade.get('pair', 'Unknown')
        pnl = trade.get('pnl', 0)
        pair_pnl[pair] = pair_pnl.get(pair, 0) + pnl
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(pair_pnl.keys()),
            y=list(pair_pnl.values()),
            marker_color=[
                self.config.up_color if v > 0 else self.config.down_color
                for v in pair_pnl.values()
            ]
        )
    ])
    
    fig.update_layout(
        title='P&L by Pair',
        plot_bgcolor=self.config.background_color,
        paper_bgcolor=self.config.background_color
    )
    
    st.plotly_chart(fig, use_container_width=True)
```

### 4.8 Correlation Matrix

```python
def render_correlation_matrix(self, returns: pd.DataFrame):
    corr = returns.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr.values.round(2),
        texttemplate='%{text}'
    ))
    
    fig.update_layout(
        title='Asset Correlation Matrix',
        plot_bgcolor=self.config.background_color,
        paper_bgcolor=self.config.background_color
    )
    
    st.plotly_chart(fig, use_container_width=True)
```

---

## 5. Sidebar Controls

```python
def render_sidebar(self, config_values: Dict):
    with st.sidebar:
        st.title("⚙️ Controls")
        
        st.subheader("Strategy Parameters")
        
        entry_threshold = st.slider(
            "Entry Threshold (Z-Score)",
            min_value=1.0,
            max_value=4.0,
            value=config_values.get('entry_threshold', 2.0),
            step=0.1
        )
        
        exit_threshold = st.slider(
            "Exit Threshold (Z-Score)",
            min_value=0.0,
            max_value=2.0,
            value=config_values.get('exit_threshold', 0.5),
            step=0.1
        )
        
        position_size = st.slider(
            "Position Size (%)",
            min_value=5,
            max_value=50,
            value=int(config_values.get('position_size', 25) * 100),
            step=5
        )
        
        st.divider()
        
        st.subheader("Display Options")
        show_live = st.checkbox("Live Mode", value=True)
        refresh_rate = st.selectbox("Refresh Rate", ["1s", "5s", "10s", "30s"])
        
        st.divider()
        
        st.subheader("Data Source")
        data_source = st.selectbox("Source", ["Historical", "Live Feed", "Paper Trading"])
```

---

## 6. Running the Dashboard

### 6.1 Basic Usage

```bash
# Start Streamlit server
streamlit run dashboard/app.py

# Access in browser
http://localhost:8501
```

### 6.2 With Custom Port

```bash
streamlit run dashboard/app.py --server.port 8502
```

### 6.3 Remote Access

```bash
streamlit run dashboard/app.py --server.address 0.0.0.0
```

### 6.4 Loading Backtest Results

```python
# In dashboard/app.py
import pandas as pd
import json
from pathlib import Path

def load_backtest_results(output_dir: str = "output"):
    output_path = Path(output_dir)
    
    equity_curve = pd.read_csv(
        output_path / 'equity_curve.csv',
        index_col=0,
        parse_dates=True
    )
    
    with open(output_path / 'summary.json', 'r') as f:
        summary = json.load(f)
    
    trades = pd.read_csv(output_path / 'trades.csv')
    
    return equity_curve, summary, trades

# Use in dashboard
equity_curve, summary, trades = load_backtest_results()
```

---

## 7. Full Dashboard Example

```python
from dashboard.app import run_dashboard
import pandas as pd
import json

# Load results
equity_curve = pd.read_csv('output/equity_curve.csv', index_col=0, parse_dates=True)
drawdown_curve = pd.read_csv('output/drawdown_curve.csv', index_col=0, parse_dates=True)

with open('output/summary.json', 'r') as f:
    summary = json.load(f)

metrics = summary['metrics']
trades = pd.read_csv('output/trades.csv')

# Prepare positions and signals
positions = []  # Would come from live system
signals = []    # Would come from signal generator

# Run dashboard
run_dashboard(
    metrics=metrics,
    equity_curve=equity_curve['equity'],
    drawdown_curve=drawdown_curve['drawdown'],
    positions=positions,
    signals=signals,
    trades=trades.to_dict('records')
)
```

---

## 8. Customization

### 8.1 Adding New Panels

```python
def render_risk_panel(self, risk_metrics: Dict):
    st.subheader("⚠️ Risk Metrics")
    
    cols = st.columns(3)
    
    with cols[0]:
        st.metric("VaR (1D)", f"{risk_metrics.get('var_1d', 0):.2%}")
    
    with cols[1]:
        st.metric("Gross Exposure", f"{risk_metrics.get('gross_exposure', 0):.2f}")
    
    with cols[2]:
        st.metric("Net Exposure", f"{risk_metrics.get('net_exposure', 0):.2f}")
```

### 8.2 Custom Color Schemes

```python
# Light theme
light_config = DashboardConfig(
    theme='light',
    primary_color='#0066CC',
    background_color='#FFFFFF',
    grid_color='#E0E0E0',
    font_color='#333333'
)

# Blue theme
blue_config = DashboardConfig(
    theme='blue',
    primary_color='#0066FF',
    secondary_color='#00CCFF',
    up_color='#00AA00',
    down_color='#FF0000',
    background_color='#0A0E27',
    grid_color='#1A1E3E',
    font_color='#EEEEEE'
)
```

### 8.3 Adding Alerts

```python
def render_alerts(self, breaches: List[str]):
    if breaches:
        for breach in breaches:
            st.error(f"⚠️ {breach}")
    else:
        st.success("✅ All risk limits OK")
```

---

## 9. Live Mode Integration

### 9.1 Real-time Data Updates

```python
import time
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 5 seconds
count = st_autorefresh(interval=5000, limit=100)

# Fetch latest data
latest_data = fetch_latest_market_data()
latest_signals = generate_signals(latest_data)

# Update dashboard
render_metric_cards(latest_metrics)
render_positions_table(latest_positions)
```

### 9.2 WebSocket Integration

```python
import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    st.session_state['latest_price'] = data['price']
    st.session_state['latest_signal'] = data['signal']

ws = websocket.WebSocketApp(
    "ws://localhost:8765",
    on_message=on_message
)
```

---

## 10. Export and Reporting

### 10.1 Export Dashboard Data

```python
def export_dashboard_data(output_dir: str = "output"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Export equity curve
    equity_curve.to_csv(output_path / 'dashboard_equity.csv')
    
    # Export positions
    pd.DataFrame(positions).to_csv(output_path / 'dashboard_positions.csv')
    
    # Export signals
    pd.DataFrame(signals).to_csv(output_path / 'dashboard_signals.csv')
    
    # Export metrics as JSON
    with open(output_path / 'dashboard_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
```

### 10.2 Screenshot Functionality

```python
import streamlit as st
from streamlit_screencapture import st_screencapture

# Add capture button
if st.button("📸 Capture Dashboard"):
    st_screencapture(
        selector=".main",
        filename="dashboard_capture.png"
    )
```

---

## 11. Troubleshooting

### Issue: Charts Not Rendering

**Solution**: Ensure Plotly is properly installed
```bash
pip install plotly>=5.18.0
```

### Issue: Slow Performance

**Solutions**:
- Reduce data points: `equity_curve = equity_curve[::10]`
- Use caching: `@st.cache_data`
- Limit refresh rate

### Issue: Dark Theme Not Applied

**Solution**: Ensure CSS is loaded before components
```python
st.markdown("""<style>...</style>""", unsafe_allow_html=True)
```

---

This dashboard provides a professional, Obsidian-inspired interface for monitoring your statistical arbitrage system with real-time updates and comprehensive visualizations.
