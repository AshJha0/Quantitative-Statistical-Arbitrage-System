"""
Dashboard Module - Obsidian-Style Quant Terminal
Modern, dark-themed trading dashboard with modular panels.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import streamlit as st


@dataclass
class DashboardConfig:
    """Configuration for dashboard appearance."""
    theme: str = 'dark'
    primary_color: str = '#00D4AA'  # Teal
    secondary_color: str = '#7B61FF'  # Purple
    up_color: str = '#00D4AA'
    down_color: str = '#FF4B4B'
    background_color: str = '#0E1117'
    grid_color: str = '#262730'
    font_color: str = '#FAFAFA'


class QuantDashboard:
    """
    Main dashboard class for the quant terminal.
    Provides modular panels for monitoring strategies and performance.
    """

    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        self._setup_streamlit_page()

    def _setup_streamlit_page(self):
        """Configure Streamlit page."""
        st.set_page_config(
            page_title="Quant Terminal - Statistical Arbitrage",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Custom CSS for Obsidian-like dark theme
        st.markdown("""
        <style>
        .main {
            background-color: #0E1117;
        }
        .stApp {
            background-color: #0E1117;
        }
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
        .signal-long {
            color: #00D4AA;
            font-weight: bold;
        }
        .signal-short {
            color: #FF4B4B;
            font-weight: bold;
        }
        .signal-neutral {
            color: #888888;
        }
        </style>
        """, unsafe_allow_html=True)

    def render_header(self, title: str, subtitle: str = ""):
        """Render dashboard header."""
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title(title)
            if subtitle:
                st.markdown(f"*{subtitle}*")
        with col2:
            st.markdown(f"<div style='text-align: right; color: #888;'>🟢 System Online</div>", unsafe_allow_html=True)

    def render_metric_cards(self, metrics: Dict[str, float]) -> None:
        """Render key metrics as cards."""
        cols = st.columns(4)

        metric_configs = [
            ('Total P&L', '$', 0, True),
            ('Sharpe Ratio', '', 2, False),
            ('Win Rate', '%', 2, False),
            ('Max Drawdown', '%', 2, True)
        ]

        for i, (key, unit, decimals, invert) in enumerate(metric_configs):
            if key in metrics:
                value = metrics[key]
                if invert and key != 'Total P&L':
                    value = -value
                display_value = f"{unit}{value:,.{decimals}f}" if unit == '$' else f"{value:.{decimals}f}{unit}"

                with cols[i]:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{display_value}</div>
                        <div class='metric-label'>{key}</div>
                    </div>
                    """, unsafe_allow_html=True)

    def render_equity_chart(self, equity_curve: pd.Series, drawdown_curve: pd.Series) -> None:
        """Render equity curve with drawdown."""
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
            paper_bgcolor=self.config.background_color,
            font=dict(color=self.config.font_color),
            margin=dict(l=50, r=50, t=50, b=50)
        )

        fig.update_xaxes(showgrid=True, gridcolor=self.config.grid_color)
        fig.update_yaxes(showgrid=True, gridcolor=self.config.grid_color)

        st.plotly_chart(fig, use_container_width=True)

    def render_spread_chart(
        self,
        spread: pd.Series,
        z_score: pd.Series,
        entry_signals: List[int],
        exit_signals: List[int],
        pair_name: str
    ) -> None:
        """Render spread with Z-score and signals."""
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

        # Signal markers
        if entry_signals:
            fig.add_trace(
                go.Scatter(
                    x=[z_score.index[i] for i in entry_signals],
                    y=[z_score.values[i] for i in entry_signals],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=12, color=self.config.up_color),
                    name='Entry Signal'
                ),
                row=2, col=1
            )

        fig.update_layout(
            height=600,
            showlegend=True,
            plot_bgcolor=self.config.background_color,
            paper_bgcolor=self.config.background_color,
            font=dict(color=self.config.font_color)
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_positions_table(self, positions: List[Dict]) -> None:
        """Render open positions table."""
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

    def render_signals_panel(self, signals: List[Dict]) -> None:
        """Render latest trading signals."""
        st.subheader("📡 Latest Signals")

        if not signals:
            st.markdown("No recent signals")
            return

        for signal in signals[-10:]:
            signal_type = signal.get('signal', 'NEUTRAL')
            pair = signal.get('pair', '')
            z_score = signal.get('z_score', 0)
            timestamp = signal.get('timestamp', '')

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
            <div style='padding: 10px; margin: 5px 0; background: #1E2330; border-radius: 8px; border-left: 4px solid {"#00D4AA" if signal_type > 0 else "#FF4B4B" if signal_type < 0 else "#888"};'>
                {signal_icon} <span class='{signal_class}'>{pair}</span> | Z-Score: {z_score:.2f} | {timestamp}
            </div>
            """, unsafe_allow_html=True)

    def render_performance_attribution(self, trades: List[Dict]) -> None:
        """Render performance attribution chart."""
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
            paper_bgcolor=self.config.background_color,
            font=dict(color=self.config.font_color),
            showlegend=False
        )

        fig.update_xaxes(showgrid=True, gridcolor=self.config.grid_color)
        fig.update_yaxes(showgrid=True, gridcolor=self.config.grid_color)

        st.plotly_chart(fig, use_container_width=True)

    def render_correlation_matrix(self, returns: pd.DataFrame) -> None:
        """Render correlation heatmap."""
        corr = returns.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))

        fig.update_layout(
            title='Asset Correlation Matrix',
            plot_bgcolor=self.config.background_color,
            paper_bgcolor=self.config.background_color,
            font=dict(color=self.config.font_color)
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_sidebar(self, config_values: Dict) -> None:
        """Render sidebar with controls."""
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
            refresh_rate = st.selectbox(
                "Refresh Rate",
                ["1s", "5s", "10s", "30s"],
                index=1
            )

            st.divider()

            st.subheader("Data Source")
            data_source = st.selectbox(
                "Source",
                ["Historical", "Live Feed", "Paper Trading"]
            )


def create_dashboard() -> QuantDashboard:
    """Factory function to create dashboard instance."""
    return QuantDashboard()


def run_dashboard(
    metrics: Dict,
    equity_curve: pd.Series,
    drawdown_curve: pd.Series,
    positions: List[Dict],
    signals: List[Dict],
    trades: List[Dict]
):
    """Run the complete dashboard."""
    dashboard = create_dashboard()

    # Header
    dashboard.render_header("Quant Terminal", "Statistical Arbitrage System")

    # Metric cards
    dashboard.render_metric_cards(metrics)

    # Main charts
    col1, col2 = st.columns([2, 1])

    with col1:
        dashboard.render_equity_chart(equity_curve, drawdown_curve)

    with col2:
        dashboard.render_positions_table(positions)

    # Signals and attribution
    col3, col4 = st.columns([1, 1])

    with col3:
        dashboard.render_signals_panel(signals)

    with col4:
        dashboard.render_performance_attribution(trades)

    # Sidebar controls
    dashboard.render_sidebar({
        'entry_threshold': 2.0,
        'exit_threshold': 0.5,
        'position_size': 0.25
    })
