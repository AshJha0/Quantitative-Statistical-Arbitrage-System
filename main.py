"""
Quantitative Statistical Arbitrage System
Main entry point and demonstration script.

This system implements relative value trading using spread models
between correlated assets, enhanced with reinforcement learning (RLHF).

Usage:
    python main.py                    # Run complete demo
    python main.py --no-rl           # Skip RL training
    python main.py --timesteps 100000 # Custom RL steps
"""

import argparse
import logging
from typing import Dict

from system import QuantSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_demo(
    train_rl: bool = True,
    rl_timesteps: int = 50000
) -> Dict:
    """
    Run complete system demonstration.

    Args:
        train_rl: Whether to train RL agent
        rl_timesteps: Number of RL training steps

    Returns:
        Dictionary with all results
    """
    logger.info("=" * 70)
    logger.info("QUANTITATIVE STATISTICAL ARBITRAGE SYSTEM")
    logger.info("Relative Value Trading with RLHF Optimization")
    logger.info("=" * 70)

    # Create system instance
    system = QuantSystem(
        data_dir="data",
        model_dir="models",
        output_dir="output",
        initial_capital=1_000_000
    )

    # Run complete pipeline
    results = system.run_complete_pipeline(
        symbols=['XAUUSD', 'XAGUSD', 'EURUSD', 'GBPUSD'],
        pairs=[('XAUUSD', 'XAGUSD'), ('EURUSD', 'GBPUSD')],
        n_points=10000,
        train_rl=train_rl,
        rl_timesteps=rl_timesteps,
        export_results=True
    )

    # Print summary
    print("\n" + "=" * 70)
    print("SYSTEM DEMO COMPLETE")
    print("=" * 70)
    print()
    print(f"Pairs traded:        {len(system._pairs)}")
    print(f"Total trades:        {results['n_trades']}")
    print(f"Signals generated:   {results['n_signals']}")
    print()
    print("Key Metrics:")
    metrics = results['backtest_metrics']
    print(f"  Total Return:      {metrics.get('total_return', 0):.2%}")
    print(f"  Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"  Max Drawdown:      {metrics.get('max_drawdown', 0):.2%}")
    print(f"  Win Rate:          {metrics.get('win_rate', 0):.2%}")
    print()

    if train_rl and results.get('rl_metrics'):
        print("RL Agent Metrics:")
        rl = results['rl_metrics']
        print(f"  Mean Reward:       {rl.get('mean_reward', 0):.4f}")
        print(f"  Sharpe Ratio:      {rl.get('sharpe_ratio', 0):.4f}")
        print(f"  Win Rate:          {rl.get('win_rate', 0):.4f}")
        print()

    print("Output Files:")
    print("  - output/equity_curve.csv")
    print("  - output/drawdown_curve.csv")
    print("  - output/trades.csv")
    print("  - output/metrics.csv")
    print("  - output/summary.json")
    print()
    print("Next Steps:")
    print("  - View dashboard: streamlit run dashboard/app.py")
    print("  - Add custom data: place CSV files in data/")
    print("  - Tune parameters: edit config/settings.py")
    print("=" * 70)

    return results


def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Statistical Arbitrage System with RLHF'
    )
    parser.add_argument(
        '--no-rl',
        action='store_true',
        help='Skip RL agent training'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=50000,
        help='Number of RL training timesteps'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=1_000_000,
        help='Initial capital'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run demo
    results = run_demo(
        train_rl=not args.no_rl,
        rl_timesteps=args.timesteps
    )

    return results


if __name__ == "__main__":
    main()
